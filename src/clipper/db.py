"""
SQLite database interface for clipper.
"""

import sqlite3
from datetime import datetime
from typing import Any

from .config import Config
from .models import JobReport


class Database:
    """SQLite database manager for clipper."""

    def __init__(self, config: Config):
        self.config = config
        self.db_path = self.config.resolved_data_dir / "clipper.db"
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn

    def init_db(self):
        """Initialize the database with required tables."""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Create schema migration table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT
            )
        """)

        # Create jobs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                source_path TEXT,
                client TEXT,
                niche TEXT,
                state TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)

        # Create stages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stages (
                job_id TEXT,
                name TEXT,
                state TEXT,
                artifact_path TEXT,
                error TEXT,
                started_at TEXT,
                finished_at TEXT,
                PRIMARY KEY(job_id, name)
            )
        """)

        # Create clips table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS clips (
                clip_id TEXT PRIMARY KEY,
                job_id TEXT,
                plan_json TEXT,
                status TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)

        # Create candidates table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS candidates (
                cand_id TEXT PRIMARY KEY,
                job_id TEXT,
                candidate_json TEXT,
                score_json TEXT
            )
        """)

        # Create posts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS posts (
                id TEXT PRIMARY KEY,
                clip_id TEXT,
                platform TEXT,
                state TEXT,
                payload_json TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)

        # Enable WAL mode (fetch the single result row so the statement is
        # complete — an unfetched statement blocks the final commit)
        cursor.execute("PRAGMA journal_mode=WAL").fetchone()

        # Check migration version and apply updates if needed
        self._apply_migrations(conn)

        conn.commit()
        conn.close()

    def _apply_migrations(self, conn):
        """Apply database migrations."""
        cursor = conn.cursor()

        # Get current schema version
        try:
            cursor.execute("SELECT MAX(version) FROM schema_migrations")
            result = cursor.fetchone()[0]
            version = result if result is not None else 0
        except Exception:
            version = 0

        # Migration v2: Add content_hash column and unique index (M1)
        if version < 2:
            try:
                cursor.execute("ALTER TABLE jobs ADD COLUMN content_hash TEXT")
                cursor.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_content_hash "
                    "ON jobs(content_hash)"
                )
            except Exception:
                # Column/index already present (e.g. DB built by the old
                # buggy init) — the v2 change is effectively applied.
                pass
            # Record on THIS connection: no second writer, no WAL busy
            # timeout (docs/spec/cards/m0-db-init-timeout.md).
            self._record_schema_version(cursor, 2)

        # Migration v3: per-job worker report table (M8a, docs/spec/m8.md)
        if version < 3:
            try:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS job_reports (
                        job_id TEXT PRIMARY KEY,
                        report_json TEXT,
                        updated_at TEXT
                    )
                    """
                )
            except Exception:
                pass
            self._record_schema_version(cursor, 3)

    def _record_schema_version(self, cursor, version: int) -> None:
        """Record a migration on the caller's open cursor/connection.

        ``init_db`` commits schema + version rows together at the end, so
        no second connection (and no WAL write-lock contention) is ever
        needed — see docs/spec/cards/m0-db-init-timeout.md.
        """
        cursor.execute(
            "INSERT OR REPLACE INTO schema_migrations "
            "(version, applied_at) VALUES (?, ?)",
            (version, datetime.now().isoformat())
        )

    def get_schema_version(self) -> int | None:
        """Get current schema version."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT MAX(version) FROM schema_migrations")
            result = cursor.fetchone()[0]
            return result if result is not None else 0
        except Exception:
            return 0
        finally:
            conn.close()

    def set_schema_version(self, version: int):
        """Set current schema version.

        Kept for API stability; ``init_db`` records migrations on its own
        connection (see ``_record_schema_version``).
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            timestamp = datetime.now().isoformat()
            cursor.execute(
                "INSERT OR REPLACE INTO schema_migrations "
                "(version, applied_at) VALUES (?, ?)",
                (version, timestamp)
            )
            conn.commit()
        finally:
            conn.close()

    def get_all_jobs(self) -> list[dict[str, Any]]:
        """Get all jobs."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM jobs ORDER BY created_at DESC")
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Get a specific job."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def create_job(
        self, job_id: str, source_path: str, client: str, niche: str
    ) -> bool:
        """Create a new job."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            timestamp = datetime.now().isoformat()
            cursor.execute("""
                INSERT OR IGNORE INTO jobs
                (job_id, source_path, client, niche, state, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'queued', ?, ?)
            """, (job_id, source_path, client, niche, timestamp, timestamp))
            conn.commit()
            return cursor.rowcount > 0
        except Exception:
            conn.rollback()
            return False
        finally:
            conn.close()

    def update_job_state(self, job_id: str, state: str) -> bool:
        """Update job state."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            timestamp = datetime.now().isoformat()
            cursor.execute("""
                UPDATE jobs
                SET state = ?, updated_at = ?
                WHERE job_id = ?
            """, (state, timestamp, job_id))
            conn.commit()
            return cursor.rowcount > 0
        except Exception:
            conn.rollback()
            return False
        finally:
            conn.close()

    def get_all_stages_for_job(self, job_id: str) -> list[dict[str, Any]]:
        """Get all stages for a specific job."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT * FROM stages WHERE job_id = ? ORDER BY name", (job_id,)
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_stage(self, job_id: str, stage_name: str) -> dict[str, Any] | None:
        """Get a specific stage for a job."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT * FROM stages WHERE job_id = ? AND name = ?",
                (job_id, stage_name)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def create_stage(
        self, job_id: str, stage_name: str, state: str = "pending"
    ) -> bool:
        """Create a new stage for a job."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            timestamp = datetime.now().isoformat()
            cursor.execute("""
                INSERT OR IGNORE INTO stages
                (job_id, name, state, started_at, finished_at)
                VALUES (?, ?, ?, ?, ?)
            """, (job_id, stage_name, state, timestamp, None))
            conn.commit()
            return cursor.rowcount > 0
        except Exception:
            conn.rollback()
            return False
        finally:
            conn.close()

    def update_stage(
        self,
        job_id: str,
        stage_name: str,
        state: str,
        artifact_path: str | None = None,
        error: str | None = None,
    ) -> bool:
        """Update a stage state."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            timestamp = datetime.now().isoformat()
            if state == "completed":
                cursor.execute("""
                    UPDATE stages
                    SET state = ?, artifact_path = ?, finished_at = ?
                    WHERE job_id = ? AND name = ?
                """, (state, artifact_path, timestamp, job_id, stage_name))
            elif state == "failed":
                cursor.execute("""
                    UPDATE stages
                    SET state = ?, error = ?, finished_at = ?
                    WHERE job_id = ? AND name = ?
                """, (state, error, timestamp, job_id, stage_name))
            else:
                cursor.execute("""
                    UPDATE stages
                    SET state = ?
                    WHERE job_id = ? AND name = ?
                """, (state, job_id, stage_name))
            conn.commit()
            return cursor.rowcount > 0
        except Exception:
            conn.rollback()
            return False
        finally:
            conn.close()

    def create_clip(
        self, clip_id: str, job_id: str, plan_json: str, status: str = "pending_qc"
    ) -> bool:
        """Create a new clip."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            timestamp = datetime.now().isoformat()
            cursor.execute("""
                INSERT OR IGNORE INTO clips
                (clip_id, job_id, plan_json, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (clip_id, job_id, plan_json, status, timestamp, timestamp))
            conn.commit()
            return cursor.rowcount > 0
        except Exception:
            conn.rollback()
            return False
        finally:
            conn.close()

    def update_clip_status(self, clip_id: str, status: str) -> bool:
        """Update clip status."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            timestamp = datetime.now().isoformat()
            cursor.execute("""
                UPDATE clips
                SET status = ?, updated_at = ?
                WHERE clip_id = ?
            """, (status, timestamp, clip_id))
            conn.commit()
            return cursor.rowcount > 0
        except Exception:
            conn.rollback()
            return False
        finally:
            conn.close()

    def get_all_clips(self) -> list[dict[str, Any]]:
        """Get all clips."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM clips ORDER BY created_at DESC")
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_clips_for_job(self, job_id: str) -> list[dict[str, Any]]:
        """Get all clips for a specific job."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT * FROM clips WHERE job_id = ? ORDER BY created_at DESC",
                (job_id,),
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def create_candidate(
        self, cand_id: str, job_id: str, candidate_json: str, score_json: str
    ) -> bool:
        """Create a new candidate."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO candidates
                (cand_id, job_id, candidate_json, score_json)
                VALUES (?, ?, ?, ?)
            """, (cand_id, job_id, candidate_json, score_json))
            conn.commit()
            return cursor.rowcount > 0
        except Exception:
            conn.rollback()
            return False
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# M8a worker helpers (module-level, take an open connection — used by the
# M8 worker layer; see docs/spec/m8.md). Additive: the Database class and
# its migrations above are untouched.
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now().isoformat()


def get_job(conn: sqlite3.Connection, job_id: str) -> dict[str, Any] | None:
    """Return one job row as a dict, or None when the job is unknown."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
    row = cursor.fetchone()
    return dict(row) if row else None


def list_jobs(
    conn: sqlite3.Connection, state: str | None = None
) -> list[dict[str, Any]]:
    """List jobs (oldest first); optionally filter by `state`."""
    cursor = conn.cursor()
    if state is None:
        cursor.execute("SELECT * FROM jobs ORDER BY created_at ASC, job_id ASC")
    else:
        cursor.execute(
            "SELECT * FROM jobs WHERE state = ? "
            "ORDER BY created_at ASC, job_id ASC",
            (state,),
        )
    return [dict(row) for row in cursor.fetchall()]


def next_queued_job(conn: sqlite3.Connection) -> dict[str, Any] | None:
    """The oldest `queued` job (one job at a time — M8), or None."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM jobs WHERE state = 'queued' "
        "ORDER BY created_at ASC, job_id ASC LIMIT 1"
    )
    row = cursor.fetchone()
    return dict(row) if row else None


def set_job_state(
    conn: sqlite3.Connection, job_id: str, state: str
) -> None:
    """Set a job's state (queued/running/awaiting_qc/done/failed)."""
    conn.execute(
        "UPDATE jobs SET state = ?, updated_at = ? WHERE job_id = ?",
        (state, _now_iso(), job_id),
    )
    conn.commit()


def set_stage_state(
    conn: sqlite3.Connection,
    job_id: str,
    name: str,
    state: str,
    artifact_path: str | None = None,
    error: str | None = None,
) -> None:
    """Upsert a fine-grained stage row.

    - `state == "running"`: starts the stage (`started_at` set once and
      kept on re-entry), clears `finished_at`/`error`.
    - `state in {"done", "completed"}` (M1/M2 use `completed`): stamps
      `finished_at`, stores `artifact_path` when given.
    - `state == "failed"`: stamps `finished_at`, stores `error`.
    - `state == "pending"`: a clean pending row (all fields reset).
    Omitted `artifact_path`/`error` keep (or clear) the previous values;
    calling it twice with the same arguments is idempotent.
    """
    cursor = conn.cursor()
    cursor.execute(
        "SELECT started_at, finished_at, artifact_path FROM stages "
        "WHERE job_id = ? AND name = ?",
        (job_id, name),
    )
    row = cursor.fetchone()
    prev_started = row[0] if row else None
    prev_artifact = row[2] if row else None

    now = _now_iso()
    if state == "running":
        started_at, finished_at = (prev_started or now), None
        artifact, error = artifact_path or prev_artifact, None
    elif state in ("done", "completed", "failed"):
        started_at, finished_at = prev_started, now
        if state == "failed":
            artifact, error = None, error
        else:
            artifact, error = artifact_path or prev_artifact, None
    else:  # "pending" — clean reset
        started_at, finished_at, artifact, error = None, None, None, None

    cursor.execute(
        "INSERT OR REPLACE INTO stages "
        "(job_id, name, state, artifact_path, error, started_at, finished_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (job_id, name, state, artifact, error, started_at, finished_at),
    )
    conn.commit()


def reset_failed_stages(conn: sqlite3.Connection, job_id: str) -> int:
    """Reset this job's `failed` stage rows to `pending` (M8 `--retry`).

    Only `failed` rows are touched; `done`/`completed`/`running`/`
    pending` rows are left as-is. Returns the number of rows reset.
    """
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE stages "
        "SET state = 'pending', artifact_path = NULL, error = NULL, "
        "started_at = NULL, finished_at = NULL "
        "WHERE job_id = ? AND state = 'failed'",
        (job_id,),
    )
    conn.commit()
    return cursor.rowcount


def write_job_report(
    conn: sqlite3.Connection, job_id: str, report: JobReport
) -> str:
    """Persist a `JobReport` (canonical `indent=2` JSON) for one job.

    Upserts the `job_reports` table (migration v3) and returns the
    JSON string the M8 worker also writes to `data/jobs/<job_id>/
    report.json` (M8a judgment note — the file write itself belongs
    to the worker loop, not to the DB layer).
    """
    report_json = report.model_dump_json(indent=2)
    conn.execute(
        "INSERT OR REPLACE INTO job_reports (job_id, report_json, updated_at) "
        "VALUES (?, ?, ?)",
        (job_id, report_json, _now_iso()),
    )
    conn.commit()
    return report_json
