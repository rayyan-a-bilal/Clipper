"""
SQLite database interface for clipper.
"""

import sqlite3
from datetime import datetime
from typing import Any

from .config import Config


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

        # Enable WAL mode
        cursor.execute("PRAGMA journal_mode=WAL")

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
                cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_content_hash ON jobs(content_hash)")
                self.set_schema_version(2)
            except Exception:
                # Migration already applied or failed, check
                pass

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
        """Set current schema version."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            timestamp = datetime.now().isoformat()
            cursor.execute(
                "INSERT OR REPLACE INTO schema_migrations (version, applied_at) VALUES (?, ?)",
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

    def create_job(self, job_id: str, source_path: str, client: str, niche: str) -> bool:
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
            cursor.execute("SELECT * FROM stages WHERE job_id = ? ORDER BY name", (job_id,))
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

    def create_stage(self, job_id: str, stage_name: str, state: str = "pending") -> bool:
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

    def update_stage(self, job_id: str, stage_name: str, state: str,
                     artifact_path: str | None = None, error: str | None = None) -> bool:
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

    def create_clip(self, clip_id: str, job_id: str, plan_json: str, status: str = "pending_qc") -> bool:
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
            cursor.execute("SELECT * FROM clips WHERE job_id = ? ORDER BY created_at DESC", (job_id,))
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def create_candidate(self, cand_id: str, job_id: str, candidate_json: str, score_json: str) -> bool:
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
