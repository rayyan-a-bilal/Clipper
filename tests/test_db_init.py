"""Regression tests for the M0 fresh-DB init timing bug.

docs/spec/cards/m0-db-init-timeout.md — fresh `Database(cfg)` init used to
cost ~10s (two 5s sqlite busy timeouts: `set_schema_version` committed from
a SECOND connection while `init_db` held the WAL write lock, swallowed by
`except: pass`; the schema-version row was never recorded). A fresh init
must now be fast (< 5s) and record schema version 3 immediately.
"""

import time

import pytest

from clipper.config import Config, PathsConfig
from clipper.db import Database

EXPECT_VERSION = 3
FRESH_INIT_BUDGET_S = 5.0  # bug measured 10.0s before the fix


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.delenv("CLIPPER_DATA_DIR", raising=False)
    monkeypatch.delenv("CLIPPER_PATHS_DATA_DIR", raising=False)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return Config(paths=PathsConfig(data_dir=str(data_dir)))


def test_fresh_init_is_fast_and_records_schema_version(env):
    t0 = time.monotonic()
    db = Database(env)
    elapsed = time.monotonic() - t0
    assert elapsed < FRESH_INIT_BUDGET_S, (
        f"fresh Database init took {elapsed:.1f}s "
        f"(WAL busy-timeout bug — docs/spec/cards/m0-db-init-timeout.md)"
    )
    assert db.get_schema_version() == EXPECT_VERSION


def test_legacy_pre_fix_db_converges_fast(env):
    """A DB left in the pre-fix state (tables present, versions empty) must
    re-init fast and converge to the recorded version — not pay 5s timeouts."""
    first = Database(env)
    conn = first.get_connection()
    conn.execute("DELETE FROM schema_migrations")
    conn.commit()
    assert first.get_schema_version() == 0
    conn.close()

    t0 = time.monotonic()
    db = Database(env)
    elapsed = time.monotonic() - t0
    assert elapsed < FRESH_INIT_BUDGET_S
    assert db.get_schema_version() == EXPECT_VERSION

    # All tables still present and usable
    conn = db.get_connection()
    try:
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        for expected in (
            "schema_migrations",
            "jobs",
            "stages",
            "clips",
            "candidates",
            "posts",
            "job_reports",
        ):
            assert expected in tables
        assert "content_hash" in {
            r[1] for r in conn.execute("PRAGMA table_info(jobs)")
        }
    finally:
        conn.close()


def test_set_schema_version_public_api_still_works(env):
    db = Database(env)
    db.set_schema_version(9)
    # The public API records a version row (MAX semantics, as before the fix)
    assert db.get_schema_version() == 9
