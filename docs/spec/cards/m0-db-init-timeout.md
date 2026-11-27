# Fix card — M0: fresh `Database(cfg)` init costs ~10s; schema version never recorded

Date: 2026-11-27. Status: **FIXED** (branch `fix-m0-db-init-m5-faces-import-test`).

## Problem

Every fresh `Database(cfg)` construction costs ~10s on this host. Measured
before the fix (venv, temp data dir):

```
fresh init: 10.04s  schema_version=0
second init (same file): 0.01s  schema_version=3
```

Two consequences:

1. **The `~10s/test floor`** in the suite: nearly every stage/worker test
   builds a `Database` over a fresh `tmp_path`, so each one pays the full
   10s (M8a suite: 620 tests in 1540s wall; M8c: 643 tests ~26 min wall).
2. **The schema version row is never recorded** on fresh DBs
   (`schema_migrations` stays empty) — so the migration blocks are re-attempted
   (and re-time-out) on every construction, not just the first.

## Root cause (verified in `src/clipper/db.py`)

`init_db()` runs all DDL on one connection whose implicit write transaction
(and WAL **write lock**) stays open until the final `conn.commit()`.
`_apply_migrations()` then calls `self.set_schema_version(v)`, which opens a
**second connection** to `INSERT OR REPLACE INTO schema_migrations` + commit.
The second writer blocks on conn1's write lock, waits out Python sqlite3's
default **5s busy timeout**, raises `sqlite3.OperationalError`, and the error
is swallowed by the migration `except: pass` blocks. That happens twice
(migrations v2 + v3) → ~10s. The version row never lands, so the next
construction pays it again.

(Second construction on the same file is instant because all DDL there is a
no-op — no writes, no lock — so the stray `set_schema_version(3)` finally
succeeds and records version 3. That also means pre-fix production DBs are
already at version 3; only FRESH files were broken.)

## Decision (2026-11-27; M0 behavior fix — no contract change)

- `Database` is **single-connection during `init_db`**: migrations are
  executed AND version rows are recorded on `init_db`'s own connection/cursor;
  the deferred final `conn.commit()` commits schema + versions atomically.
  No second writer, no busy timeout, version recorded on the first pass.
- The public `set_schema_version(version)` / `get_schema_version()` API is
  unchanged (no external callers; `cli.py` only reads
  `get_schema_version()`; kept for API stability).
- Migration recording becomes unambiguous: DDL is idempotent
  (`IF NOT EXISTS` / try-except), then the version is recorded on the same
  cursor — a DB left in the pre-fix state (tables present, versions empty)
  converges to version 3 on the next construction, in milliseconds.
- No schema changes, no new dependencies, no `busy_timeout` additions
  (the second-connection writer is gone), no ORM, no DB file migrations —
  existing DB files are left as-is.

## Change set

- `src/clipper/db.py`: private `_record_schema_version(cursor, version)`;
  `_apply_migrations` calls it instead of `self.set_schema_version(...)`;
  `init_db` flow/commit unchanged.
- `tests/test_db_init.py` (NEW, CPU-only, temp dirs):
  - fresh init is fast (< 5s; bug measured 10.0s) and
    `get_schema_version() == 3`;
  - legacy pre-fix state (fresh init, then `DELETE FROM schema_migrations`,
    then re-init) is fast and converges to version 3 with all tables intact.

## Acceptance

- venv verified; targeted `tests/test_db_init.py` + full
  `python -m pytest -m "not gpu"` green; `python -m ruff check .` clean.
- Wall-clock evidence that the suite floor dropped (full-suite duration
  compared against the 643-test ~26 min M8c baseline).

## Out of scope

- Any WAL/locking redesign, connection pooling, ORM, or schema changes.
- M1–M8 stage/contract/config changes (none required).
