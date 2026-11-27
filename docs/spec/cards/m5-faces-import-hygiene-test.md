# Fix card — M5 test: `test_faces.py` import-hygiene test corrupts `sys.modules` for the rest of the suite

Date: 2026-11-27. Status: **FIXED** (branch `fix-m0-db-init-m5-faces-import-test`).

## Problem

`tests/test_faces.py::test_module_import_never_pulls_onnxruntime` popped
**`clipper.adapters` (the parent package)** plus `clipper.adapters.faces` and
`clipper.facedec` out of `sys.modules` and re-imported `clipper.adapters.faces`.
Re-importing the *parent* re-runs `clipper/adapters/__init__.py`, which creates
a **new** `clipper.adapters` module object. The submodules
(`...media`, `...llm`, `...render`, `...vision`) are still in `sys.modules`, so
the import succeeds — but the fresh parent object never gets its
`.media` / `.llm` / ... attribute bindings re-set by the import system.

Result: any later test file using a **string-targeted** monkeypatch —
e.g. `monkeypatch.setattr("clipper.adapters.media.get_adapter", ...)` — fails
for the rest of the run. It bit the first full M8c suite run
(16 failures, zero in file isolation) and is a landmine for every future
test author (M9+ included).

## Decision (2026-11-27; test-only fix — no production code touched)

- **Never pop a parent package from `sys.modules` in tests.** The parent
  `clipper.adapters` object (and its attribute bindings) must survive the
  whole session.
- The hygiene test may only pop the **leaf modules it is asserting on**
  (`clipper.adapters.faces`, `clipper.facedec`), does the fresh import,
  asserts `onnxruntime` did not enter `sys.modules`, and then **restores the
  prior leaf module objects** so every existing module reference (and the
  `importlib` cache) keeps pointing at one consistent set of objects.
- Test-author rule (binding, mirrors the M8c workaround and extends it):
  prefer patching the exact module objects your module bound at import
  (e.g. `worker_stages.media_adapters.get_adapter`) over string-targeted
  `monkeypatch.setattr("clipper.adapters.<sub>.<attr>", ...)`.

## Change set

- `tests/test_faces.py`: `test_module_import_never_pulls_onnxruntime`
  rewritten per the decision (pop/restore leaves only, skip-guard if
  `onnxruntime` was already imported by an earlier test — impossible in the
  CPU-only suite today, defensive for the future); assertion and the
  `test_facedec_source_is_dependency_free` source scan unchanged.

## Acceptance

- venv verified; `tests/test_faces.py` green in isolation AND in the full
  `python -m pytest -m "not gpu"` run; `python -m ruff check .` clean.
- Regression evidence: a full-suite run where a string-targeted
  `clipper.adapters.*` monkeypatch in a file ordered AFTER `test_faces.py`
  passes (the M8c 16-failure mode is gone; M8c tests already use exact
  module-object patches and stay green).

## Out of scope

- Changing `clipper/adapters/__init__.py`, `faces.py`, `facedec.py`, or any
  M6b detector behavior (the import hygiene they provide is exactly what
  this test protects).
