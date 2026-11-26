"""M8a report contract tests — CPU only, no I/O (pydantic round-trips).

Covers: StageReport / JobReport round-trip, defaults, schema_version,
package re-exports (M8a-J4: StageReport has no schema_version).
"""

from pydantic import ValidationError

import clipper.models as models
from clipper.models import JobReport, StageReport


def _stage(**overrides) -> StageReport:
    base = {
        "name": "prep",
        "required": ["ingest", "probe", "proxy", "audio"],
        "state": "done",
    }
    base.update(overrides)
    return StageReport(**base)


def _job(**overrides) -> JobReport:
    base = {
        "job_id": "job-1",
        "state": "awaiting_qc",
        "clip_count": 2,
        "stages": [_stage(), _stage(name="render", state="done", started_at="t0")],
    }
    base.update(overrides)
    return JobReport(**base)


def test_stage_report_round_trip():
    report = _stage(
        started_at="2026-11-25T10:00:00",
        finished_at="2026-11-25T10:00:05",
        artifact_paths=["/j/ingest.json", "/j/probe.json"],
        error=None,
    )
    raw = report.model_dump_json(indent=2)
    back = StageReport.model_validate_json(raw)
    assert back == report


def test_stage_report_defaults():
    report = _stage()
    assert report.started_at is None
    assert report.finished_at is None
    assert report.artifact_paths == []
    assert report.error is None
    assert report.required == ["ingest", "probe", "proxy", "audio"]


def test_stage_report_requires_all_card_fields():
    full = {"name": "prep", "required": ["ingest"], "state": "done"}
    for missing in ("name", "required", "state"):
        kwargs = {k: v for k, v in full.items() if k != missing}
        try:
            StageReport(**kwargs)
            raise AssertionError(f"StageReport accepted without {missing}")
        except ValidationError:
            pass


def test_job_report_schema_version_defaults_to_one():
    job = _job()
    assert job.schema_version == 1
    assert job.model_dump()["schema_version"] == 1
    assert (
        JobReport(schema_version=1, job_id="x", state="done", clip_count=0)
        .schema_version
        == 1
    )


def test_job_report_round_trip_preserves_everything():
    job = _job(
        started_at="2026-11-25T09:00:00",
        finished_at="2026-11-25T09:30:00",
        error=None,
    )
    raw = job.model_dump_json(indent=2)
    back = JobReport.model_validate_json(raw)
    assert back == job
    assert back.schema_version == 1
    assert back.clip_count == 2
    assert [s.name for s in back.stages] == ["prep", "render"]


def test_job_report_failed_shape():
    job = _job(
        state="failed",
        clip_count=0,
        error="scoring: LLM batch failed after retries",
        stages=[_stage(state="failed", error="same message")],
    )
    raw = job.model_dump_json()
    back = JobReport.model_validate_json(raw)
    assert back.state == "failed"
    assert back.clip_count == 0
    assert back.error and "scoring" in back.error
    assert back.stages[0].error is not None


def test_jobs_zero_clips_is_a_valid_success_shape():
    # Decision 5: done with zero clips is a valid terminal report.
    job = _job(state="done", clip_count=0)
    assert job.model_validate_json(job.model_dump_json()) == job


def test_reexports_from_clipper_models():
    assert models.StageReport is StageReport
    assert models.JobReport is JobReport


def test_report_serialization_is_canonical_json():
    # Same canonical form as the other artifacts (indent=2).
    job = _job()
    raw = job.model_dump_json(indent=2)
    assert raw.startswith("{\n  \"")
    assert "\n}\n" not in raw and raw.endswith("}")
