"""M7a — render contract round-trips (CPU-only, no GPU/ffmpeg)."""

import json

import pytest
from pydantic import ValidationError


def test_mezzanine_info_roundtrip():
    from clipper.models import MezzanineInfo

    m = MezzanineInfo(
        path="data/jobs/j1/clips/j1-clip01/mezzanine.mp4",
        width=1080, height=1920, fps=25.0, duration_s=30.0,
    )
    assert m.schema_version == 1
    back = MezzanineInfo.model_validate_json(m.model_dump_json())
    assert back == m
    assert MezzanineInfo.model_validate_json(m.model_dump_json()) == m


def test_clip_audio_info_roundtrip():
    from clipper.models import ClipAudioInfo

    a = ClipAudioInfo(
        path="data/jobs/j1/clips/j1-clip01/audio.m4a",
        sample_rate=44100, channels=2, duration_s=30.0,
    )
    assert a.schema_version == 1
    back = ClipAudioInfo.model_validate_json(a.model_dump_json())
    assert back == a


def test_render_artifact_roundtrip():
    from clipper.models import RenderArtifact

    r = RenderArtifact(
        clip_id="j1-clip01",
        mezzanine_path="data/jobs/j1/clips/j1-clip01/mezzanine.mp4",
        audio_path="data/jobs/j1/clips/j1-clip01/audio.m4a",
        captions_path="data/jobs/j1/clips/j1-clip01/captions.ass",
        final_path="data/jobs/j1/clips/j1-clip01/final.mp4",
        thumbnail_path="data/jobs/j1/clips/j1-clip01/thumbnail.jpg",
        width=1080, height=1920, fps=25.0, duration_s=30.0,
        encoder="h264_nvenc", hwaccel="cuda", render_seconds=12.34,
    )
    assert r.schema_version == 1
    back = RenderArtifact.model_validate_json(r.model_dump_json())
    assert back == r
    assert json.loads(r.model_dump_json())["encoder"] == "h264_nvenc"


def test_render_artifact_requires_fields():
    from clipper.models import RenderArtifact

    with pytest.raises(ValidationError):
        RenderArtifact(
            clip_id="x", mezzanine_path="m", audio_path="a",
            captions_path="c", final_path="f", thumbnail_path="t",
            width=1080, height=1920, fps=25.0, duration_s=1.0,
            encoder="h264_nvenc", hwaccel="cuda",
            # render_seconds missing
        )


def test_reexports_from_clipper_models():
    import clipper.models as M

    assert M.MezzanineInfo is not None
    assert M.ClipAudioInfo is not None
    assert M.RenderArtifact is not None


def test_render_config_m7a_defaults():
    from clipper.config import RenderConfig

    r = RenderConfig()
    assert r.mezz_cq == 16
    assert r.final_cq == 19
    assert r.nvenc_preset == "p5"
    assert r.hwaccel == "cuda"
    assert r.cpu_render is False
    # pre-M7a keys untouched
    assert r.output_width == 1080
    assert r.output_height == 1920
    assert r.encoder == "h264_nvenc"
