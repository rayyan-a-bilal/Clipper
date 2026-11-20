"""
Configuration management for clipper.
"""

import os
from pathlib import Path

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PathsConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CLIPPER_")

    data_dir: str = Field(default="./data")

class DevicesConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CLIPPER_")

    ffmpeg: str | None = None
    ffprobe: str | None = None
    nvidia_smi: str | None = None

class LLMConfig(BaseSettings):
    """LLM settings (M5a — managed Ollama runtime).

    New keys (M5a, see docs/spec/llm.md / docs/spec/m5.md):
    provider, manage_runtime, base_url, runtime_port, gpu_index, model,
    models_dir, num_ctx, num_batch, temperature, num_predict, timeout_s,
    startup_timeout_s, max_retries, batch_size, max_batch_tokens,
    context_before_s, context_after_s.

    Backward compatible: pre-M5a configs (endpoint/api_key/timeout) still
    load unchanged — the old keys are accepted and `endpoint` is mirrored
    into `base_url` when no explicit base_url was provided.
    """

    model_config = SettingsConfigDict(env_prefix="CLIPPER_", extra="ignore")

    # -- M5a managed Ollama runtime (defaults) --
    provider: str = "ollama"
    manage_runtime: bool = True
    base_url: str = "http://127.0.0.1:11435"
    runtime_port: int = 11435
    gpu_index: int = 1
    model: str = "gemma4:12b"
    models_dir: str | None = None
    num_ctx: int = 16384  # PER REQUEST, not cumulative (docs/spec/llm.md)
    num_batch: int = 256
    temperature: float = 0.1
    num_predict: int = 1024
    timeout_s: int = 300
    startup_timeout_s: int = 60
    max_retries: int = 2
    batch_size: int = 5
    max_batch_tokens: int = 6000
    context_before_s: float = 10.0
    context_after_s: float = 10.0

    # -- legacy pre-M5a keys (kept for backward-compatible configs) --
    endpoint: str = "http://127.0.0.1:8081/v1"
    api_key: str | None = None
    timeout: int = 300

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_llm_keys(cls, data):  # ty: ignore[invalid-argument-type]
        """Accept pre-M5a configs (`endpoint`/`api_key`/`timeout`)."""
        if not isinstance(data, dict):
            return data
        legacy_endpoint = data.get("endpoint")
        if legacy_endpoint is not None and data.get("base_url") is None:
            new = dict(data)
            new["base_url"] = str(legacy_endpoint)
            new.setdefault("provider", "ollama")
            new.setdefault("manage_runtime", True)
            return new
        return data

class ASRConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CLIPPER_")

    model: str = "large-v3-turbo"
    device: str = "cpu"  # GPU0 (cuda:0) per spec when free; CPU as safe default
    compute_type: str = "int8"  # "float16" on cuda, "int8" / "float32" on cpu
    word_timestamps: bool = True
    vad_filter: bool = True
    condition_on_previous_text: bool = False

class ClipGateConfig(BaseModel):
    """M5c quality-gate thresholds (defaults fixed by docs/spec/m5.md:
    weighted >= 7.2 AND hook >= 6 AND self_contained >= 7)."""

    weighted: float = 7.2
    hook: int = 6
    self_contained: int = 7


class ClipsConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CLIPPER_")

    target_count: int = 3
    max_count: int = 5
    min_duration: float = 15.0
    max_duration: float = 75.0
    overlap_threshold: float = 0.3
    hook_overlay_duration: float = 2.5
    # M4a — niche -> keywords; empty by default (niche score falls back to
    # neutral 0.5). Backward compatible: existing configs load unchanged.
    niche_keywords: dict[str, list[str]] = {}
    # M5c — quality-gate thresholds (defaults fixed by docs/spec/m5.md).
    gate: ClipGateConfig = Field(default_factory=ClipGateConfig)

class FacesConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CLIPPER_")

    detection_fps: float = 3.0
    min_detection_confidence: float = 0.5
    iou_threshold: float = 0.5

class CaptionsConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CLIPPER_")

    font: str = "Montserrat ExtraBold"
    font_size: int = 88
    margin_v: int = 450
    safe_zone_right: int = 150
    safe_zone_top: int = 220
    max_words_per_group: int = 4
    max_chars_per_line: int = 20
    max_lines: int = 2
    karaoke_highlight: str = "yellow"

class RenderConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CLIPPER_")

    output_width: int = 1080
    output_height: int = 1920
    crf: int = 14
    encoder: str = "h264_nvenc"

class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    paths: PathsConfig = PathsConfig()
    devices: DevicesConfig = DevicesConfig()
    llm: LLMConfig = LLMConfig()
    asr: ASRConfig = ASRConfig()
    clips: ClipsConfig = ClipsConfig()
    faces: FacesConfig = FacesConfig()
    captions: CaptionsConfig = CaptionsConfig()
    render: RenderConfig = RenderConfig()

    @property
    def data_dir(self) -> Path:
        """Get the resolved data directory path."""
        # Check for environment variable override
        if "CLIPPER_DATA_DIR" in os.environ:
            return Path(os.environ["CLIPPER_DATA_DIR"])
        else:
            return Path(self.paths.data_dir)

    @property
    def resolved_data_dir(self) -> Path:
        """Get the resolved data directory path."""
        return self.data_dir.resolve()

    @classmethod
    def load_config(cls) -> 'Config':
        """Load configuration from file and environment."""
        config = cls()

        # Create writable directories if needed
        try:
            config.resolved_data_dir.mkdir(parents=True, exist_ok=True)
            (config.resolved_data_dir / "jobs").mkdir(exist_ok=True)
            (config.resolved_data_dir / "clips").mkdir(exist_ok=True)
            (config.resolved_data_dir / "stages").mkdir(exist_ok=True)
            (config.resolved_data_dir / "run").mkdir(exist_ok=True)
            (config.resolved_data_dir / "logs").mkdir(exist_ok=True)
        except Exception:
            # Ignore errors for now - configuration can still work
            pass

        return config
