"""M5a — config tests: new LLM defaults + backward compatibility."""

from clipper.config import Config, LLMConfig, PathsConfig


def _cfg(tmp_path) -> Config:
    return Config(paths=PathsConfig(data_dir=str(tmp_path)))


def test_new_llm_defaults(tmp_path):
    cfg = _cfg(tmp_path)
    llm = cfg.llm
    assert llm.provider == "ollama"
    assert llm.manage_runtime is True
    assert llm.base_url == "http://127.0.0.1:11435"
    assert llm.runtime_port == 11435
    assert llm.gpu_index == 1
    assert llm.model == "gemma4:12b"
    assert llm.models_dir is None
    assert llm.num_ctx == 16384
    assert llm.num_batch == 256
    assert llm.temperature == 0.1
    assert llm.num_predict == 1024
    assert llm.timeout_s == 300
    assert llm.startup_timeout_s == 60
    assert llm.max_retries == 2
    assert llm.batch_size == 5
    assert llm.max_batch_tokens == 6000
    assert llm.context_before_s == 10.0
    assert llm.context_after_s == 10.0
    # never the main service
    assert llm.runtime_port != 11434


def test_legacy_config_still_loads(tmp_path):
    """Pre-M5a configs (endpoint/api_key/timeout) must keep loading."""
    cfg = Config(
        paths=PathsConfig(data_dir=str(tmp_path)),
        llm=LLMConfig(
            endpoint="http://127.0.0.1:8081/v1",
            api_key="k",
            timeout=120,
        ),
    )
    assert cfg.llm.endpoint == "http://127.0.0.1:8081/v1"
    assert cfg.llm.api_key == "k"
    assert cfg.llm.timeout == 120
    # old endpoint mirrored into base_url (no loss of information)
    assert cfg.llm.base_url == "http://127.0.0.1:8081/v1"


def test_legacy_dict_shape_loads(tmp_path):
    """As it would arrive from an old config.yaml dict."""
    cfg = Config.model_validate(
        {
            "llm": {
                "endpoint": "http://127.0.0.1:8081/v1",
                "api_key": "oldkey",
                "timeout": 90,
            },
            "paths": {"data_dir": str(tmp_path)},
        }
    )
    assert cfg.llm.api_key == "oldkey"
    assert cfg.llm.timeout == 90
    assert cfg.llm.base_url == "http://127.0.0.1:8081/v1"
    # new keys still have their defaults even in a legacy config
    assert cfg.llm.manage_runtime is True
    assert cfg.llm.gpu_index == 1


def test_selection_config_fields(tmp_path):
    """M5b/M5c selection reads the clips config fields + gate thresholds."""
    cfg = _cfg(tmp_path)
    assert cfg.clips.target_count == 3
    assert cfg.clips.max_count == 5
    assert cfg.clips.overlap_threshold == 0.3
    # M5c gate thresholds (defaults fixed by docs/spec/m5.md)
    assert cfg.clips.gate.weighted == 7.2
    assert cfg.clips.gate.hook == 6
    assert cfg.clips.gate.self_contained == 7
