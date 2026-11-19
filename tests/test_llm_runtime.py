"""M5a — managed Ollama runtime tests (CPU-only).

Subprocess is exercised with a REAL `sleep` child (so process-group teardown
actually runs) or faked entirely; httpx and nvidia-smi are always mocked.
No real Ollama, GPU, nvidia-smi, or network.
"""

import importlib.util
import subprocess
from pathlib import Path

import httpx
import pytest

from clipper import llm_runtime
from clipper.adapters.llm import LLMError
from clipper.config import Config, LLMConfig, PathsConfig
from clipper.llm_runtime import managed_ollama

# Grab the REAL Popen before any test monkeypatches subprocess.Popen.
REAL_POPEN = subprocess.Popen


@pytest.fixture
def cfg(tmp_path):
    llm = LLMConfig(
        provider="ollama",
        manage_runtime=True,
        base_url="http://127.0.0.1:11435",
        runtime_port=11435,
        gpu_index=1,
        model="gemma4:12b",
        models_dir=str(tmp_path / "models"),
        startup_timeout_s=60,
    )
    return Config(paths=PathsConfig(data_dir=str(tmp_path)), llm=llm)


class FakeProc:
    """Wraps a REAL `sleep` child so teardown/kill paths actually run."""

    instances: list["FakeProc"] = []

    def __init__(self):
        self._p = REAL_POPEN(
            ["sleep", "300"],
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.pid = self._p.pid
        self.stdout = None
        self.stderr = None
        self.stdin = None
        FakeProc.instances.append(self)

    def wait(self, timeout=None):
        return self._p.wait(timeout=timeout)

    def poll(self):
        return self._p.poll()

    @property
    def alive(self) -> bool:
        return self._p.poll() is None


def _fake_popen(sink: dict):
    def _popen(cmd, **kw):
        sink["cmd"] = cmd
        sink["kwargs"] = kw
        return FakeProc()

    return _popen


def _ok_tags():
    return httpx.Response(200, json={"models": [{"name": "gemma4:12b"}]})


def _stub_happy(monkeypatch, sink: dict) -> None:
    monkeypatch.setattr(llm_runtime.subprocess, "Popen", _fake_popen(sink))
    def _port_ok(port, timeout_s=1.5):
        return True

    monkeypatch.setattr(llm_runtime, "port_is_available", _port_ok)
    monkeypatch.setattr(
        llm_runtime,
        "gpu_memory_rows",
        lambda nvidia_smi=None: {
            0: {"uuid": "GPU-2080", "name": "2080 Super", "free": 4096, "total": 8192},
            1: {"uuid": "GPU-P100", "name": "P100", "free": 16000, "total": 16384},
        },
    )
    monkeypatch.setattr(
        llm_runtime,
        "compute_app_gpus",
        lambda nvidia_smi=None: {"ollama_llama_server": {1}, "comfyui": {0}},
    )
    monkeypatch.setattr(llm_runtime.httpx, "get", lambda url, timeout=None: _ok_tags())
    monkeypatch.setattr(
        llm_runtime.httpx, "post",
        lambda url, json=None, timeout=None: httpx.Response(
            200, json={"message": {"content": "OK"}}
        ),
    )


# ---------------------------------------------------------------------------
# happy path: env, port, log, base_url, teardown
# ---------------------------------------------------------------------------
def test_spawn_env_port_and_session(cfg, monkeypatch):
    FakeProc.instances.clear()
    sink: dict = {}
    _stub_happy(monkeypatch, sink)
    with managed_ollama(cfg) as base_url:
        assert base_url == "http://127.0.0.1:11435"

    assert sink["cmd"] == ["ollama", "serve"]
    kw = sink["kwargs"]
    assert kw.get("start_new_session") is True
    env = kw["env"]
    # CUDA_VISIBLE_DEVICES is the only reliable way to pin GPU1 at spawn time
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["OLLAMA_HOST"] == "127.0.0.1:11435"
    assert "11434" not in env["OLLAMA_HOST"]
    assert env["OLLAMA_NUM_PARALLEL"] == "1"
    assert env["OLLAMA_MAX_LOADED_MODELS"] == "1"
    assert env["OLLAMA_KEEP_ALIVE"] == "5m"
    assert env["OLLAMA_MODELS"].endswith("models")

    # log file at data/llm/m5-ollama.log
    assert (cfg.resolved_data_dir / "llm" / "m5-ollama.log").exists()


def test_process_killed_on_normal_exit(cfg, monkeypatch):
    FakeProc.instances.clear()
    sink: dict = {}
    _stub_happy(monkeypatch, sink)
    with managed_ollama(cfg):
        proc = FakeProc.instances[-1]
        assert proc.alive
    # SIGTERM/SIGKILL the process group; child must be dead afterwards
    assert proc.poll() is not None


def test_process_killed_on_exception(cfg, monkeypatch):
    FakeProc.instances.clear()
    sink: dict = {}
    _stub_happy(monkeypatch, sink)
    # model not in /api/tags -> LLMError mid-enter
    def _no_model_tags(url, timeout=None):
        return httpx.Response(200, json={"models": [{"name": "other-model"}]})

    monkeypatch.setattr(llm_runtime.httpx, "get", _no_model_tags)
    with pytest.raises(LLMError, match="not found in /api/tags"):
        with managed_ollama(cfg):
            pass
    proc = FakeProc.instances[-1]
    assert proc.poll() is not None  # never left running


# ---------------------------------------------------------------------------
# clean failures (no unknown process ever killed)
# ---------------------------------------------------------------------------
def test_port_occupied_fails_cleanly(cfg, monkeypatch):
    sink: dict = {}
    monkeypatch.setattr(llm_runtime.subprocess, "Popen", _fake_popen(sink))
    monkeypatch.setattr(llm_runtime, "port_is_available",
                       lambda port, timeout_s=1.5: False)

    with pytest.raises(LLMError, match="already in use"):
        with managed_ollama(cfg):
            pass
    assert "cmd" not in sink  # nothing was spawned


def test_insufficient_vram_fails_cleanly(cfg, monkeypatch):
    sink: dict = {}
    _stub_happy(monkeypatch, sink)
    monkeypatch.setattr(
        llm_runtime,
        "gpu_memory_rows",
        lambda nvidia_smi=None: {1: {"uuid": "GPU-P100", "free": 4096, "total": 16384}},
    )
    with pytest.raises(LLMError, match="free VRAM"):
        with managed_ollama(cfg):
            pass
    assert "cmd" not in sink


def test_wrong_gpu_fails_cleanly(cfg, monkeypatch):
    FakeProc.instances.clear()
    sink: dict = {}
    _stub_happy(monkeypatch, sink)
    # ollama only on GPU0 (the media GPU) -> not acceptable
    monkeypatch.setattr(
        llm_runtime,
        "compute_app_gpus",
        lambda nvidia_smi=None: {"ollama_llama_server": {0}},
    )
    with pytest.raises(LLMError, match="GPU1"):
        with managed_ollama(cfg):
            pass
    assert not any(p.alive for p in FakeProc.instances)


def test_nvidia_smi_missing_fails_cleanly(cfg, monkeypatch):
    sink: dict = {}
    _stub_happy(monkeypatch, sink)
    monkeypatch.setattr(
        llm_runtime,
        "gpu_memory_rows",
        lambda nvidia_smi=None: (_ for _ in ()).throw(FileNotFoundError("nvidia-smi")),
    )
    with pytest.raises(LLMError, match="nvidia-smi|startup failed"):
        with managed_ollama(cfg):
            pass
    assert "cmd" not in sink


def test_model_missing_fails_cleanly(cfg, monkeypatch):
    FakeProc.instances.clear()
    sink: dict = {}
    _stub_happy(monkeypatch, sink)
    # model NOT in the /api/tags list -> LLMError
    def _no_model_tags2(url, timeout=None):
        return httpx.Response(200, json={"models": [{"name": "other-model"}]})

    monkeypatch.setattr(llm_runtime.httpx, "get", _no_model_tags2)
    with pytest.raises(LLMError, match="[Mm]odel"):
        with managed_ollama(cfg):
            pass
    # spawned before the check -> must have been torn down
    assert FakeProc.instances and not any(p.alive for p in FakeProc.instances)


def test_refuses_to_manage_main_service_port(cfg, monkeypatch):
    cfg2 = cfg.model_copy(deep=True)
    cfg2.llm = LLMConfig(**{**cfg.llm.model_dump(), "runtime_port": 11434})
    sink: dict = {}
    monkeypatch.setattr(llm_runtime.subprocess, "Popen", _fake_popen(sink))
    def _port_ok2(port, timeout_s=1.5):
        return True

    monkeypatch.setattr(llm_runtime, "port_is_available", _port_ok2)
    with pytest.raises(LLMError, match="11434|main"):
        with managed_ollama(cfg2):
            pass
    assert "cmd" not in sink


# ---------------------------------------------------------------------------
# manage_runtime=false -> no spawn, base_url yielded unchanged
# ---------------------------------------------------------------------------
def test_manage_runtime_false_bypasses_spawning(cfg, monkeypatch):
    def _no_spawn(*a, **kw):
        raise AssertionError("must not spawn when manage_runtime=false")

    cfg2 = cfg.model_copy(deep=True)
    cfg2.llm = LLMConfig(
        **{**cfg.llm.model_dump(), "manage_runtime": False,
           "base_url": "http://127.0.0.1:11434"}
    )
    monkeypatch.setattr(llm_runtime.subprocess, "Popen", _no_spawn)
    with managed_ollama(cfg2) as base_url:
        # must be the user's existing service, untouched
        assert base_url == "http://127.0.0.1:11434"


# ---------------------------------------------------------------------------
# preflight script: cleans up on success AND failure
# ---------------------------------------------------------------------------
def _load_preflight_script() -> importlib:
    path = Path(__file__).resolve().parents[1] / "scripts" / "llm_preflight.py"
    spec = importlib.util.spec_from_file_location("clipper_llm_preflight_script", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_preflight_success_and_failure(monkeypatch, capsys):
    pf = _load_preflight_script()
    class _Client:
        def generate_json(self, *a, **kw):
            from clipper.models import ScoringResponse

            return ScoringResponse(scores=[])

    def _fake_managed(cfg_in):
        import contextlib

        @contextlib.contextmanager
        def _ctx():
            yield "http://127.0.0.1:11435"

        return _ctx()

    monkeypatch.setattr(pf, "managed_ollama", _fake_managed)
    monkeypatch.setattr(pf, "OllamaLLMClient", lambda cfg, base_url=None: _Client())
    monkeypatch.setattr(
        pf, "gpu_memory_rows",
        lambda nvidia_smi=None: {1: {"uuid": "GPU-P100", "free": 6000, "total": 16384}},
    )
    monkeypatch.setattr(pf, "compute_app_gpus", lambda nvidia_smi=None: {"ollama": {1}})

    assert pf.main() == 0
    out = capsys.readouterr().out
    assert "gemma4:12b" in out and "11435" in out and "GPU-P100" in out

    def _boom(cfg_in):
        raise LLMError("startup failed (log: /tmp/x.log)")

    monkeypatch.setattr(pf, "managed_ollama", _boom)
    assert pf.main() == 1
