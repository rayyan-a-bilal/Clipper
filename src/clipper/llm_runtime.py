"""M5a — managed ephemeral Ollama runtime (docs/spec/llm.md).

The user's existing Ollama on ``127.0.0.1:11434`` is NEVER modified: this
module only ever manages the ephemeral child on ``runtime_port`` (11435).
Ollama pins GPUs at process start, not per request, so the child is spawned
with ``CUDA_VISIBLE_DEVICES=`` the configured GPU (GPU1, the first P100) to
guarantee placement. GPU0 (2080 Super) is media-only; GPU2 stays idle.

``managed_ollama(cfg)`` is a context manager that:
- enters: port check -> VRAM check -> spawn ``ollama serve`` in its own
  session/process-group -> readiness poll -> model presence check -> warm-up
  -> GPU compute-app verification;
- exits (ALWAYS, including on exception): SIGTERM the process group, SIGKILL
  after 5s, and wait for the port to be released again.

The child reuses the existing model store (``ollama pull``-already models such
as ``gemma4:12b`` need no re-download) and dies with the M5 stage run.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from clipper.adapters.llm import LLMError

if TYPE_CHECKING:
    from clipper.config import Config

logger = logging.getLogger("clipper.llm_runtime")

# Minimum free VRAM on the target GPU before spawning (MiB).
MIN_FREE_VRAM_MIB = 12000
# Warm-up options are intentionally tiny (one token) — see docs/spec/llm.md.
WARMUP_NUM_PREDICT = 8
WARMUP_TIMEOUT_S = 40.0
WARMUP_KEEP_ALIVE = "5m"
_PORT_RELEASE_TIMEOUT_S = 10.0
_KILL_GRACE_S = 5.0
_READY_POLL_S = 0.5
_MAIN_PORT = 11434  # user's existing service — never touched


def _fail(reason: str, log_path: str | None = None) -> LLMError:
    if log_path:
        return LLMError(f"{reason} (log: {log_path})")
    return LLMError(reason)


def _resolve_models_dir(cfg: Config) -> str:
    """cfg.llm.models_dir -> $OLLAMA_MODELS -> ~/.ollama/models."""
    if cfg.llm.models_dir:
        return cfg.llm.models_dir
    env = os.environ.get("OLLAMA_MODELS")
    if env:
        return env
    return str(Path.home() / ".ollama" / "models")


def _base_url(port: int) -> str:
    return f"http://127.0.0.1:{port}"


def port_is_available(port: int, timeout_s: float = 1.5) -> bool:
    """True when nothing is LISTENING on 127.0.0.1:port."""
    try:
        socket.create_connection(("127.0.0.1", port), timeout=timeout_s).close()
    except OSError:
        return True
    return False


def _port_is_released(port: int, timeout_s: float = 1.0) -> bool:
    try:
        socket.create_connection(("127.0.0.1", port), timeout=timeout_s).close()
    except OSError:
        return True
    return False


def gpu_memory_rows(nvidia_smi: str | None = None) -> dict[int, dict]:
    """Parse `nvidia-smi --query-gpu=index,uuid,name,memory.free,memory.total`.

    Returns {gpu_index: {"uuid", "name", "free": MiB, "total": MiB}}.
    """
    exe = nvidia_smi or "nvidia-smi"
    out = subprocess.run(
        [
            exe,
            "--query-gpu=index,uuid,name,memory.free,memory.total",
            "--format=csv,noheader,nounits",
            "-i",
            "0,1,2",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    rows: dict[int, dict] = {}
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 5 and parts[0].isdigit():
            rows[int(parts[0])] = {
                "uuid": parts[1],
                "name": parts[2],
                "free": int(parts[3]),
                "total": int(parts[4]),
            }
    return rows


def compute_app_gpus(nvidia_smi: str | None = None) -> dict[str, set[int]]:
    """Map running compute process -> set of GPU indices it occupies.

    Joins the process list with the per-GPU process query. Processes
    visible under more than one index (multi-GPUs) get every index listed.
    """
    exe = nvidia_smi or "nvidia-smi"
    out = subprocess.run(
        [
            exe,
            "--query-compute-apps=process_name,pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    out_g = subprocess.run(
        [
            exe,
            "--query-gpu-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
            "-i",
            "0,1,2",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    maps: dict[str, set[int]] = {}
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[1].isdigit():
            pid = parts[1]
            idxs = {
                int(pline.split(",")[0].strip())
                for pline in out_g.stdout.splitlines()
                if len(pline.split(",")) and pline.split(",")[1].strip() == pid
                and pline.split(",")[0].strip().isdigit()
            }
            maps[parts[0].lower()] = idxs
    return maps


def _spawn(
    cfg: Config,
    models_dir: str,
    log_path: Path,
) -> subprocess.Popen:  # ty: ignore[invalid-argument-type]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8")
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(cfg.llm.gpu_index),
            "OLLAMA_HOST": f"127.0.0.1:{cfg.llm.runtime_port}",  # host:port, no scheme
            "OLLAMA_NUM_PARALLEL": "1",
            "OLLAMA_MAX_LOADED_MODELS": "1",
            "OLLAMA_KEEP_ALIVE": "5m",
            "OLLAMA_MODELS": models_dir,
        }
    )
    return subprocess.Popen(
        ["ollama", "serve"],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )


def _wait_port_released(port: int, deadline_s: float) -> None:
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        if _port_is_released(port):
            return
        time.sleep(0.05)


@contextmanager
def managed_ollama(cfg: Config) -> Iterator[str]:
    """Context manager: manages the ephemeral Ollama child, yields base_url."""
    port = int(cfg.llm.runtime_port)
    if not cfg.llm.manage_runtime:
        # Bypass: use the caller-provided base_url unchanged (e.g. the user's
        # existing Ollama). No spawning, no process management.
        yield cfg.llm.base_url
        return

    log_path = Path(cfg.resolved_data_dir) / "llm" / "m5-ollama.log"
    handler = None
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.setLevel(max(logger.level, logging.INFO))
    except OSError:
        logger.setLevel(logging.INFO)

    proc = None
    try:
        if _MAIN_PORT == port:
            raise _fail(
                f"runtime_port {port} is the main Ollama service — refuse to "
                "manage it (never touch 127.0.0.1:11434)",
                str(log_path),
            )
        if not port_is_available(port):
            raise _fail(
                f"port {port} is already in use by another process; "
                "refusing to continue (never kill unknown processes)",
                str(log_path),
            )

        models_dir = _resolve_models_dir(cfg)

        # Verify target GPU free VRAM >= MIN_FREE_VRAM_MIB via nvidia-smi.
        rows = gpu_memory_rows(cfg.devices.nvidia_smi)  # may FileNotFoundError
        row = rows.get(cfg.llm.gpu_index)
        if row is None:
            raise _fail(
                f"nvidia-smi returned no usable data for GPU{cfg.llm.gpu_index}",
                str(log_path),
            )
        if row["free"] < MIN_FREE_VRAM_MIB:
            raise _fail(
                f"GPU{cfg.llm.gpu_index} free VRAM {row['free']} MiB "
                f"< required {MIN_FREE_VRAM_MIB} MiB",
                str(log_path),
            )

        proc = _spawn(cfg, models_dir, log_path)
        logger.info(
            "spawned ollama serve pid=%s models_dir=%s port=%s gpu=%s",
            proc.pid,
            models_dir,
            port,
            cfg.llm.gpu_index,
        )

        # Poll GET /api/tags until ready or startup_timeout_s.
        base = _base_url(port)
        deadline = time.monotonic() + cfg.llm.startup_timeout_s
        tags_list: list | None = None
        last_err = "timed out"
        while time.monotonic() < deadline:
            try:
                r = httpx.get(base + "/api/tags", timeout=2.0)
            except httpx.HTTPError as e:
                last_err = repr(e)
                time.sleep(_READY_POLL_S)
                continue
            if r.status_code == 200:
                tags_list = r.json().get("models", [])
                break
            last_err = f"HTTP {r.status_code}"
            time.sleep(_READY_POLL_S)
        if tags_list is None:
            raise _fail(
                f"ollama not ready within {cfg.llm.startup_timeout_s}s ({last_err})",
                str(log_path),
            )

        # Verify the configured model is present in /api/tags.
        model_names = [m.get("name") for m in tags_list if isinstance(m, dict)]
        if cfg.llm.model not in model_names:
            raise _fail(
                f"model '{cfg.llm.model}' not found in /api/tags "
                f"(have: {model_names})",
                str(log_path),
            )

        # Warm-up: tiny non-streaming chat so the model loads on the GPU.
        warm = httpx.post(
            base + "/api/chat",
            json={
                "model": cfg.llm.model,
                "stream": False,
                "options": {
                    "num_ctx": cfg.llm.num_ctx,
                    "num_batch": cfg.llm.num_batch,
                    "num_predict": WARMUP_NUM_PREDICT,
                },
                "keep_alive": WARMUP_KEEP_ALIVE,
                "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
            },
            timeout=WARMUP_TIMEOUT_S,
        )
        if warm.status_code != 200:
            body = warm.text[:300]
            raise _fail(
                f"warm-up call failed: HTTP {warm.status_code} {body}", str(log_path),
            )

        # Verify a compute app is on the UUID matching cfg.llm.gpu_index
        # (process list from `nvidia-smi --query-compute-apps`).
        apps = compute_app_gpus(cfg.devices.nvidia_smi)
        on_target = [
            name
            for name in ("ollama", "ollama_llama_server")
            if name in apps and cfg.llm.gpu_index in apps[name]
        ]
        if not on_target:
            raise _fail(
                f"no ollama compute process detected on GPU{cfg.llm.gpu_index} "
                f"(apps: {apps})",
                str(log_path),
            )

        yield base
    except LLMError:
        if proc is not None:
            _terminate_process_tree(proc, port)
        raise
    except Exception as e:
        # Any other failure -> clean LLMError with reason + log path.
        if proc is not None:
            _terminate_process_tree(proc, port)
        raise _fail(f"managed Ollama startup failed: {e!r}", str(log_path)) from e
    finally:
        if proc is not None:
            _terminate_process_tree(proc, port)
        if handler is not None:
            logger.removeHandler(handler)
            handler.close()


def _terminate_process_tree(proc: subprocess.Popen, port: int) -> None:
    """SIGTERM the process group; SIGKILL after 5s; wait for port release."""
    # Give any still-open stdio a moment so we don't signal through pipes.
    for stream in (proc.stdout, proc.stderr, proc.stdin):
        if stream is not None and not stream.closed:
            stream.close()
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, TypeError, OSError):
        pass
    try:
        proc.wait(timeout=_KILL_GRACE_S)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, TypeError, OSError):
            pass
        try:
            proc.wait(timeout=_KILL_GRACE_S)
        except subprocess.TimeoutExpired:
            pass
    # Wait until the port is released again (guards against the child
    # re-binding during its exit path).
    _wait_port_released(port, _PORT_RELEASE_TIMEOUT_S)
