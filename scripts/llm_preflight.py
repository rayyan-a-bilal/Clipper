"""M5a — preflight check for the managed Ollama runtime.

Usage: `.venv/bin/python scripts/llm_preflight.py`

Spawns the managed ephemeral Ollama (docs/spec/llm.md), runs a tiny
`generate_json` sanity check against it, prints model / base_url /
gpu_index / GPU UUID / free VRAM, and ALWAYS tears the child down (both on
success and on failure). Exit 0 on success, 1 on any failure.
"""

import sys
import traceback

from pydantic import BaseModel

from clipper.adapters.llm import OllamaLLMClient
from clipper.config import Config
from clipper.llm_runtime import compute_app_gpus, gpu_memory_rows, managed_ollama


class PreflightReply(BaseModel):
    ok: bool = True
    note: str = ""


def main() -> int:
    cfg = Config.load_config()
    rows_before: dict[int, dict] = {}
    apps: dict[str, set[int]] = {}
    try:
        with managed_ollama(cfg) as base_url:
            rows = gpu_memory_rows(cfg.devices.nvidia_smi)
            rows_before = rows
            apps = compute_app_gpus(cfg.devices.nvidia_smi)
            client = OllamaLLMClient(cfg, base_url=base_url)
            client.timeout_s = 60
            client.generate_json(
                "You are a health check. Reply with valid JSON only.",
                "Preflight: confirm the model is serving.",
                PreflightReply,
            )
    except Exception as e:  # noqa: B014
        print(f"FAIL: {e}", file=sys.stderr)
        # Full chained traceback: the real cause often lives in __cause__
        # (e.g. a startup error wrapped into LLMError with `from e`).
        traceback.print_exc(file=sys.stderr)
        return 1

    gpu = rows_before.get(cfg.llm.gpu_index, {})
    print(f"model:       {cfg.llm.model}")
    print(f"base_url:    {base_url}")
    print(f"gpu_index:   {cfg.llm.gpu_index} ({gpu.get('name', 'unknown GPU')})")
    print(f"gpu_uuid:    {gpu.get('uuid', 'unknown')}")
    print(f"free_vram:   {gpu.get('free', 0)} MiB (of {gpu.get('total', 0)} MiB)")
    print(
        "compute_apps: "
        + ", ".join(
            f"{name} -> GPU{sorted(gpus)}" for name, gpus in sorted(apps.items())
        )
        or "none"
    )
    print("OK: managed Ollama preflight passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
