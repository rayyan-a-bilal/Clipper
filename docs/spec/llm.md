# M5 — Managed Ollama Runtime (M5a)

Fixed decisions for M5a. See `docs/spec/m5.md` for all M5 constants (M5b/M5c
must not drift from them).

## Runtime policy (non-negotiable)

- The user's existing Ollama on `127.0.0.1:11434` is **NEVER modified** — not
  killed, not reconfigured, not written to. The main service on 11434 is
  off-limits at all times.
- M5 manages its own **ephemeral** Ollama process on
  **`127.0.0.1:11435`** (`cfg.llm.runtime_port`).
- Ollama **pins GPUs at process start, not per request**. Therefore M5 spawns
  `ollama serve` with **`CUDA_VISIBLE_DEVICES=1`** to guarantee the model is
  placed on **GPU1** (the first P100) — the only way to get a placement
  guarantee from Ollama.
- The child **reuses the existing model store**
  (`cfg.llm.models_dir` -> `$OLLAMA_MODELS` -> `~/.ollama/models`).
  `config.yaml` pins `llm.models_dir` to **`/usr/share/ollama/.ollama/models`**
  — the store used by the user's running Ollama (rbilal is in the `ollama`
  group, so the child can access it). `~/.ollama/models` does NOT contain
  manifest data and must not be used. `gemma4:12b` is already there:
  **no re-download** ever happens in M5.
- Additionally, `_resolve_models_dir` (M5a fix) verifies that the chosen
  store's `manifests/` tree actually contains `cfg.llm.model`; if not it
  probes known system stores (`llm_runtime._SYSTEM_STORES`) and uses the
  first one that does. Note: `Config` does not read `config.yaml` (env
  `CLIPPER_*` / `.env` only), so the runtime source of truth for the models
  dir is `cfg.llm.models_dir` -> env -> probes.
- The child lives **only for the duration of a single M5 stage run** and is
  **always terminated afterwards** (normal exit AND exception paths):
  SIGTERM the process group, SIGKILL after 5s, wait for the port to be
  released again. Never leave orphaned `ollama_llama_server` children.
- **GPU0 / RTX 2080 Super is media-only** (ASR/ffencode). **GPU2 is idle
  during M5.**
- The **PI agent is stopped** before real M5 runs (it is a large CPU/GPU
  consumer).

## GPU pinning rationale

Ollama does not expose a per-request device pin. Setting
`CUDA_VISIBLE_DEVICES=1` in the child environment means the CUDA runtime
inside the child only sees GPU1, so the loaded model physically occupies
GPU1 regardless of Ollama's internal scheduler. This is why the runtime is
managed per stage-run rather than a shared long-lived service.

## `num_ctx` semantics

`num_ctx = 16384` is a **PER REQUEST** limit, not a cumulative/rolling
context. It does NOT grow per clip and it does NOT require any restart of
the child between batches; the child is (re)started once per M5 stage run
and lives for the whole run. **Never restart the runtime per clip.**

## Client behavior (`src/clipper/adapters/llm.py`)

- `OllamaLLMClient` speaks plain HTTP (httpx only — no `ollama` python
  package) to `{base_url}/api/chat`.
- Request: `model`, `messages=[system, user]`, `stream=false`,
  `format = schema.model_json_schema()` (strict mode), and
  `options = {temperature, num_ctx, num_batch, num_predict}`.
- **Schema-format fallback (recorded):** if the server rejects the schema
  format (HTTP 400/422/500 naming the format/schema), the client falls back
  to `format="json"` + pydantic validation of the parsed content. The mode
  used is recorded in `client.last_mode` ("schema" or "json") — this doc
  records BOTH modes as supported. This is a *format* fallback, never a
  silent fallback to another model or endpoint.
- Retries: up to `cfg.llm.max_retries` (2) on HTTP error / timeout /
  invalid JSON / schema-validation failure; then `LLMError`.
- **`MockLLM` is a test double**, not a model: a queue of canned responses
  (valid JSON, invalid JSON, invalid schema, exceptions), records calls,
  zero network I/O. Tests use it (or `get_llm_client` monkeypatching), never
  the real model.

## Spawning contract (`src/clipper/llm_runtime.py::managed_ollama(cfg)`)

Context manager yielding the base URL. On enter:

1. If `cfg.llm.manage_runtime` is **false**: yield `cfg.llm.base_url`
   unchanged; do NOT spawn or manage anything.
2. If `runtime_port == 11434`: refuse (never manage the main service).
3. If anything is already listening on `runtime_port`: fail cleanly with
   `LLMError` — **never kill unknown processes**.
4. Resolve models dir: `cfg.llm.models_dir` -> `$OLLAMA_MODELS` ->
   `~/.ollama/models` -> known system stores (`_SYSTEM_STORES`), picking the
   first whose `manifests/` tree contains `cfg.llm.model`; if none does, the
   first candidate is used (and `/api/tags` will name the missing model).
5. Verify via `nvidia-smi` that GPU`cfg.llm.gpu_index` has **>= 12000 MiB**
   free VRAM (constant `MIN_FREE_VRAM_MIB` in `clipper.llm_runtime`).
6. Spawn `["ollama", "serve"]` with `start_new_session=True` and env:
```
CUDA_VISIBLE_DEVICES = str(cfg.llm.gpu_index)   # "1"
OLLAMA_HOST          = 127.0.0.1:{runtime_port}
OLLAMA_NUM_PARALLEL  = 1
OLLAMA_MAX_LOADED_MODELS = 1
OLLAMA_KEEP_ALIVE    = 5m
OLLAMA_MODELS        = <resolved models dir>
```
   stdout+stderr append to `data/llm/m5-ollama.log`.
7. Poll `GET {base}/api/tags` until 200 or `startup_timeout_s` (60s).
8. Verify `cfg.llm.model` is present in the `/api/tags` list.
9. Warm-up: `POST {base}/api/chat` with
   `messages=[{user: "Reply with exactly: OK"}]`, `stream=false`,
   `options={num_ctx:16384, num_batch:256, num_predict:8}`,
   `keep_alive="5m"`. (One token — the model just needs to load.)
10. Verify via `nvidia-smi --query-compute-apps` (+ per-GPU pid query) that
    an `ollama` / `ollama_llama_server` process occupies the GPU whose
    index==`cfg.llm.gpu_index` (the UUID that `nvidia-smi` reports for that
    index).

Any failure at any step -> `LLMError` carrying the reason **and** the log
path, and the child (if it was spawned) is torn down.

On exit (always, including on exception):

- SIGTERM the process group (`os.killpg`), SIGKILL after 5s if still alive,
  `wait()` the child, then poll until the port is released again.
- Never leave orphaned `ollama_llama_server` children.
- Never touch the main service on 11434.

## Preflight

`scripts/llm_preflight.py`: enters `managed_ollama(cfg)`, runs a tiny
`generate_json` sanity check, prints model / base_url / gpu_index / GPU UUID /
free VRAM / compute apps, exits 0/1 — with the child cleaned up on success
AND failure.

## Testing rules

All tests are CPU-only: subprocess spawning is exercised with a real
`sleep`-style child process or a fake `Popen`; `httpx` calls are monkeypatched;
`nvidia-smi` is replaced with canned strings/`FileNotFoundError`. No real
Ollama, no GPU, no `nvidia-smi`, no network in the test suite.
