"""M5a — Ollama LLM client (HTTP, via httpx) + MockLLM test double.

- No `ollama` python package: only plain HTTP against the local endpoint.
- `OllamaLLMClient.generate_json()` POSTs `{base_url}/api/chat` with
  `messages = [system, user]`, `stream = False`,
  `format = schema.model_json_schema()` (strict mode) and
  `options = {temperature, num_ctx, num_batch, num_predict}`.
  If the server rejects the schema format (HTTP 400/422/500 naming the
  format/schema), it falls back to `format="json"` + pydantic validation
  (`schema.model_validate` on the parsed content) — the mode used is recorded
  in `client.last_mode` and in docs/spec/llm.md.
- Retries up to `cfg.llm.max_retries` times on HTTP error / timeout /
  invalid JSON / schema validation failure, then raises `LLMError`.
  There is NEVER a silent fallback to another model or endpoint.
- `MockLLM` is a test double: a queue of canned responses (valid JSON,
  invalid JSON, invalid schema, exceptions), records every call, zero
  network I/O. It is NOT a model.
"""

from __future__ import annotations

import json
from collections import deque
from typing import Any, Protocol, runtime_checkable

import httpx
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from clipper.config import Config


class LLMError(Exception):
    """Raised when an LLM call cannot be completed cleanly."""


class _SchemaRejected(Exception):
    """Internal: server rejected the JSON-schema `format` value."""


@runtime_checkable
class LLMClient(Protocol):
    """Protocol every LLM client (real or mock) must satisfy."""

    def generate_json(
        self, system: str, user: str, schema: type[BaseModel]
    ) -> BaseModel: ...


def _default_client_factory(timeout_s: float) -> httpx.Client:
    return httpx.Client(timeout=timeout_s)


class OllamaLLMClient:
    """Thin Ollama `/api/chat` client with JSON-schema constrained output."""

    def __init__(
        self,
        cfg: Config,
        base_url: str | None = None,
        client_factory=_default_client_factory,
    ) -> None:
        llm = cfg.llm
        self.cfg = cfg
        self.model = llm.model
        self.base_url = (base_url or llm.base_url).rstrip("/")
        self.timeout_s = llm.timeout_s
        self.max_retries = llm.max_retries
        self.temperature = llm.temperature
        self.num_ctx = llm.num_ctx
        self.num_batch = llm.num_batch
        self.num_predict = llm.num_predict
        self._client_factory = client_factory
        self.last_mode: str | None = None

    # -- internals ------------------------------------------------------
    def _request(self, payload: dict[str, Any]) -> dict:
        client = self._client_factory(self.timeout_s)
        try:
            r = client.post(self.base_url + "/api/chat", json=payload)
            r.raise_for_status()
            data: dict = r.json()
        except httpx.HTTPStatusError as e:
            body = e.response.text[:300]
            if (
                e.response.status_code in (400, 422, 500)
                and ("format" in body.lower() or "schema" in body.lower())
            ):
                raise _SchemaRejected(
                    f"server rejected schema format: {body}"
                ) from e
            raise LLMError(f"HTTP {e.response.status_code}: {body}") from e
        except httpx.HTTPError as e:
            raise LLMError(f"transport error: {e!r}") from e
        finally:
            client.close()
        return data

    @staticmethod
    def _parse(data: dict, schema: type[BaseModel]) -> BaseModel:
        content = (data.get("message") or {}).get("content") or ""
        try:
            obj = json.loads(content)
        except json.JSONDecodeError as e:
            raise LLMError(f"model returned invalid JSON: {content[:300]!r}") from e
        try:
            return schema.model_validate(obj)
        except PydanticValidationError as e:
            raise LLMError(f"model response failed schema validation: {e}") from e

    # -- public API ------------------------------------------------------
    def generate_json(
        self,
        system: str,
        user: str,
        schema: type[BaseModel],
    ) -> BaseModel:
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "think": False,  # thinking models would otherwise spend the whole
            # num_predict budget on hidden reasoning and return empty content
            "format": schema.model_json_schema(),
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_batch": self.num_batch,
                "num_predict": self.num_predict,
            },
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        attempts = self.max_retries + 1
        last: BaseException | None = None
        schema_rejected = False
        for _ in range(attempts):
            if schema_rejected:
                payload["format"] = "json"
            try:
                data = self._request(payload)
                self.last_mode = "json" if schema_rejected else "schema"
                return self._parse(data, schema)
            except _SchemaRejected as e:
                # Server rejected the schema format: switch to format="json"
                # + pydantic validation (documented fallback, NOT a retry of
                # the same request).
                last = e
                schema_rejected = True
            except LLMError as e:
                last = e
        assert last is not None
        if isinstance(last, LLMError):
            raise last
        raise LLMError(f"gave up after {attempts} attempt(s): {last!r}") from last


def get_llm_client(cfg: Config, base_url: str | None = None) -> OllamaLLMClient:
    """Factory; tests monkeypatch this instead of touching the network."""
    return OllamaLLMClient(cfg, base_url=base_url)


class MockLLM:
    """Test double: queue of canned responses, records calls, zero network I/O.

    Queue entries may be:
    - a `BaseModel` instance of the expected schema type,
    - a dict (auto-validated into the requested schema),
    - an `Exception` instance to be raised (invalid JSON, invalid schema, ...).

    It is a test double, NOT a model.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._q: deque[Any] = deque()

    def enqueue(self, *responses: Any) -> None:
        self._q.extend(responses)

    @property
    def pending(self) -> int:
        return len(self._q)

    def generate_json(
        self,
        system: str,
        user: str,
        schema: type[BaseModel],
    ) -> BaseModel:
        self.calls.append({"system": system, "user": user, "schema": schema})
        if not self._q:
            raise LLMError("MockLLM: response queue is empty")
        resp = self._q.popleft()
        if isinstance(resp, BaseException):
            raise resp
        if isinstance(resp, dict):
            return schema.model_validate(resp)  # ty: ignore[invalid-argument-type]
        if isinstance(resp, BaseModel) and type(resp) is not schema:
            return schema(**resp.model_dump())  # ty: ignore[invalid-argument-type]
        return resp
