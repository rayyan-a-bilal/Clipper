"""M5a — OllamaLLMClient (request shape, retries, fallback) + MockLLM."""

import json

import httpx
import pytest

from clipper.adapters.llm import (
    LLMClient,
    LLMError,
    MockLLM,
    OllamaLLMClient,
    get_llm_client,
)
from clipper.config import Config, LLMConfig, PathsConfig
from clipper.models import ScoringResponse


@pytest.fixture
def cfg(tmp_path):
    return Config(
        paths=PathsConfig(data_dir=str(tmp_path)),
        llm=LLMConfig(model="gemma4:12b"),
    )


def _ok_body(scores: list[dict] | None = None) -> dict:
    if scores is None:
        scores = [
            {
                "cand_id": "j-c0001-0004",
                "hook": 7,
                "payoff": 8,
                "self_contained": 7,
                "emotion": 6,
                "clarity": 9,
                "suggested_in": 1.25,
                "suggested_out": 30.5,
                "reject_reason": None,
            }
        ]
    return {"message": {"role": "assistant", "content": json.dumps({"scores": scores})}}


class FakeClient:
    """Duck-typed httpx.Client: records POSTs, returns canned responses."""

    def __init__(self, responder, sink: list):
        self._responder = responder
        self._sink = sink

    def post(self, url, json=None):
        # snapshot the body: the client may mutate the same dict across attempts
        from copy import deepcopy

        body = deepcopy(json)
        self._sink.append({"url": url, "json": body})
        resp = self._responder({"url": url, "json": body})
        resp.request = httpx.Request("POST", url)
        return resp

    def close(self):
        pass


def make_client(cfg, responder, sink: list):
    return OllamaLLMClient(
        cfg,
        base_url="http://127.0.0.1:11435",
        client_factory=lambda timeout_s: FakeClient(responder, sink),
    )


# ---------------------------------------------------------------------------
# request body correctness + happy path
# ---------------------------------------------------------------------------
def test_request_body_and_valid_response(cfg):
    sink: list = []
    client = make_client(cfg, lambda req: httpx.Response(200, json=_ok_body()), sink)
    out = client.generate_json(
        "system-prompt", "user-prompt", ScoringResponse
    )
    assert isinstance(out, ScoringResponse)
    assert len(out.scores) == 1
    assert out.scores[0].hook == 7 and out.scores[0].payoff == 8

    req = sink[0]
    assert req["url"] == "http://127.0.0.1:11435/api/chat"
    body = req["json"]
    assert body["model"] == "gemma4:12b"
    assert body["stream"] is False
    assert body["messages"] == [
        {"role": "system", "content": "system-prompt"},
        {"role": "user", "content": "user-prompt"},
    ]
    # strict schema mode by default
    assert body["format"] == ScoringResponse.model_json_schema()
    opts = body["options"]
    assert opts["temperature"] == pytest.approx(0.1)
    assert opts["num_ctx"] == 16384
    assert opts["num_batch"] == 256
    assert opts["num_predict"] == 1024
    assert client.last_mode == "schema"


def test_get_llm_client_returns_ollama_client(cfg):
    client = get_llm_client(cfg, base_url="http://127.0.0.1:11435")
    assert isinstance(client, OllamaLLMClient)
    assert client.base_url == "http://127.0.0.1:11435"


# ---------------------------------------------------------------------------
# failure modes: each retries (default max_retries=2 => 3 attempts)
# ---------------------------------------------------------------------------
def test_invalid_json_retries_then_llmerror(cfg):
    sink: list = []
    client = make_client(
        cfg,
        lambda req: httpx.Response(
            200, json={"message": {"content": "not json, sorry"}}
        ),
        sink,
    )
    with pytest.raises(LLMError, match="invalid JSON"):
        client.generate_json("s", "u", ScoringResponse)
    assert len(sink) == 3  # 1 + max_retries(2)


def test_invalid_schema_retries_then_llmerror(cfg):
    sink: list = []
    client = make_client(
        cfg,
        # HTTP 200 but body does not match ScoringResponse
        lambda req: httpx.Response(
            200, json={"message": {"content": '{"nope": 1}'}}
        ),
        sink,
    )
    with pytest.raises(LLMError, match="schema validation"):
        client.generate_json("s", "u", ScoringResponse)
    assert len(sink) == 3


def test_http_500_retries_then_llmerror(cfg):
    sink: list = []
    client = make_client(
        cfg,
        lambda req: httpx.Response(500, text="internal error"),
        sink,
    )
    with pytest.raises(LLMError, match="HTTP 500"):
        client.generate_json("s", "u", ScoringResponse)
    assert len(sink) == 3


def test_timeout_retries_then_llmerror(cfg):
    sink: list = []

    def boom(req):
        raise httpx.ConnectTimeout("timed out")

    client = make_client(cfg, boom, sink)
    with pytest.raises(LLMError, match="timeout|timed out"):
        client.generate_json("s", "u", ScoringResponse)
    assert len(sink) == 3


def test_retry_then_success(cfg):
    calls = {"n": 0}
    sink: list = []

    def flaky(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(500, text="boom")
        return httpx.Response(200, json=_ok_body())

    client = make_client(cfg, flaky, sink)
    out = client.generate_json("s", "u", ScoringResponse)
    assert isinstance(out, ScoringResponse)
    assert calls["n"] == 2


# ---------------------------------------------------------------------------
# documented fallback: schema format rejected -> format="json" + pydantic
# ---------------------------------------------------------------------------
def test_schema_format_rejected_falls_back_to_json(cfg):
    sink: list = []

    def responder(req):
        fmt = req["json"]["format"]
        if fmt == "json":
            return httpx.Response(200, json=_ok_body())
        # server rejects the JSON-schema format value
        return httpx.Response(
            500, text="format field: json schema not supported"
        )

    client = make_client(cfg, responder, sink)
    out = client.generate_json("s", "u", ScoringResponse)
    assert isinstance(out, ScoringResponse)
    assert out.scores[0].self_contained == 7
    assert client.last_mode == "json"
    assert sink[0]["json"]["format"] != "json"
    assert sink[1]["json"]["format"] == "json"


# ---------------------------------------------------------------------------
# MockLLM — test double, zero network I/O
# ---------------------------------------------------------------------------
def test_mockllm_valid_dict_and_model(cfg):
    from clipper.models import ScoredClip as SC

    m = MockLLM()
    m.enqueue(
        {
            "scores": [
                {
                    "cand_id": "x",
                    "hook": 5,
                    "payoff": 5,
                    "self_contained": 5,
                    "emotion": 5,
                    "clarity": 5,
                    "suggested_in": 1.0,
                    "suggested_out": 20.0,
                }
            ]
        },
        ScoringResponse(scores=[SC(cand_id="y", hook=1, payoff=1, self_contained=1,
                                  emotion=1, clarity=1, suggested_in=0.0,
                                  suggested_out=15.0)]),
    )
    r1 = m.generate_json("sys1", "usr1", ScoringResponse)
    r2 = m.generate_json("sys2", "usr2", ScoringResponse)
    assert r1.scores[0].cand_id == "x"
    assert r2.scores[0].cand_id == "y"
    assert len(m.calls) == 2
    assert m.calls[0] == {"system": "sys1", "user": "usr1", "schema": ScoringResponse}


def test_mockllm_exception_and_empty_queue(cfg):
    m = MockLLM()
    m.enqueue(ValueError("invalid JSON in canned data"))
    with pytest.raises(ValueError, match="canned"):
        m.generate_json("s", "u", ScoringResponse)
    assert m.pending == 0
    with pytest.raises(LLMError, match="empty"):
        m.generate_json("s", "u", ScoringResponse)


def test_mockllm_is_llmclient(cfg):
    m = MockLLM()
    m.enqueue(ScoringResponse(scores=[]))
    assert isinstance(m, LLMClient)
    out = m.generate_json("s", "u", ScoringResponse)
    assert out.scores == []
