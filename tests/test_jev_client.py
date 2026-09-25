"""The Jev client reads its key from the environment, speaks both doors, and keeps the key out of results.

What would go wrong silently: a key pasted into a result file, a 429 on the
free tier read as an answer, the gateway's response shape parsed as
TypeSafe's (they agree on `noul`, differ on `usage` casing in the AI SDK
shape), and a missing key failing with a stack trace instead of the URL to
get one.
"""

from __future__ import annotations

import json

import httpx
import pytest

from aoi_agent.llm import jev


def _client(handler, **kwargs) -> jev.JevClient:
    client = jev.JevClient("https://example.test/typesafe", "vck_secret", "typesafe-ai/jev",
                           sleep=lambda _s: None, **kwargs)
    client._client = httpx.Client(transport=httpx.MockTransport(handler),
                                  headers={"Authorization": "Bearer vck_secret"})
    return client


def _typesafe_shape(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    assert request.url.path.endswith("/v1/systemone")
    assert set(body) == {"model", "state", "questions"}
    return httpx.Response(200, json={
        "model": "jev-1.13.0",
        "answers": {name: {"type": "noul", "noul": 0.87} for name in body["questions"]},
        "usage": {"input_tokens": 392, "output_tokens": 65}})


def test_the_typesafe_shape_is_read_back_with_the_model_string():
    client = _client(_typesafe_shape)
    answer = client.ask("M22 這兩天 open 多不多？", {"ok": jev.noul("Does the question name a machine?")})
    assert answer.noul("ok") == pytest.approx(0.87)
    assert answer.model == "jev-1.13.0"
    assert (answer.input_tokens, answer.output_tokens) == (392, 65)
    assert answer.latency_s >= 0


def test_the_ai_sdk_shape_is_read_back_too():
    def handler(_request):
        return httpx.Response(200, json={
            "model": "typesafe-ai/jev",
            "answers": {"ok": {"type": "boolean", "probability": 0.12}},
            "usage": {"inputTokens": 275, "outputTokens": 20}})
    answer = _client(handler).ask("x", {"ok": jev.noul("y")})
    assert answer.noul("ok") == pytest.approx(0.12) and answer.input_tokens == 275


def test_a_429_is_retried_and_the_answer_after_it_is_returned():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(429, json={"message": "slow down"})
        return _typesafe_shape(request)
    answer = _client(handler).ask("x", {"ok": jev.noul("y")})
    assert len(calls) == 3 and answer.noul("ok") == pytest.approx(0.87)


def test_a_persistent_429_raises_instead_of_answering():
    client = _client(lambda _r: httpx.Response(429, json={"message": "slow down"}))
    with pytest.raises(httpx.HTTPStatusError):
        client.ask("x", {"ok": jev.noul("y")})


def test_the_key_never_reaches_the_answer():
    answer = _client(_typesafe_shape).ask("x", {"ok": jev.noul("y")})
    assert "vck_secret" not in json.dumps(answer.raw) and "vck_secret" not in repr(answer)


def test_from_env_prefers_the_gateway_and_says_where_to_get_a_key():
    gateway = jev.JevClient.from_env({"AI_GATEWAY_API_KEY": "vck_1", "TYPESAFE_API_KEY": "ts_1"})
    assert (gateway.door, gateway.model) == ("gateway", "typesafe-ai/jev")
    direct = jev.JevClient.from_env({"TYPESAFE_API_KEY": "ts_1"})
    assert (direct.door, direct.model) == ("typesafe", "jev-latest")
    with pytest.raises(SystemExit, match="vercel.com"):
        jev.JevClient.from_env({})


def test_question_factories_write_the_documented_shapes():
    assert jev.noul("q") == {"type": "noul", "instructions": "q"}
    assert jev.noul("q", {"true": "a", "false": "b"})["criteria"] == {"true": "a", "false": "b"}
    assert jev.choice("q", {"x": "one", "y": "two"}) == {
        "type": "choice", "instructions": "q", "criteria": {"x": "one", "y": "two"}}


def test_the_gateway_is_pinned_to_typesafe_and_the_typesafe_door_is_not():
    """Measured 2026-09-25: unpinned, the gateway sent some requests to a backend that 503'd."""
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"model": "jev-1.13.0", "answers": {"ok": {"noul": 0.5}}})
    gateway = jev.JevClient(jev.GATEWAY_BASE_URL, "vck", jev.GATEWAY_MODEL, sleep=lambda _s: None)
    gateway._client = httpx.Client(transport=httpx.MockTransport(handler))
    gateway.ask("x", {"ok": jev.noul("y")})
    assert seen[0]["providerOptions"] == {"gateway": {"only": ["typesafe-ai"]}}
    direct = jev.JevClient("https://api.typesafe.ai", "ts", "jev-latest", sleep=lambda _s: None)
    direct._client = httpx.Client(transport=httpx.MockTransport(handler))
    direct.ask("x", {"ok": jev.noul("y")})
    assert "providerOptions" not in seen[1]


def test_requests_are_paced_and_latency_excludes_the_waits():
    now, slept = [100.0], []

    def sleep(s):
        slept.append(s)
        now[0] += s
    calls = []

    def handler(request):
        calls.append(1)
        now[0] += 0.5
        return httpx.Response(429, json={}) if len(calls) == 2 else _typesafe_shape(request)
    client = jev.JevClient("https://api.typesafe.ai", "ts", "jev-latest", sleep=sleep,
                           min_interval_s=3.0, clock=lambda: now[0])
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    client.ask("x", {"ok": jev.noul("y")})
    answer = client.ask("x", {"ok": jev.noul("y")})
    assert slept[0] == pytest.approx(2.5)            # 3 s interval less the 0.5 s the first call took
    assert len(calls) == 3 and answer.latency_s < 1.0
