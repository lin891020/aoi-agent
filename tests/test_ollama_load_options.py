"""Options that shape the runner travel on every call, so a measurement pins them.

`num_batch` and `use_mmap` decide how the server loads the model, and the
server chooses them by free memory when a request leaves them out -- batch 512
or 1024 by whether 18.6 GiB was free on 2026-09-23, and 22 of 70 plans with it.
A client that carries them on every request makes the runner a controlled
variable instead of a machine-state one.
"""

from __future__ import annotations

import json

import httpx

from aoi_agent.llm.ollama import OllamaClient


def _capture(seen: list[dict]):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={
            "message": {"content": "{}"}, "total_duration": 1, "load_duration": 0,
            "prompt_eval_duration": 0, "eval_duration": 1, "eval_count": 1})
    return handler


def _client(**kwargs) -> tuple[OllamaClient, list[dict]]:
    seen: list[dict] = []
    client = OllamaClient("stub-model", **kwargs)
    client._client = httpx.Client(transport=httpx.MockTransport(_capture(seen)))
    return client, seen


def test_load_options_reach_every_request():
    client, seen = _client(load_options={"num_batch": 1024, "use_mmap": False})
    client.chat([{"role": "user", "content": "hi"}])
    client.chat([{"role": "user", "content": "again"}], temperature=0.5)
    assert all(m["options"]["num_batch"] == 1024 and m["options"]["use_mmap"] is False
               for m in seen)
    assert seen[1]["options"]["temperature"] == 0.5


def test_a_call_cannot_override_the_temperature_with_a_load_option():
    """`temperature` is the call's; a load option of the same name does not win."""
    client, seen = _client(load_options={"temperature": 0.9})
    client.chat([{"role": "user", "content": "hi"}])
    assert seen[0]["options"]["temperature"] == 0.0


def test_no_load_options_means_the_payload_is_what_it_was():
    client, seen = _client()
    client.chat([{"role": "user", "content": "hi"}])
    assert seen[0]["options"] == {"temperature": 0.0}
