"""Ollama chat client with tool calling.

Timing here is deliberately fussy. A naive measurement on this machine reported
85 s cold and 145 s warm for a single 132-token tool call, and ``ollama ps``
showed a different model resident than the one being called: models were being
evicted and reloaded between requests, so almost all of the measured time was
load time.

Two things follow, and both are enforced below:

- ``keep_alive`` is set explicitly on every request, and callers should group
  their work by model rather than interleaving.
- ``total_duration`` is never reported as latency. ``load_duration`` is broken
  out, and the number that matters is ``eval_duration``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

BASE_URL = "http://localhost:11434"
KEEP_ALIVE = "30m"

#: How long this client waits for a response before giving up.
#:
#: **Not WI-300's response budget.** That budget is a promise to the operator
#: about when a *verdict* reaches the record, it lives on the disposition path
#: in ``graph.flow``, and it is met by the classifier in single-digit
#: milliseconds. This number is a resource bound on a client waiting for prose,
#: and the two were the same constant until 2026-08-23. Sharing it made more
#: than half of the station's explanations fail by construction: measured
#: service time on ``gpt-oss:20b`` is a median of 12.5s and a p90 of 15.6s, so
#: a 10s client timeout cut 20 calls in 24 -- and writing the operator's
#: explanation is the only job the LLM still has.
#:
#: Sized from that measurement rather than chosen. The slowest of 24 calls on a
#: verified-quiet machine was 21.1s, so anything under that discards healthy
#: work; contention can multiply an LLM call's wall time by 25x, which no
#: deadline should absorb and the old 600s value did -- it turned a busy GPU
#: into a ten-minute blocked workstation. 60s is 2.8x the slowest healthy call
#: and a minute is what a stuck worker costs. Nothing waits on it: the
#: disposition is decided before this call is made.
#:
#: Raising it is a measurement, not a preference. See
#: ``scripts/latency_report.py`` and docs/benchmarks.md.
EXPLANATION_DEADLINE_S = 60.0

Think = bool | Literal["low", "medium", "high"] | None


@dataclass
class Timing:
    """Per-request timings, in milliseconds, split by what they measure."""

    wall_ms: float
    load_ms: float
    prompt_eval_ms: float
    eval_ms: float
    prompt_tokens: int
    eval_tokens: int

    @property
    def tokens_per_second(self) -> float:
        return self.eval_tokens / (self.eval_ms / 1000) if self.eval_ms else 0.0

    #: Measured on this machine: a warm, resident `gpt-oss:20b` still reports
    #: ``load_duration`` of a steady ~168ms, so zero is not the warm value and a
    #: gate at 100ms flags every healthy request. Pulling 12GB back onto the GPU
    #: takes seconds, so the two populations are orders of magnitude apart and
    #: anything between them separates them safely.
    RELOAD_MS = 2000.0

    @property
    def was_reloaded(self) -> bool:
        """True when the model had to be pulled back into memory.

        A reload means this request's latency is not comparable with a warm one
        and should be dropped from any benchmark.
        """
        return self.load_ms > self.RELOAD_MS


@dataclass
class ChatResult:
    text: str
    tool_calls: list[dict[str, Any]]
    thinking: str
    timing: Timing
    raw: dict[str, Any] = field(repr=False, default_factory=dict)


class OllamaClient:
    """Thin wrapper over ``/api/chat``."""

    def __init__(
        self,
        model: str,
        base_url: str = BASE_URL,
        keep_alive: str = KEEP_ALIVE,
        timeout: float = EXPLANATION_DEADLINE_S,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.keep_alive = keep_alive
        self._client = httpx.Client(timeout=timeout)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        think: Think = None,
        temperature: float = 0.0,
        response_format: dict | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        options: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": options,
        }
        if tools:
            payload["tools"] = tools
        if think is not None:
            payload["think"] = think
        if response_format is not None:
            payload["format"] = response_format

        started = time.perf_counter()
        response = self._client.post(f"{self.base_url}/api/chat", json=payload)
        response.raise_for_status()
        wall_ms = (time.perf_counter() - started) * 1000

        body = response.json()
        message = body.get("message", {})
        return ChatResult(
            text=message.get("content", "") or "",
            tool_calls=message.get("tool_calls") or [],
            thinking=message.get("thinking") or "",
            timing=Timing(
                wall_ms=wall_ms,
                load_ms=body.get("load_duration", 0) / 1e6,
                prompt_eval_ms=body.get("prompt_eval_duration", 0) / 1e6,
                eval_ms=body.get("eval_duration", 0) / 1e6,
                prompt_tokens=body.get("prompt_eval_count", 0),
                eval_tokens=body.get("eval_count", 0),
            ),
            raw=body,
        )

    def warm_up(self) -> Timing:
        """Load the model into memory and generate as little as possible.

        Every benchmark should call this first and ignore what it returns.

        The output is capped at one token deliberately. This previously sent a
        bare "ok" with no cap, which is an unbounded prompt to a reasoning
        model: it answered with an essay, took longer than the benchmark calls
        it was meant to precede, and on one run exceeded a 180s ceiling. Warming
        up means paying the load, not paying for tokens nobody reads.
        """
        return self.chat(
            [{"role": "user", "content": "Reply with one word."}],
            think=False,
            max_tokens=1,
        ).timing

    def resident_models(self) -> list[dict[str, Any]]:
        response = self._client.get(f"{self.base_url}/api/ps")
        response.raise_for_status()
        return response.json().get("models", [])

    def assert_resident(self) -> None:
        """Fail loudly if this client's model is not the one loaded on the GPU."""
        names = [m.get("name", "") for m in self.resident_models()]
        if not any(name.startswith(self.model.split(":")[0]) for name in names):
            raise RuntimeError(
                f"{self.model} is not resident (loaded: {names or 'nothing'}). "
                "Latency measured now would be load time, not inference."
            )
