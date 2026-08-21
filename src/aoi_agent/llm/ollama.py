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

    @property
    def was_reloaded(self) -> bool:
        """True when the model had to be pulled back into memory.

        A reload means this request's latency is not comparable with a warm one
        and should be dropped from any benchmark.
        """
        return self.load_ms > 100


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
        timeout: float = 600.0,
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
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": temperature},
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
        """Load the model and discard the result.

        Every benchmark should call this first and ignore what it returns.
        """
        return self.chat([{"role": "user", "content": "ok"}], think=False).timing

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
