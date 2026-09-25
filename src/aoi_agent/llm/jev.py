"""A client for Jev, TypeSafe AI's decision model, through either door it has.

Jev is not a chat model. It takes one piece of *state* and a set of typed
questions, and returns every answer at once, as probabilities rather than
prose: a ``noul`` question ("is this true?") comes back as a number in
[0, 1], a ``choice`` as a probability per option. Nothing is generated, so
nothing is parsed and no argument is invented -- which is also why it cannot
plan a call on its own: it can say *whether* a question is answerable and
*which* tool the answer needs, and gpt-oss still writes the arguments.

Two endpoints speak the same request and response shape (checked on
2026-09-25 against docs.typesafe.ai and vercel.com/docs/ai-gateway):

* TypeSafe's own: ``POST https://api.typesafe.ai/v1/systemone``, key in
  ``TYPESAFE_API_KEY``, model ``jev-latest``.
* Vercel AI Gateway's TypeSafe-compatible door:
  ``POST https://ai-gateway.vercel.sh/typesafe/v1/systemone``, key in
  ``AI_GATEWAY_API_KEY``, model ``typesafe-ai/jev``. Jev is on the gateway's
  free tier ($0.04 per million input tokens, output free).

The key is read from the environment and from nowhere else, and it is never
written into a result: ``JevAnswer`` carries the answers, the model string
the server reports (``jev-1.13.0`` at the time of writing -- a cloud model
changes under you, so every measurement stores it), token usage and the
wall time. 429s are retried with a backoff because the free tier rate-limits
harder than the paid one.

On the gateway every request is pinned to TypeSafe's own backend. Measured
2026-09-25: the gateway otherwise routes some requests to DigitalOcean, which
answered 503 and was not failed over, and two backends would also mean two
possibly different deployments of one model inside a single measurement.
TypeSafe itself answered 429 ("high demand") on roughly two requests in three
sent back to back, so the client waits a minimum interval between requests
and backs off for up to a minute.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/typesafe"
GATEWAY_MODEL = "typesafe-ai/jev"
GATEWAY_KEY_ENV = "AI_GATEWAY_API_KEY"

TYPESAFE_BASE_URL = "https://api.typesafe.ai"
TYPESAFE_MODEL = "jev-latest"
TYPESAFE_KEY_ENV = "TYPESAFE_API_KEY"

HOW_TO_GET_A_KEY = (
    "No Jev key in the environment. Either:\n"
    f"  * Vercel AI Gateway (Jev is on its free tier): sign up at https://vercel.com,\n"
    "    open AI Gateway -> API Keys -> Create key, and export it as\n"
    f"    {GATEWAY_KEY_ENV}=vck_... in your shell; or\n"
    f"  * TypeSafe directly: https://console.typesafe.ai/keys, exported as {TYPESAFE_KEY_ENV}.\n"
    "Never put the key in the repository or in a file it tracks."
)

RETRY_STATUSES = (429, 500, 502, 503, 504)
RETRY_BACKOFF_S = (2.0, 5.0, 10.0, 20.0, 40.0, 60.0)
GATEWAY_PROVIDER_PIN = {"gateway": {"only": ["typesafe-ai"]}}


def noul(instructions: str, criteria: dict[str, str] | None = None) -> dict[str, Any]:
    """A yes/no question. The answer is P(yes); write it so that high means yes."""
    question: dict[str, Any] = {"type": "noul", "instructions": instructions}
    if criteria:
        question["criteria"] = {"true": criteria["true"], "false": criteria["false"]}
    return question


def choice(instructions: str, criteria: dict[str, str]) -> dict[str, Any]:
    """One option out of a named set; the answer carries a probability per option."""
    return {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}


@dataclass
class JevAnswer:
    """What one request returned, and what it cost."""

    answers: dict[str, dict[str, Any]]
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def noul(self, name: str) -> float:
        """P(yes) for one noul question, whichever field name the door used."""
        answer = self.answers[name]
        if "noul" in answer:
            return float(answer["noul"])
        if "probability" in answer:          # the AI SDK's `evaluate` shape
            return float(answer["probability"])
        raise KeyError(f"{name}: no noul probability in {sorted(answer)}")


class JevClient:
    """``systemone``: state in, typed answers out."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 30.0,
                 sleep=time.sleep, min_interval_s: float = 0.0, clock=time.monotonic):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.Client(timeout=timeout,
                                    headers={"Authorization": f"Bearer {api_key}",
                                             "Content-Type": "application/json"})
        self._sleep = sleep
        self._clock = clock
        self.min_interval_s = min_interval_s
        self._last_sent: float | None = None

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None, **kwargs) -> JevClient:
        """The gateway if its key is set, else TypeSafe's, else an exit that says where to get one."""
        env = os.environ if environ is None else environ
        if env.get(GATEWAY_KEY_ENV):
            return cls(GATEWAY_BASE_URL, env[GATEWAY_KEY_ENV], GATEWAY_MODEL, **kwargs)
        if env.get(TYPESAFE_KEY_ENV):
            return cls(TYPESAFE_BASE_URL, env[TYPESAFE_KEY_ENV], TYPESAFE_MODEL, **kwargs)
        raise SystemExit(HOW_TO_GET_A_KEY)

    @property
    def door(self) -> str:
        return "gateway" if self.base_url == GATEWAY_BASE_URL else "typesafe"

    def ask(self, state: Any, questions: dict[str, dict[str, Any]]) -> JevAnswer:
        payload: dict[str, Any] = {"model": self.model, "state": state, "questions": questions}
        if self.door == "gateway":
            payload["providerOptions"] = GATEWAY_PROVIDER_PIN
        for backoff in (*RETRY_BACKOFF_S, None):
            self._pace()
            started = time.perf_counter()      # the answered attempt only, not the waits before it
            response = self._client.post(f"{self.base_url}/v1/systemone", json=payload)
            if response.status_code in RETRY_STATUSES and backoff is not None:
                self._sleep(backoff)
                continue
            break
        response.raise_for_status()
        latency = time.perf_counter() - started
        body = response.json()
        usage = body.get("usage") or {}
        return JevAnswer(
            answers=body.get("answers") or {},
            model=str(body.get("model", "")),
            input_tokens=int(usage.get("input_tokens", usage.get("inputTokens", 0)) or 0),
            output_tokens=int(usage.get("output_tokens", usage.get("outputTokens", 0)) or 0),
            latency_s=latency,
            raw=body,
        )

    def _pace(self) -> None:
        """Wait out what is left of ``min_interval_s`` since the last request went out."""
        if self._last_sent is not None and self.min_interval_s > 0:
            wait = self.min_interval_s - (self._clock() - self._last_sent)
            if wait > 0:
                self._sleep(wait)
        self._last_sent = self._clock()
