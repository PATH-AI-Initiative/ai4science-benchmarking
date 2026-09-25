"""LLM-as-judge.

Used by metrics that require model judgement at scale: logical consistency,
citation-support classification, and assists to the SME-led tractability
review.

``LLMJudge`` is a protocol so the backend is swappable and pinnable. Each
concrete judge below wraps a
:class:`~benchmarking_pipeline.services.structured_chat.StructuredChatClient`,
the schema-constrained "ask a model to fill in this JSON" primitive shared
with structured extraction. Pin the model and record it in run metadata so
judge output is traceable.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .structured_chat import AnthropicChat, OllamaChat, OpenAIChat, StructuredChatClient

# JSON schema the judge is constrained to return.
_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string"},
        "confidence": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["verdict", "confidence", "rationale"],
    "additionalProperties": False,
}

_JUDGE_SYSTEM_PROMPT = (
    "You are a careful scientific evaluator. Respond only with the requested JSON."
)


@dataclass
class Judgement:
    """A single structured verdict from the judge."""

    verdict: str  # e.g. "supports" | "refutes" | "neutral", or "consistent" | "contradiction"
    confidence: float  # 0..1
    rationale: str | None = None


@runtime_checkable
class LLMJudge(Protocol):
    name: str

    def judge(self, prompt: str, *, choices: list[str] | None = None) -> Judgement:
        """Return a structured judgement for ``prompt``.

        ``choices`` optionally constrains the verdict to a fixed label set.
        """
        ...


def _augment_prompt(prompt: str, choices: list[str] | None) -> str:
    if choices:
        return f"{prompt}\n\nThe verdict must be exactly one of: {', '.join(choices)}."
    return prompt


class _ChatBackedJudge:
    """Shared ``judge()`` implementation for every backend below."""

    _chat: StructuredChatClient

    @property
    def name(self) -> str:
        return self._chat.name

    def judge(self, prompt: str, *, choices: list[str] | None = None) -> Judgement:
        data = self._chat.complete(
            system=_JUDGE_SYSTEM_PROMPT,
            user=_augment_prompt(prompt, choices),
            schema=_VERDICT_SCHEMA,
            max_tokens=1024,
        )
        return Judgement(
            verdict=data["verdict"],
            confidence=float(data["confidence"]),
            rationale=data.get("rationale"),
        )


class AnthropicJudge(_ChatBackedJudge):
    """Claude-backed judge (default). Requires the ``anthropic`` package."""

    def __init__(self, model: str = "claude-opus-4-8"):
        self._chat = AnthropicChat(model)


class OpenAIJudge(_ChatBackedJudge):
    """OpenAI-backed judge. Requires the ``openai`` package.

    Pass ``base_url`` to point at any OpenAI-compatible server instead of
    real OpenAI (e.g. Ollama's compatibility layer) — but prefer
    :class:`OllamaJudge` for Ollama specifically; it's more reliable.
    """

    def __init__(
        self,
        model: str = "gpt-4.1",
        *,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self._chat = OpenAIChat(model, base_url=base_url, api_key=api_key)


class OllamaJudge(_ChatBackedJudge):
    """Judge backed by a local Ollama model — no API key, no network call.

    Requires ``pip install ollama`` and a running Ollama server (``ollama
    serve``) with the model pulled (``ollama pull llama3.1``).
    """

    def __init__(self, model: str = "llama3.1", *, host: str | None = None):
        self._chat = OllamaChat(model, host=host)


class ProgressJudge:
    """Wraps any judge, printing a line per call as it completes.

    Judge-heavy metrics (logical_consistency's role + pairwise checks, in
    particular) can make many calls, and a single call against a slow local
    model can take tens of seconds — with no feedback that looks identical to
    a hang. Wrap the real judge with this before attaching it to a
    ``Context`` to get visible per-call progress instead. Purely additive:
    the wrapped judge's behaviour is unchanged, this only prints.

    Pass ``total`` (the expected call count, if known ahead of time) for
    "N/total" progress; omit it for a running count with no denominator.
    """

    def __init__(self, judge: LLMJudge, *, total: int | None = None, stream=sys.stderr):
        self._judge = judge
        self.name = judge.name
        self._total = total
        self._count = 0
        self._stream = stream

    def judge(self, prompt: str, *, choices: list[str] | None = None) -> Judgement:
        self._count += 1
        label = f"{self._count}/{self._total}" if self._total else str(self._count)
        print(f"  judge call {label}...", end="", flush=True, file=self._stream)

        start = time.monotonic()
        result = self._judge.judge(prompt, choices=choices)
        elapsed = time.monotonic() - start

        print(f" {result.verdict}  ({elapsed:.1f}s)", file=self._stream)
        return result
