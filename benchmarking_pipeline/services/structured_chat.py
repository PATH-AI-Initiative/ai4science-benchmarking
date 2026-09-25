"""Shared primitive: ask a model to fill in a JSON schema.

The LLM judge (:mod:`llm_judge`) and structured extraction
(:mod:`benchmarking_pipeline.io.extraction`) need the same thing: send a
system+user prompt, constrain the response to a JSON schema, get back a
dict. Implemented once here, so adding a new backend (Anthropic, OpenAI, a
local Ollama model, ...) is one class, reusable by both.
"""

from __future__ import annotations

import json
from typing import Protocol, runtime_checkable


@runtime_checkable
class StructuredChatClient(Protocol):
    name: str

    def complete(self, *, system: str, user: str, schema: dict, max_tokens: int = 4096) -> dict:
        """Return a dict conforming to ``schema``."""
        ...


class AnthropicChat:
    """Claude via the Messages API, constrained with ``output_config.format``."""

    def __init__(self, model: str = "claude-opus-4-8"):
        import anthropic  # noqa: PLC0415

        self._client = anthropic.Anthropic()  # resolves credentials from the environment
        self.model = model
        self.name = f"anthropic:{model}"

    def complete(self, *, system: str, user: str, schema: dict, max_tokens: int = 4096) -> dict:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            output_config={"format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": user}],
        )
        text = next(b.text for b in resp.content if b.type == "text")
        return json.loads(text)


class OpenAIChat:
    """OpenAI's chat completions API, or any OpenAI-compatible server.

    Passing ``base_url`` points this at a compatible local server (e.g.
    Ollama's OpenAI-compatible endpoint at ``http://localhost:11434/v1``) — but
    prefer :class:`OllamaChat` for Ollama, whose native structured-output
    support (grammar-constrained decoding) is more reliable across models than
    Ollama's OpenAI-compatibility layer.
    """

    def __init__(
        self,
        model: str = "gpt-4.1",
        *,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        from openai import OpenAI  # noqa: PLC0415

        # A non-empty api_key is required by the client even when the server
        # (e.g. a local OpenAI-compatible endpoint) doesn't check it -- but only
        # fall back to a placeholder for such servers (base_url set). For real
        # OpenAI, leave api_key as None so the client resolves OPENAI_API_KEY
        # from the environment itself; substituting "not-needed" there would
        # silently override that lookup and break real authentication.
        self._client = OpenAI(base_url=base_url, api_key=api_key or ("not-needed" if base_url else None))
        self.model = model
        self.name = f"openai:{model}" if base_url is None else f"openai-compatible:{model}"

    def complete(self, *, system: str, user: str, schema: dict, max_tokens: int = 4096) -> dict:
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema, "strict": True},
            },
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return json.loads(resp.choices[0].message.content)


class OllamaChat:
    """A local model served by Ollama (``ollama serve``).

    Runs entirely on your machine — no API key, no network call. Requires the
    ``ollama`` package (``pip install ollama``) and the model to be pulled
    first (``ollama pull llama3.1``). Uses Ollama's native structured-output
    parameter (``format=<json schema>``), which constrains generation via
    grammar rather than relying on the model's own tool-calling ability, so it
    works even with models that weren't trained for structured output.

    Ollama's *runtime* context window defaults to a few thousand tokens
    regardless of the model's maximum (``ollama show <model>`` reports the
    max, not the default in use) — with a long input that leaves almost no
    room for output, so the response is truncated after a token or two and
    fails to parse as JSON. ``complete()`` sizes ``num_ctx`` off the actual
    prompt length instead of relying on Ollama's default.
    """

    def __init__(self, model: str = "llama3.1", *, host: str | None = None):
        import ollama  # noqa: PLC0415

        self._client = ollama.Client(host=host) if host else ollama.Client()
        self.model = model
        self.name = f"ollama:{model}"

    def complete(self, *, system: str, user: str, schema: dict, max_tokens: int = 4096) -> dict:
        # Rough token estimate (~4 chars/token) with generous headroom, rounded
        # up to a power-of-two-ish bucket so small prompt-length changes don't
        # force a full model reload (Ollama reloads the model when num_ctx changes).
        estimated_input_tokens = (len(system) + len(user)) // 3
        num_ctx = max(4096, _round_up_pow2(estimated_input_tokens + max_tokens + 512))

        resp = self._client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            format=schema,
            options={"num_predict": max_tokens, "num_ctx": num_ctx},
        )
        content = resp.message.content
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            preview = content[:200] + ("..." if len(content) > 200 else "")
            raise ValueError(
                f"Ollama model {self.model!r} did not return valid JSON "
                f"(num_ctx={num_ctx}). Raw response: {preview!r}"
            ) from e


def _round_up_pow2(n: int) -> int:
    power = 1
    while power < n:
        power *= 2
    return power
