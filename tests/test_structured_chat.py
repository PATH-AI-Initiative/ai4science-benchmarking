"""Regression test for OpenAIChat's api_key resolution.

A prior version always substituted the literal string "not-needed" when no
api_key was passed explicitly -- correct for a local/self-hosted
OpenAI-compatible server (base_url set, no real auth), but wrong for real
OpenAI: it silently overrode the SDK's own OPENAI_API_KEY environment lookup,
so a real API key set in the environment was never used."""

from __future__ import annotations

from benchmarking_pipeline.services.structured_chat import OpenAIChat


def test_real_openai_resolves_key_from_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key-from-env")
    chat = OpenAIChat("gpt-4o-mini")
    assert chat._client.api_key == "sk-real-key-from-env"


def test_local_server_falls_back_to_placeholder_when_no_key_given(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    chat = OpenAIChat("llama3.1", base_url="http://localhost:11434/v1")
    assert chat._client.api_key == "not-needed"


def test_explicit_api_key_always_wins(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key-from-env")
    chat = OpenAIChat("gpt-4o-mini", api_key="sk-explicit-override")
    assert chat._client.api_key == "sk-explicit-override"
