"""Tool adapters — how an EvaluationRun is produced for a given tool.

Tools differ: single-prompt vs multi-turn, injectable knowledge base vs closed,
API-accessible vs manual. The :class:`ToolAdapter` protocol hides those
differences so the pipeline treats every tool the same. Start with the
file-based adapter (paste in captured output) and add live API adapters per tool
as access allows.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ...core.models import EvaluationRun


@runtime_checkable
class ToolAdapter(Protocol):
    name: str

    def run(self, prompt: str) -> EvaluationRun:
        """Submit ``prompt`` to the tool and capture its output as an EvaluationRun."""
        ...
