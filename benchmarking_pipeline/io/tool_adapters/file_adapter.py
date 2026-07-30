"""File-based adapter: load previously-captured tool output from disk.

This unblocks metric development without waiting on live integrations for every
tool — you paste a tool's output into a JSON file and evaluate it.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from ...core.models import EvaluationRun, Tool
from ..parsers import parse_json


class FileAdapter:
    name = "file"

    def __init__(self, tool: Tool, output_path: str | Path, evaluated_on: date | None = None):
        self.tool = tool
        self.output_path = Path(output_path)
        self.evaluated_on = evaluated_on

    def run(self, prompt: str) -> EvaluationRun:
        outputs = parse_json(self.output_path)
        return EvaluationRun(
            tool=self.tool,
            prompt=prompt,
            outputs=outputs,
            evaluated_on=self.evaluated_on,
            metadata={"source_file": str(self.output_path)},
        )
