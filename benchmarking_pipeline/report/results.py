"""Serialise a ToolScore to a structured, machine-readable dict/JSON."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ..core.metric import MetricResult
from ..core.scoring import ToolScore


def _result_to_dict(r: MetricResult) -> dict:
    d = asdict(r)
    d["axis"] = r.axis.value
    return d


def to_dict(score: ToolScore) -> dict:
    return {
        "tool": score.tool,
        "disqualified": score.disqualified,
        "disqualified_reason": score.disqualified_reason,
        "composite": score.composite,
        "axis_scores": {a.value: v for a, v in score.axis_scores.items()},
        "hypotheses": [
            {
                "id": h.hypothesis_id,
                "rank": h.rank,
                "passed_accuracy_floor": h.passed_accuracy_floor,
                "axis_scores": {a.value: v for a, v in h.axis_scores.items()},
                "results": [_result_to_dict(r) for r in h.results],
            }
            for h in score.hypothesis_scores
        ],
        "set_results": [_result_to_dict(r) for r in score.set_results],
        "novelty_profile": [_result_to_dict(r) for r in score.novelty_profile],
    }


def write_json(score: ToolScore, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(to_dict(score), indent=2))
