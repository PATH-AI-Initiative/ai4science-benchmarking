"""Serialise a ToolScore to a structured, machine-readable dict/JSON."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ..core.config import RunConfig
from ..core.metric import MetricResult
from ..core.scoring import ToolScore

# Attached to every result carrying a composite/axis score: RunConfig's own
# docstring says its weights and thresholds "are placeholders to make the
# pipeline runnable, not calibrated values" pending structured expert
# elicitation. Surfaced here so a reader of the JSON sees the caveat with the
# number, not only in a docstring they may never open.
COMPOSITE_CAVEAT = (
    "Weights/thresholds in `config` below are placeholders, not values derived "
    "from structured expert elicitation (see RunConfig's own docstring) -- "
    "treat composite and axis scores as provisional, not a calibrated verdict."
)


def _result_to_dict(r: MetricResult) -> dict:
    d = asdict(r)
    d["axis"] = r.axis.value
    return d


def hypothesis_metric_summary(hypotheses: list[dict]) -> dict[str, dict]:
    """Per-metric coverage/mean across one tool's hypotheses (``hypotheses`` is
    the already-serialised list from :func:`to_dict`, i.e. each has a
    ``results`` list of ``_result_to_dict`` dicts).

    A hypothesis-level metric scoring only some hypotheses (e.g. citation_accuracy
    finding no references to check on 2 of 3) is easy to miss buried in the
    per-hypothesis results list -- this surfaces ``n_scored``/``n_total``
    alongside the mean so a thin sample is visible at a glance, not just the
    mean itself. Shared with ``report.compare``, which aggregates the same way
    across tools.
    """
    per_metric: dict[str, dict] = {}
    for hyp in hypotheses:
        for result in hyp["results"]:
            bucket = per_metric.setdefault(
                result["metric"], {"axis": result["axis"], "scores": [], "n_total": 0}
            )
            bucket["n_total"] += 1
            if result["score"] is not None:
                bucket["scores"].append(result["score"])

    return {
        name: {
            "axis": bucket["axis"],
            "mean_score": (
                round(sum(bucket["scores"]) / len(bucket["scores"]), 4)
                if bucket["scores"] else None
            ),
            "n_scored": len(bucket["scores"]),
            "n_total": bucket["n_total"],
        }
        for name, bucket in per_metric.items()
    }


def _config_to_dict(config: RunConfig) -> dict:
    return {
        "accuracy_floor": config.accuracy_floor,
        "safety_gate": config.safety_gate,
        "axis_weights": {a.value: w for a, w in config.axis_weights.items()},
        "metric_weights": config.metric_weights,
        "top_k_hypotheses": config.top_k_hypotheses,
        "logical_consistency_max_pairs": config.logical_consistency_max_pairs,
        "multi_run_comparison": config.multi_run_comparison,
        "adversarial_traps": config.adversarial_traps,
    }


def to_dict(score: ToolScore) -> dict:
    hypotheses = [
        {
            "id": h.hypothesis_id,
            "rank": h.rank,
            "passed_accuracy_floor": h.passed_accuracy_floor,
            "axis_scores": {a.value: v for a, v in h.axis_scores.items()},
            "results": [_result_to_dict(r) for r in h.results],
        }
        for h in score.hypothesis_scores
    ]
    return {
        "tool": score.tool,
        "disqualified": score.disqualified,
        "disqualified_reason": score.disqualified_reason,
        "composite": score.composite,
        "composite_caveat": COMPOSITE_CAVEAT,
        "config": _config_to_dict(score.config),
        "axis_scores": {a.value: v for a, v in score.axis_scores.items()},
        "hypotheses": hypotheses,
        "hypothesis_metric_coverage": hypothesis_metric_summary(hypotheses),
        "set_results": [_result_to_dict(r) for r in score.set_results],
        "novelty_profile": [_result_to_dict(r) for r in score.novelty_profile],
    }


def write_json(score: ToolScore, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(to_dict(score), indent=2))
