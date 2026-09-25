"""Cross-tool comparison: merge multiple ``to_dict()``-shaped results (see
``report.results``) into one metric x tool view.

Reads already-serialized results, not in-memory ``ToolScore`` objects -- each
tool is evaluated via its own ``benchmarking run`` invocation, often in a
separate session, and written to its own JSON file.

Hypothesis-level metrics (citation_accuracy, entity_accuracy, ...) don't
carry a single tool-level number the way set/multi-run metrics (diversity,
robustness, adversarial, ...) do; their official aggregate is the composite
``core.scoring`` already computes into ``axis_scores``. For a side-by-side
table, this reports a plain unweighted mean across a tool's hypotheses
instead -- transparent, but a looser number than the axis-level composite.
Never treat the two as interchangeable.

A composite/axis score is only comparable across tools if produced under the
same ``RunConfig`` (weights, thresholds, ``top_k_hypotheses``, ...) --
``compare()`` checks each input's ``config`` and flags a mismatch rather than
silently comparing composites as if they meant the same thing.
"""

from __future__ import annotations

import json
from pathlib import Path

from .results import COMPOSITE_CAVEAT, hypothesis_metric_summary


def _collect_set_metric(metrics: dict, tool: str, result: dict) -> None:
    name = result["metric"]
    entry = metrics.setdefault(name, {"axis": result["axis"], "level": "set", "scores": {}})
    entry["scores"][tool] = result["score"]


def _collect_hypothesis_metrics(metrics: dict, tool: str, hypotheses: list[dict]) -> None:
    for name, bucket in hypothesis_metric_summary(hypotheses).items():
        entry = metrics.setdefault(
            name, {"axis": bucket["axis"], "level": "hypothesis",
                  "scores": {}, "n_scored": {}, "n_total": {}}
        )
        entry["scores"][tool] = bucket["mean_score"]
        entry["n_scored"][tool] = bucket["n_scored"]
        entry["n_total"][tool] = bucket["n_total"]


def compare(results: list[dict]) -> dict:
    """Build a metric x tool comparison from a list of ``report.results.to_dict()``
    (equivalently, ``benchmarking run --out ...``) shaped dicts, one per tool."""
    tools = [r["tool"] for r in results]
    if len(set(tools)) != len(tools):
        raise ValueError(f"duplicate tool name in comparison input: {tools}")

    axis_names = {axis for r in results for axis in r["axis_scores"]}
    axis_scores = {
        axis: {r["tool"]: r["axis_scores"].get(axis) for r in results} for axis in axis_names
    }

    configs = {r["tool"]: r.get("config") for r in results}
    distinct_configs = {json.dumps(c, sort_keys=True) for c in configs.values()}
    config_mismatch = len(distinct_configs) > 1

    metrics: dict[str, dict] = {}
    for r in results:
        for sr in (*r["set_results"], *r["novelty_profile"]):
            _collect_set_metric(metrics, r["tool"], sr)
        _collect_hypothesis_metrics(metrics, r["tool"], r["hypotheses"])
    # A metric only produced for some tools (e.g. one run had no judge
    # attached) still gets an entry for every tool, valued None.
    for entry in metrics.values():
        for tool in tools:
            entry["scores"].setdefault(tool, None)

    return {
        "tools": tools,
        "composite": {r["tool"]: r["composite"] for r in results},
        "composite_caveat": COMPOSITE_CAVEAT,
        "config_mismatch": config_mismatch,
        "config_mismatch_warning": (
            "Compared tools were scored under different RunConfig settings -- "
            "composite/axis scores below are not directly comparable. See `configs`."
            if config_mismatch else None
        ),
        "configs": configs,
        "disqualified": {r["tool"]: r["disqualified"] for r in results},
        "disqualified_reason": {r["tool"]: r["disqualified_reason"] for r in results},
        "axis_scores": axis_scores,
        "metrics": metrics,
    }


def load_and_compare(paths: list[str | Path]) -> dict:
    results = [json.loads(Path(p).read_text()) for p in paths]
    return compare(results)


def write_json(comparison: dict, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(comparison, indent=2))
