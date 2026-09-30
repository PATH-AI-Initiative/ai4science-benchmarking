"""Tests for report.compare: merging multiple tools' to_dict()-shaped results
into one metric x tool comparison.
"""

from __future__ import annotations

import json

from benchmarking_pipeline.report.compare import compare, load_and_compare, write_json
from benchmarking_pipeline.report.results import COMPOSITE_CAVEAT


def _result(tool, composite=None, disqualified=False, disqualified_reason=None,
            axis_scores=None, set_results=None, novelty_profile=None, hypotheses=None,
            config=None):
    return {
        "tool": tool,
        "disqualified": disqualified,
        "disqualified_reason": disqualified_reason,
        "composite": composite,
        "config": config,
        "axis_scores": axis_scores or {},
        "hypotheses": hypotheses or [],
        "set_results": set_results or [],
        "novelty_profile": novelty_profile or [],
    }


def _metric_result(metric, axis, score):
    return {"metric": metric, "axis": axis, "score": score, "anchor": None, "evidence": {}, "error": None}


def _hyp(hyp_id, results):
    return {"id": hyp_id, "rank": 1, "axis_scores": {}, "results": results}


def test_composite_and_disqualified_merge():
    a = _result("A", composite=0.7)
    b = _result("B", composite=None, disqualified=True,
                disqualified_reason="failed safety gate (safety=0.20)")

    out = compare([a, b])

    assert out["tools"] == ["A", "B"]
    assert out["composite"] == {"A": 0.7, "B": None}
    assert out["disqualified"] == {"A": False, "B": True}
    assert out["disqualified_reason"] == {"A": None, "B": "failed safety gate (safety=0.20)"}


def test_axis_scores_backfill_missing_tool_with_none():
    a = _result("A", axis_scores={"accuracy": 0.8, "quality": 0.6})
    b = _result("B", axis_scores={"accuracy": 0.5})  # no quality score for B

    out = compare([a, b])

    assert out["axis_scores"]["accuracy"] == {"A": 0.8, "B": 0.5}
    assert out["axis_scores"]["quality"] == {"A": 0.6, "B": None}


def test_set_level_metric_collected_directly():
    a = _result("A", set_results=[_metric_result("robustness", "quality", 0.9)])
    b = _result("B", set_results=[_metric_result("robustness", "quality", 0.4)])

    out = compare([a, b])

    assert out["metrics"]["robustness"]["level"] == "set"
    assert out["metrics"]["robustness"]["axis"] == "quality"
    assert out["metrics"]["robustness"]["scores"] == {"A": 0.9, "B": 0.4}


def test_novelty_profile_metric_collected_alongside_set_results():
    a = _result("A", novelty_profile=[_metric_result("diversity", "novelty", 0.6)])
    b = _result("B", novelty_profile=[])

    out = compare([a, b])

    assert out["metrics"]["diversity"]["scores"] == {"A": 0.6, "B": None}


def test_hypothesis_level_metric_is_unweighted_mean_across_hypotheses():
    hyps = [
        _hyp("h1", [_metric_result("citation_accuracy", "accuracy", 1.0)]),
        _hyp("h2", [_metric_result("citation_accuracy", "accuracy", 0.5)]),
    ]
    a = _result("A", hypotheses=hyps)

    out = compare([a])

    entry = out["metrics"]["citation_accuracy"]
    assert entry["level"] == "hypothesis"
    assert entry["scores"]["A"] == 0.75
    assert entry["n_scored"]["A"] == 2
    assert entry["n_total"]["A"] == 2


def test_hypothesis_level_metric_excludes_null_scores_from_mean_but_counts_total():
    hyps = [
        _hyp("h1", [_metric_result("logical_consistency", "accuracy", 0.8)]),
        _hyp("h2", [_metric_result("logical_consistency", "accuracy", None)]),  # not assessed
    ]
    a = _result("A", hypotheses=hyps)

    out = compare([a])

    entry = out["metrics"]["logical_consistency"]
    assert entry["scores"]["A"] == 0.8  # mean over the one assessed hypothesis
    assert entry["n_scored"]["A"] == 1
    assert entry["n_total"]["A"] == 2


def test_metric_missing_for_one_tool_backfills_none():
    a = _result("A", set_results=[_metric_result("adversarial", "quality", 0.9)])
    b = _result("B")  # no adversarial_traps configured for this run

    out = compare([a, b])

    assert out["metrics"]["adversarial"]["scores"] == {"A": 0.9, "B": None}


def test_duplicate_tool_name_rejected():
    a = _result("A", composite=0.5)
    a_again = _result("A", composite=0.9)
    try:
        compare([a, a_again])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_composite_caveat_present_in_comparison_output():
    a = _result("A", composite=0.7)
    out = compare([a])
    assert out["composite_caveat"] == COMPOSITE_CAVEAT


def test_config_mismatch_detected_and_flagged():
    a = _result("A", composite=0.7, config={"safety_gate": 0.5})
    b = _result("B", composite=0.6, config={"safety_gate": 0.3})

    out = compare([a, b])

    assert out["config_mismatch"] is True
    assert "not directly comparable" in out["config_mismatch_warning"]
    assert out["configs"] == {"A": {"safety_gate": 0.5}, "B": {"safety_gate": 0.3}}


def test_same_config_across_tools_not_flagged():
    a = _result("A", composite=0.7, config={"safety_gate": 0.5})
    b = _result("B", composite=0.6, config={"safety_gate": 0.5})

    out = compare([a, b])

    assert out["config_mismatch"] is False
    assert out["config_mismatch_warning"] is None


def test_write_json_and_load_and_compare_round_trip(tmp_path):
    a = _result("A", composite=0.7)
    b = _result("B", composite=0.3)
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    path_a.write_text(json.dumps(a))
    path_b.write_text(json.dumps(b))

    comparison = load_and_compare([path_a, path_b])
    assert comparison["composite"] == {"A": 0.7, "B": 0.3}

    out_path = tmp_path / "comparison.json"
    write_json(comparison, out_path)
    assert json.loads(out_path.read_text())["tools"] == ["A", "B"]
