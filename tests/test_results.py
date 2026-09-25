"""Tests for report.results: ToolScore -> dict serialization, including the
composite-caveat and config passthrough this adds.
"""

from __future__ import annotations

from benchmarking_pipeline.core.config import RunConfig
from benchmarking_pipeline.core.metric import Axis, MetricResult
from benchmarking_pipeline.core.scoring import HypothesisScore, aggregate_tool_score
from benchmarking_pipeline.report.results import COMPOSITE_CAVEAT, to_dict


def test_to_dict_carries_composite_caveat_and_config():
    config = RunConfig(accuracy_floor=0.4, top_k_hypotheses=2)
    score = aggregate_tool_score("t", [], [], [], config)

    d = to_dict(score)

    assert d["composite_caveat"] == COMPOSITE_CAVEAT
    assert d["config"]["accuracy_floor"] == 0.4
    assert d["config"]["top_k_hypotheses"] == 2
    assert d["config"]["axis_weights"] == {"accuracy": 0.5, "quality": 0.5}


def test_axis_weight_keys_are_plain_strings_not_enum_members():
    score = aggregate_tool_score("t", [], [], [], RunConfig())
    d = to_dict(score)
    assert set(d["config"]["axis_weights"]) == {"accuracy", "quality"}
    for key in d["config"]["axis_weights"]:
        assert type(key) is str


def test_hypothesis_metric_coverage_surfaces_a_thin_sample():
    """Regression test for a real case: citation_accuracy scored only 1 of 3
    hypotheses (the other 2 had no claim-level references at all) and still
    came back a perfect 1.0 -- that thin sample is easy to miss buried in the
    per-hypothesis results list, so it must be visible at the top level too."""
    hyp_scores = [
        HypothesisScore(
            hypothesis_id="h1", rank=1, passed_accuracy_floor=True,
            results=[MetricResult(metric="citation_accuracy", axis=Axis.ACCURACY, score=1.0)],
        ),
        HypothesisScore(
            hypothesis_id="h2", rank=2, passed_accuracy_floor=True,
            results=[MetricResult(metric="citation_accuracy", axis=Axis.ACCURACY, score=None,
                                  evidence={"status": "no_references"})],
        ),
        HypothesisScore(
            hypothesis_id="h3", rank=3, passed_accuracy_floor=True,
            results=[MetricResult(metric="citation_accuracy", axis=Axis.ACCURACY, score=None,
                                  evidence={"status": "no_references"})],
        ),
    ]
    score = aggregate_tool_score("t", hyp_scores, [], [], RunConfig())

    coverage = to_dict(score)["hypothesis_metric_coverage"]["citation_accuracy"]

    assert coverage == {"axis": "accuracy", "mean_score": 1.0, "n_scored": 1, "n_total": 3}
