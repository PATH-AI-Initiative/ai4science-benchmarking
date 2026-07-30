"""Smoke test: the full pipeline runs end to end and produces a score."""

from __future__ import annotations

from benchmarking_pipeline.core.config import RunConfig
from benchmarking_pipeline.core.context import Context
from benchmarking_pipeline.core.metric import Axis, MetricResult
from benchmarking_pipeline.core.models import (
    Claim,
    EvaluationRun,
    Hypothesis,
    HypothesisSet,
    Reference,
    RunBundle,
    Tool,
)
from benchmarking_pipeline.core.scoring import HypothesisScore, aggregate_tool_score
from benchmarking_pipeline.pipeline import evaluate, evaluate_bundle
from benchmarking_pipeline.services.embeddings import HashingEmbedding


def _sample_run() -> EvaluationRun:
    hyps = [
        Hypothesis(
            id="h1", text="PfKelch13 mutations drive artemisinin resistance in P. knowlesi.",
            rank=1,
            claims=[Claim(text="Kelch13 is implicated in resistance.",
                          references=[Reference(raw="Ariey et al. 2014")])],
        ),
        Hypothesis(
            id="h2", text="Altered lipid metabolism modulates parasite drug tolerance.",
            rank=2,
            claims=[Claim(text="Lipid pathways affect tolerance.")],
        ),
    ]
    return EvaluationRun(
        tool=Tool(name="TestTool", version="v0"),
        prompt="artemisinin-resistant Plasmodium knowlesi",
        outputs=HypothesisSet(hypotheses=hyps),
    )


def test_pipeline_runs_and_reports_diversity():
    ctx = Context(config=RunConfig(), embeddings=HashingEmbedding())
    score = evaluate(_sample_run(), ctx)

    assert not score.disqualified
    assert len(score.hypothesis_scores) == 2

    # Diversity is a real computation and must appear in the novelty profile.
    diversity = [r for r in score.novelty_profile if r.metric == "diversity"]
    assert diversity and diversity[0].score is not None
    assert all(r.axis is Axis.NOVELTY for r in score.novelty_profile)


def test_details_checklist_is_captured():
    ctx = Context(config=RunConfig(), embeddings=HashingEmbedding())
    score = evaluate(_sample_run(), ctx)
    details = [r for r in score.set_results if r.metric == "details_checklist"]
    # Tier 1 is descriptive: present, score is None, carries a profile.
    assert details and details[0].score is None
    assert details[0].evidence["profile"]["identity"]["name"] == "TestTool"


def test_reproducibility_needs_multiple_runs():
    ctx = Context(config=RunConfig(), embeddings=HashingEmbedding())

    # Single run: reproducibility can't be computed.
    single = evaluate(_sample_run(), ctx)
    repro_single = [r for r in single.set_results if r.metric == "reproducibility"]
    assert repro_single == []  # multi-run metrics don't run in evaluate()

    # Bundle of identical repeats: reproducibility is real and near-perfect.
    base = _sample_run()
    bundle = RunBundle(base=base, repeats=[_sample_run(), _sample_run()])
    scored = evaluate_bundle(bundle, ctx)
    repro = [r for r in scored.set_results if r.metric == "reproducibility"]
    assert repro and repro[0].score is not None
    assert repro[0].score > 0.99  # identical outputs => ~1.0 similarity


def test_axis_fold_is_order_independent_weighted_mean():
    """Regression test: set-level metrics used to be folded one at a time via
    pairwise averaging, so the composite depended on metric registration order.
    Two set-level Quality metrics with the same weight should average evenly
    regardless of which order they appear in ``set_results``.
    """
    config = RunConfig()
    hyp_scores = [
        HypothesisScore(
            hypothesis_id="h1", rank=1, results=[], passed_accuracy_floor=True,
            axis_scores={Axis.QUALITY: 0.8},
        )
    ]
    reproducibility = MetricResult(metric="reproducibility", axis=Axis.QUALITY, score=0.9)
    robustness = MetricResult(metric="robustness", axis=Axis.QUALITY, score=0.5)

    forward = aggregate_tool_score("t", hyp_scores, [reproducibility, robustness], [], config)
    reversed_ = aggregate_tool_score("t", hyp_scores, [robustness, reproducibility], [], config)

    assert forward.axis_scores[Axis.QUALITY] == reversed_.axis_scores[Axis.QUALITY]
    # Equal-weight mean of 0.8 (hypotheses), 0.9, 0.5 = 0.7333...
    assert abs(forward.axis_scores[Axis.QUALITY] - (0.8 + 0.9 + 0.5) / 3) < 1e-9
