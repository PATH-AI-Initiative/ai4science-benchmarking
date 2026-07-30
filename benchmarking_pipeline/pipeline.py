"""Evaluation orchestrator.

Ties the pieces together, mirroring Figure 1 of the framework:
    prompt -> tool adapter -> HypothesisSet -> [Tier 1..4 metrics] -> ToolScore

Two entry points:

* :func:`evaluate` scores a single :class:`EvaluationRun`. Multi-run metrics
  (reproducibility, robustness) report ``None`` here — they need more than one run.
* :func:`evaluate_bundle` scores a :class:`RunBundle` (a base run plus repeated
  and/or perturbed runs), so the multi-run metrics produce real numbers. This is
  the "collect N runs and feed them in" layer; single-run metrics still run
  against the bundle's base run.
"""

from __future__ import annotations

# Importing the axes package registers every metric.
from . import axes  # noqa: F401
from .core.context import Context
from .core.metric import Axis, HypothesisMetric, MultiRunMetric, SetMetric
from .core.models import EvaluationRun, RunBundle
from .core.registry import all_metrics
from .core.scoring import ToolScore, aggregate_tool_score, score_hypothesis


def _score_hypotheses(run: EvaluationRun, ctx: Context) -> list:
    hyp_metrics = [m for m in all_metrics() if isinstance(m, HypothesisMetric)]
    scores = []
    for hyp in run.outputs:
        results = [m.score(hyp, run, ctx) for m in hyp_metrics]
        scores.append(score_hypothesis(hyp.id, hyp.rank, results, ctx.config))
    return scores


def _score_sets(run: EvaluationRun, ctx: Context) -> list:
    set_metrics = [m for m in all_metrics() if isinstance(m, SetMetric)]
    return [m.score(run.outputs, run, ctx) for m in set_metrics]


def _aggregate(run: EvaluationRun, hypothesis_scores, set_results, ctx: Context) -> ToolScore:
    novelty_profile = [r for r in set_results if r.axis is Axis.NOVELTY]
    scored_set_results = [r for r in set_results if r.axis is not Axis.NOVELTY]
    return aggregate_tool_score(
        tool=run.tool.name,
        hypothesis_scores=hypothesis_scores,
        set_results=scored_set_results,
        novelty_profile=novelty_profile,
        config=ctx.config,
    )


def evaluate(run: EvaluationRun, ctx: Context) -> ToolScore:
    """Evaluate a single run. Multi-run metrics are skipped."""
    hypothesis_scores = _score_hypotheses(run, ctx)
    set_results = _score_sets(run, ctx)
    return _aggregate(run, hypothesis_scores, set_results, ctx)


def evaluate_bundle(bundle: RunBundle, ctx: Context) -> ToolScore:
    """Evaluate a bundle of runs. Single-run metrics score ``bundle.base``;
    multi-run metrics (reproducibility, robustness) score the whole bundle.
    """
    base = bundle.base
    hypothesis_scores = _score_hypotheses(base, ctx)
    set_results = _score_sets(base, ctx)

    multi_run = [m for m in all_metrics() if isinstance(m, MultiRunMetric)]
    set_results += [m.score(bundle, ctx) for m in multi_run]

    return _aggregate(base, hypothesis_scores, set_results, ctx)
