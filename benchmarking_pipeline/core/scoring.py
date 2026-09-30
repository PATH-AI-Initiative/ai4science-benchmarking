"""Scoring and aggregation.

Turns per-metric results into a tool-level composite, encoding the rubric
rules that would otherwise be scattered across the axes:

* **Safety gate** — a tool that fails any gate metric is disqualified outright.
* **Best-hypothesis focus** — the tool-level score is built from the top-k
  ranked hypotheses, not the average, since a resource-limited team may only
  follow up one lead.
* **Set-level metrics** (diversity, reproducibility, ...) are added on top of
  the aggregated hypothesis-level score.

Novelty is computed but never folded into the composite; the ``report`` layer
reports it as a separate profile.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import RunConfig
from .metric import Axis, MetricResult


@dataclass
class HypothesisScore:
    hypothesis_id: str
    rank: int | None
    results: list[MetricResult]
    axis_scores: dict[Axis, float] = field(default_factory=dict)


@dataclass
class ToolScore:
    tool: str
    disqualified: bool
    disqualified_reason: str | None
    composite: float | None
    axis_scores: dict[Axis, float]
    hypothesis_scores: list[HypothesisScore]
    set_results: list[MetricResult]
    novelty_profile: list[MetricResult]
    # The settings this score was actually produced under. Weights/thresholds
    # here are placeholders until derived through structured expert
    # elicitation (see RunConfig's own docstring) -- carrying the config
    # alongside the score keeps that caveat attached to the number itself,
    # and lets a comparison across tools check they were scored under the
    # same settings rather than silently comparing apples to oranges.
    config: RunConfig


def _mean(values: list[float]) -> float | None:
    scored = [v for v in values if v is not None]
    return sum(scored) / len(scored) if scored else None


def _weighted_axis_score(
    results: list[MetricResult], axis: Axis, config: RunConfig
) -> float | None:
    """Weighted mean of the scored metrics belonging to ``axis``."""
    contributions: list[tuple[float, float]] = []  # (weight, score)
    for r in results:
        if r.axis is not axis or r.score is None:
            continue
        weight = config.metric_weights.get(r.metric, 1.0)
        contributions.append((weight, r.score))
    if not contributions:
        return None
    total_weight = sum(w for w, _ in contributions)
    return sum(w * s for w, s in contributions) / total_weight


def score_hypothesis(
    hypothesis_id: str,
    rank: int | None,
    results: list[MetricResult],
    config: RunConfig,
) -> HypothesisScore:
    axis_scores: dict[Axis, float] = {}
    for axis in (Axis.ACCURACY, Axis.QUALITY):
        score = _weighted_axis_score(results, axis, config)
        if score is not None:
            axis_scores[axis] = score

    return HypothesisScore(
        hypothesis_id=hypothesis_id,
        rank=rank,
        results=results,
        axis_scores=axis_scores,
    )


def _top_k(scores: list[HypothesisScore], k: int) -> list[HypothesisScore]:
    """Select the k best-ranked hypotheses (rank 1 = top; unranked sort last)."""
    ordered = sorted(scores, key=lambda s: (s.rank is None, s.rank if s.rank is not None else 0))
    return ordered[:k]


def aggregate_tool_score(
    tool: str,
    hypothesis_scores: list[HypothesisScore],
    set_results: list[MetricResult],
    novelty_profile: list[MetricResult],
    config: RunConfig,
) -> ToolScore:
    # Safety gate: any gate metric below threshold disqualifies the tool.
    for r in set_results:
        if r.score is not None and r.axis is Axis.QUALITY and r.score < config.safety_gate:
            # Gate membership is signalled by the metric via evidence["gate"].
            if r.evidence.get("gate"):
                return ToolScore(
                    tool=tool,
                    disqualified=True,
                    disqualified_reason=f"failed safety gate ({r.metric}={r.score:.2f})",
                    composite=None,
                    axis_scores={},
                    hypothesis_scores=hypothesis_scores,
                    set_results=set_results,
                    novelty_profile=novelty_profile,
                    config=config,
                )

    top = _top_k(hypothesis_scores, config.top_k_hypotheses)

    axis_scores: dict[Axis, float] = {}
    for axis in (Axis.ACCURACY, Axis.QUALITY):
        per_hyp = [s.axis_scores.get(axis) for s in top]
        hyp_mean = _mean([v for v in per_hyp if v is not None])

        # Combine the hypothesis-level aggregate with every set-level metric for
        # this axis (diversity, reproducibility, robustness, ...) into a single
        # weighted mean. Folding them in one at a time via repeated pairwise
        # averaging would make the composite depend on metric registration
        # order — same weighting mechanism as _weighted_axis_score, so the
        # hypothesis aggregate is just one more named contribution ("_hypotheses").
        contributions = [r for r in set_results if r.axis is axis]
        if hyp_mean is not None:
            contributions = [
                MetricResult(metric="_hypotheses", axis=axis, score=hyp_mean),
                *contributions,
            ]
        axis_score = _weighted_axis_score(contributions, axis, config)
        if axis_score is not None:
            axis_scores[axis] = axis_score

    composite = None
    weighted = [
        (config.axis_weights.get(a, 0.0), s)
        for a, s in axis_scores.items()
        if config.axis_weights.get(a, 0.0) > 0
    ]
    if weighted:
        total_w = sum(w for w, _ in weighted)
        composite = sum(w * s for w, s in weighted) / total_w

    return ToolScore(
        tool=tool,
        disqualified=False,
        disqualified_reason=None,
        composite=composite,
        axis_scores=axis_scores,
        hypothesis_scores=hypothesis_scores,
        set_results=set_results,
        novelty_profile=novelty_profile,
        config=config,
    )
