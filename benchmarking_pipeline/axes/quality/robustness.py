"""Robustness to perturbation — a property of base vs perturbed runs.

Compares the base run's comparison vector against each perturbation run's,
grouped by perturbation type (tagged in ``run.metadata["perturbation"]``):

* ``reword`` — a paraphrased prompt *should* leave outputs stable (high
  similarity = genuine reasoning; low = surface pattern-matching). This is the
  scored signal.

# Not currently exercised: no tool we've evaluated exposes a way to tweak its
# knowledge base, so there's nothing to point --perturb kb_removal=... at yet.
# Left here (rather than deleted) because the scoring code below is already
# perturbation-type-agnostic -- it costs nothing to keep, and this is the
# design note for whenever a tool that supports it shows up:
#
# * ``kb_removal`` — removing a paper a hypothesis depends on *should* change
#   the outputs (low similarity is the healthy response). Reported for
#   interpretation, not folded into the score, and only feasible for
#   open-knowledge-base tools.

What gets compared per run is ``config.multi_run_comparison`` (see
``RunConfig``): the whole hypothesis set mean-pooled into one vector, or just
the rank-1 hypothesis. Not yet settled which is the better default — both are
available.

Implemented as a :class:`MultiRunMetric`: it reads ``bundle.base`` + ``bundle.perturbations``.
"""

from __future__ import annotations

from ...core.context import Context
from ...core.metric import Axis, MetricResult, MultiRunMetric
from ...core.models import RunBundle
from ...core.registry import register
from ...services.embeddings import cosine_similarity
from ._shared import run_vector


@register
class Robustness(MultiRunMetric):
    name = "robustness"
    axis = Axis.QUALITY

    def score(self, bundle: RunBundle, ctx: Context) -> MetricResult:
        if ctx.embeddings is None:
            return MetricResult(metric=self.name, axis=self.axis, score=None,
                                evidence={"status": "no_embedding_model"})
        if not bundle.perturbations:
            return MetricResult(metric=self.name, axis=self.axis, score=None,
                                evidence={"status": "no_perturbation_runs"})

        comparison = ctx.config.multi_run_comparison
        base_vec = run_vector(bundle.base, ctx)
        if base_vec is None:
            return MetricResult(
                metric=self.name, axis=self.axis, score=None,
                evidence={"status": "base_run_returned_no_hypotheses", "comparison_unit": comparison},
            )

        by_type: dict[str, list[float]] = {}
        for run in bundle.perturbations:
            vec = run_vector(run, ctx)
            if vec is None:
                continue
            ptype = run.metadata.get("perturbation", "reword")
            by_type.setdefault(ptype, []).append(cosine_similarity(base_vec, vec))

        # Stability under rewording is the scored signal (higher = more robust).
        reword_sims = by_type.get("reword", [])
        score = round(sum(reword_sims) / len(reword_sims), 4) if reword_sims else None

        return MetricResult(
            metric=self.name, axis=self.axis, score=score,
            anchor=_anchor(score) if score is not None else None,
            evidence={
                "comparison_unit": comparison,
                "reword_stability": score,
                "similarity_by_perturbation": {
                    k: [round(s, 4) for s in v] for k, v in by_type.items()
                },
                "embedding_model": ctx.embeddings.name,
                "note": "kb_removal similarities reported for interpretation, not scored",
            },
        )


def _anchor(score: float) -> str:
    if score >= 0.8:
        return "stable under rewording"
    if score >= 0.5:
        return "partially stable"
    return "unstable under rewording (surface pattern-matching)"
