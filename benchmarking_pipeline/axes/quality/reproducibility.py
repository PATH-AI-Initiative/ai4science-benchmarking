"""Output reproducibility — a property of repeated identical runs.

Runs the identical prompt N times and measures the mean and variance of
pairwise cosine similarity between the runs' comparison vectors. High mean
similarity with low variance = reliable; low mean or high variance separates
instability from productive stochasticity (genuine exploration).

What gets compared per run is ``config.multi_run_comparison`` (see
``RunConfig``): the whole hypothesis set mean-pooled into one vector, or just
the rank-1 hypothesis. Not yet settled which is the better default — both are
available.

Implemented as a :class:`MultiRunMetric`: it reads ``bundle.base`` + ``bundle.repeats``.
"""

from __future__ import annotations

import numpy as np

from ...core.context import Context
from ...core.metric import Axis, MetricResult, MultiRunMetric
from ...core.models import RunBundle
from ...core.registry import register
from ...services.embeddings import pairwise_cosine
from ._shared import run_vector


@register
class Reproducibility(MultiRunMetric):
    name = "reproducibility"
    axis = Axis.QUALITY

    def score(self, bundle: RunBundle, ctx: Context) -> MetricResult:
        if ctx.embeddings is None:
            return MetricResult(metric=self.name, axis=self.axis, score=None,
                                evidence={"status": "no_embedding_model"})
        runs = [bundle.base, *bundle.repeats]
        if len(runs) < 2:
            return MetricResult(metric=self.name, axis=self.axis, score=None,
                                evidence={"status": "need_at_least_two_runs", "n": len(runs)})

        comparison = ctx.config.multi_run_comparison
        vectors_list = [run_vector(r, ctx) for r in runs]
        if any(v is None for v in vectors_list):
            return MetricResult(
                metric=self.name, axis=self.axis, score=None,
                evidence={"status": "a_run_returned_no_hypotheses", "comparison_unit": comparison},
            )

        vectors = np.vstack(vectors_list)
        sim = pairwise_cosine(vectors)
        iu = np.triu_indices(len(runs), k=1)
        pair_sims = sim[iu]
        mean_sim = float(pair_sims.mean())
        variance = float(pair_sims.var())

        return MetricResult(
            metric=self.name, axis=self.axis,
            score=round(max(0.0, min(1.0, mean_sim)), 4),
            anchor=_anchor(mean_sim, variance),
            evidence={
                "n_runs": len(runs),
                "comparison_unit": comparison,
                "mean_pairwise_similarity": round(mean_sim, 4),
                "variance": round(variance, 6),
                "embedding_model": ctx.embeddings.name,
            },
        )


def _anchor(mean_sim: float, variance: float) -> str:
    if mean_sim >= 0.85 and variance <= 0.01:
        return "highly reproducible"
    if mean_sim >= 0.6:
        return "reproducible with some variation"
    if variance > 0.05:
        return "unstable (high variance)"
    return "low reproducibility"
