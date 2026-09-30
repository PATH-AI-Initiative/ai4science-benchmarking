"""Diversity of ideas: how spread are the hypotheses across conceptual space?

Computes average pairwise cosine distance between output embeddings. A
cluster of similar hypotheses is effectively one idea repeated; a spread,
non-redundant set is worth more.

Reported as a novelty-profile figure, not a composite contributor.
"""

from __future__ import annotations

import numpy as np

from ...core.context import Context
from ...core.metric import Axis, MetricResult, SetMetric
from ...core.models import EvaluationRun, HypothesisSet
from ...core.registry import register
from ...services.embeddings import pairwise_cosine


@register
class Diversity(SetMetric):
    name = "diversity"
    axis = Axis.NOVELTY

    def score(self, outputs: HypothesisSet, run: EvaluationRun, ctx: Context) -> MetricResult:
        if ctx.embeddings is None:
            return MetricResult(
                metric=self.name, axis=self.axis, score=None,
                evidence={"status": "no_embedding_model"},
            )
        texts = [h.text for h in outputs]
        if len(texts) < 2:
            return MetricResult(
                metric=self.name, axis=self.axis, score=None,
                evidence={"status": "need_at_least_two_hypotheses", "n": len(texts)},
            )

        vectors = ctx.embeddings.embed(texts)
        sim = pairwise_cosine(vectors)
        # Mean of the strict upper triangle = mean pairwise similarity.
        iu = np.triu_indices(len(texts), k=1)
        mean_similarity = float(sim[iu].mean())
        mean_distance = 1.0 - mean_similarity  # higher = more diverse

        return MetricResult(
            metric=self.name, axis=self.axis,
            score=round(max(0.0, min(1.0, mean_distance)), 4),
            anchor=_anchor(mean_distance),
            evidence={
                "mean_pairwise_similarity": round(mean_similarity, 4),
                "mean_pairwise_distance": round(mean_distance, 4),
                "n_hypotheses": len(texts),
                "embedding_model": ctx.embeddings.name,
            },
        )


def _anchor(distance: float) -> str:
    if distance >= 0.6:
        return "broadly spread across conceptual space"
    if distance >= 0.3:
        return "moderately diverse"
    return "clustered / redundant"
