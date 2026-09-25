"""Output reproducibility — a property of repeated identical runs.

Runs the identical prompt N times and measures the mean and variance of
pairwise cosine similarity between the runs' comparison vectors. High mean
similarity with low variance = reliable; low mean or high variance separates
instability from productive stochasticity (genuine exploration).

What gets compared per run is ``config.multi_run_comparison`` (see
``RunConfig``): the whole hypothesis set mean-pooled into one vector, or just
the rank-1 hypothesis. Not yet settled which is the better default — both are
available.

In ``top_hypothesis`` mode specifically, embedding similarity turns out not to
discriminate much at all: measured directly, two totally unrelated research
questions embed around 0.83 cosine similarity, while two runs proposing
*completely different drug classes* for the identical question still embed
0.91-0.96 -- specter2's scale is compressed enough that "same narrow research
question" alone accounts for nearly all of that, regardless of which specific
intervention each run actually proposes. A fixed threshold like "0.85 =
highly reproducible" is barely more discriminating than chance here.

So in ``top_hypothesis`` mode, when a judge is available, the *scored* signal
is ``mechanism_match_rate`` -- not the embedding similarity. It isn't a plain
match-rate on rank-1-vs-rank-1 either: a run's top idea reappearing at rank 2
in another run is a different (better) outcome than it vanishing outright,
and a strict rank-1-only comparison can't tell those apart. Each pair of runs
is checked via :func:`_shared.mechanism_match_credit` -- does run A's rank-1
idea survive anywhere in run B's top-``config.top_k_hypotheses``, and
vice versa, credited by how far down it dropped (rank 1 = full credit, rank 2
= half, ...). Embedding similarity is still computed and reported in full
(``mean_pairwise_similarity``, ``variance``) since it's informative context,
just not treated as if it discriminates reproducibility on its own. Falls
back to the embedding-based score when there's no judge, or in ``whole_set``
mode (there's no single top-hypothesis pair to check mechanism agreement on).

Implemented as a :class:`MultiRunMetric`: it reads ``bundle.base`` + ``bundle.repeats``.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from ...core.context import Context
from ...core.metric import Axis, MetricResult, MultiRunMetric
from ...core.models import RunBundle
from ...core.registry import register
from ...services.embeddings import pairwise_cosine
from ._shared import mechanism_anchor, mechanism_match_credit, run_vector, top_k_hypotheses


def _mechanism_match_rate(runs: list, ctx: Context) -> tuple[float | None, list[dict] | None]:
    if ctx.config.multi_run_comparison != "top_hypothesis" or ctx.judge is None:
        return None, None
    k = ctx.config.top_k_hypotheses
    top_lists = [top_k_hypotheses(r.outputs, k) for r in runs]
    if any(not tl for tl in top_lists):
        return None, None
    pairs = []
    for i, j in combinations(range(len(runs)), 2):
        credit, detail = mechanism_match_credit(ctx.judge, top_lists[i], top_lists[j])
        pairs.append({"i": i, "j": j, "credit": credit, **detail})
    rate = round(sum(p["credit"] for p in pairs) / len(pairs), 4)
    return rate, pairs


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
        mechanism_match_rate, mechanism_matches = _mechanism_match_rate(runs, ctx)

        if mechanism_match_rate is not None:
            score_value = mechanism_match_rate
            anchor = mechanism_anchor(mechanism_match_rate, mean_sim)
            score_basis = "mechanism_match_rate"
        else:
            score_value = round(max(0.0, min(1.0, mean_sim)), 4)
            anchor = _anchor(mean_sim, variance)
            score_basis = "mean_pairwise_similarity"

        return MetricResult(
            metric=self.name, axis=self.axis,
            score=score_value,
            anchor=anchor,
            evidence={
                "score_basis": score_basis,
                "n_runs": len(runs),
                "comparison_unit": comparison,
                "mean_pairwise_similarity": round(mean_sim, 4),
                "variance": round(variance, 6),
                "embedding_model": ctx.embeddings.name,
                "mechanism_match_rate": mechanism_match_rate,
                "mechanism_matches": mechanism_matches,
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
