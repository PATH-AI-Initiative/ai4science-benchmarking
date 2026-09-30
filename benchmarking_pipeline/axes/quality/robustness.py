"""Robustness to perturbation — a property of base vs perturbed runs.

Compares the base run's comparison vector against each perturbation run's,
grouped by perturbation type (tagged in ``run.metadata["perturbation"]``):

* ``reword`` — a paraphrased prompt should leave outputs stable (high
  similarity = genuine reasoning; low = surface pattern-matching). This is
  the scored signal.
* ``kb_removal`` — removing a paper a hypothesis depends on should change the
  outputs (low similarity is healthy). Reported for interpretation, not
  scored; not currently exercised since no evaluated tool exposes a way to
  tweak its knowledge base. Kept because the scoring code is already
  perturbation-type-agnostic.

What gets compared per run is ``config.multi_run_comparison`` (see
``RunConfig``): the whole hypothesis set mean-pooled, or just the rank-1
hypothesis. Default is unsettled -- both are available.

In ``top_hypothesis`` mode, embedding similarity barely discriminates (see
``reproducibility.py``: two unrelated research questions embed ~0.83; two
runs proposing different drugs for the identical question embed 0.91-0.96).
So when a judge is available, the scored signal is
``reword_mechanism_match_rate``, not embedding similarity -- and not a plain
rank-1-vs-rank-1 match, since a top idea dropping to rank 2 is a different
outcome than vanishing. Each perturbation run is checked via
:func:`_shared.mechanism_match_credit` against base's top-k, credited by how
far the matching idea dropped. Embedding similarity is still computed and
reported, just not trusted as the discriminating signal. Falls back to it
when there's no judge, or in ``whole_set`` mode.

Implemented as a :class:`MultiRunMetric`: reads ``bundle.base`` + ``bundle.perturbations``.
"""

from __future__ import annotations

from ...core.context import Context
from ...core.metric import Axis, MetricResult, MultiRunMetric
from ...core.models import RunBundle
from ...core.registry import register
from ...services.embeddings import cosine_similarity
from ._shared import mechanism_anchor, mechanism_match_credit, run_vector, top_k_hypotheses


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

        check_mechanism = comparison == "top_hypothesis" and ctx.judge is not None
        top_k = ctx.config.top_k_hypotheses
        base_top_k = top_k_hypotheses(bundle.base.outputs, top_k) if check_mechanism else []

        by_type: dict[str, list[float]] = {}
        mechanism_by_type: dict[str, list[dict]] = {}
        for run in bundle.perturbations:
            vec = run_vector(run, ctx)
            if vec is None:
                continue
            ptype = run.metadata.get("perturbation", "reword")
            by_type.setdefault(ptype, []).append(cosine_similarity(base_vec, vec))

            if check_mechanism and base_top_k:
                other_top_k = top_k_hypotheses(run.outputs, top_k)
                if other_top_k:
                    credit, detail = mechanism_match_credit(ctx.judge, base_top_k, other_top_k)
                    mechanism_by_type.setdefault(ptype, []).append({"credit": credit, **detail})

        # Embedding-based stability under rewording -- always computed and
        # reported, but see the module docstring for why it isn't trusted as
        # the scored signal on its own when mechanism-match is available.
        reword_sims = by_type.get("reword", [])
        reword_stability = round(sum(reword_sims) / len(reword_sims), 4) if reword_sims else None

        reword_mechanisms = mechanism_by_type.get("reword", [])
        reword_mechanism_match_rate = (
            round(sum(m["credit"] for m in reword_mechanisms) / len(reword_mechanisms), 4)
            if reword_mechanisms else None
        )

        if reword_mechanism_match_rate is not None:
            score = reword_mechanism_match_rate
            anchor = mechanism_anchor(
                reword_mechanism_match_rate, reword_stability,
                labels=("stable under rewording", "partially stable",
                        "unstable under rewording"),
            )
            score_basis = "reword_mechanism_match_rate"
        else:
            score = reword_stability
            anchor = _anchor(score) if score is not None else None
            score_basis = "reword_stability" if score is not None else None

        return MetricResult(
            metric=self.name, axis=self.axis, score=score,
            anchor=anchor,
            evidence={
                "score_basis": score_basis,
                "comparison_unit": comparison,
                "reword_stability": reword_stability,
                "similarity_by_perturbation": {
                    k: [round(s, 4) for s in v] for k, v in by_type.items()
                },
                "reword_mechanism_match_rate": reword_mechanism_match_rate,
                "mechanism_match_by_perturbation": mechanism_by_type or None,
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
