"""Citation accuracy: do the sources a hypothesis cites actually exist?

Two layered checks:

1. **Existence** (always computed, given a literature client): does a paper
   matching this reference actually exist? This is the metric's ``score`` —
   its meaning never changes regardless of what else is attached, so scores
   stay comparable across runs with different optional services wired up.
2. **Title match** (folded into existence, only when a judge is available and
   the record came from a fuzzy bibliographic search rather than an exact DOI
   lookup): bibliographic search relevance (e.g. CrossRef's own ranking score)
   is a text-overlap heuristic, not a topical one -- a press-release-style
   title like "Phase 3 trial ... meets primary endpoint" can out-score-match
   an unrelated drug's unrelated trial announcement purely on shared generic
   phrasing. An exact DOI resolution needs no such check (it's already
   authoritative); a fuzzy search hit does, via
   :func:`~benchmarking_pipeline.services.relationship_judge.judge_relationship`.

Whether an existing citation actually *supports* the claim it's attached to
is a separate, sibling metric -- see ``citation_support``. A citation can be
perfectly real and still misused, so conflating "exists" with "supports" into
one number would make this score's meaning depend on which optional services
happen to be attached for a given run. Both metrics share the same
existence/title-match resolution step (see ``_citation_resolution``) so
adding citation_support doesn't double the literature-API and judge calls
this metric already makes.

Degrades gracefully: no literature client -> ``score`` is ``None``; no judge
-> the title-match check is skipped (a fuzzy search hit is trusted at face
value, the old behavior).
"""

from __future__ import annotations

from ...core.context import Context
from ...core.metric import Axis, HypothesisMetric, MetricResult
from ...core.models import EvaluationRun, Hypothesis
from ...core.registry import register
from ._citation_resolution import resolve_citation


@register
class CitationAccuracy(HypothesisMetric):
    name = "citation_accuracy"
    axis = Axis.ACCURACY
    is_floor = True

    def score(self, hypothesis: Hypothesis, run: EvaluationRun, ctx: Context) -> MetricResult:
        claim_refs = [(claim, ref) for claim in hypothesis.claims for ref in claim.references]
        if not claim_refs:
            return MetricResult(
                metric=self.name, axis=self.axis, score=None,
                evidence={"status": "no_references"},
            )
        if ctx.literature is None:
            return MetricResult(
                metric=self.name, axis=self.axis, score=None,
                evidence={"status": "no_literature_client"},
            )

        checked = []
        for claim, ref in claim_refs:
            resolved = resolve_citation(ref, ctx)
            checked.append({
                "reference": ref.raw, "exists": resolved.exists,
                "matched_title": resolved.matched_title,
                "title_match": resolved.title_match,
            })

        n_exist = sum(1 for c in checked if c["exists"])
        score = n_exist / len(checked)

        return MetricResult(
            metric=self.name, axis=self.axis, score=score,
            anchor=_anchor(score),
            evidence={"checked": checked, "n_exist": n_exist, "n_total": len(checked)},
        )


def _anchor(score: float) -> str:
    if score >= 0.95:
        return "all citations verifiable"
    if score >= 0.7:
        return "most citations verifiable"
    if score >= 0.4:
        return "many citations unverifiable"
    return "citations largely unverifiable"
