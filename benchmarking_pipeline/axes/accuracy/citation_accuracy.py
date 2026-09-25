"""Citation accuracy: do the sources a hypothesis cites actually exist?

Two layered checks:

1. **Existence** (always computed, given a literature client): does a paper
   matching this reference exist? This is the metric's ``score`` -- its
   meaning doesn't change with what else is attached, so scores stay
   comparable across runs with different optional services wired up.
2. **Title match** (only when a judge is available, and only for a fuzzy
   bibliographic-search hit rather than an exact DOI lookup): search
   relevance ranking is a text-overlap heuristic, not a topical one -- a
   press-release-style title can out-score-match an unrelated trial
   announcement on shared generic phrasing. Checked via
   :func:`~benchmarking_pipeline.services.relationship_judge.judge_relationship`.
   An exact DOI resolution is already authoritative and skips this.

Whether an existing citation actually *supports* its claim is a separate
sibling metric -- see ``citation_support``. Conflating "exists" with
"supports" into one number would make this score's meaning depend on which
optional services happen to be attached. Both metrics share the same
existence/title-match resolution step (see ``_citation_resolution``), so
adding citation_support doesn't double the literature-API and judge calls.

Degrades gracefully: no literature client -> ``score`` is ``None``. No judge
-> title-match check is skipped, fuzzy hits trusted at face value.
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
