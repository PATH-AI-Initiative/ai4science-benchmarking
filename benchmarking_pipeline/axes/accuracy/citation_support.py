"""Citation support: does the resolved paper's abstract actually support the
specific claim it's attached to?

A sibling of ``citation_accuracy``, not a layer within it: a citation can be
perfectly real and correctly identified while still saying something
unrelated or contradictory to the claim it's cited against -- a different
failure from the source not existing. Only checked for citations that already
passed ``citation_accuracy``'s existence/title-match gate (via the shared
:func:`~._citation_resolution.resolve_citation`, so no repeated API/judge
calls) and that have an abstract available.

For each checkable citation, a judge compares the source's abstract against
the claim text and classifies the relationship as
``supports``/``contradicts``/``neutral`` (defaulting to ``neutral`` when
ambiguous), via the same
:func:`~benchmarking_pipeline.services.relationship_judge.judge_relationship`
utility ``logical_consistency`` uses. ``score`` is the proportion classified
``supports`` -- ``contradicts`` and ``neutral`` both count against it.

Degrades gracefully: no literature client, no judge, or no citations that are
both existing and abstract-bearing -> ``score`` is ``None``.
"""

from __future__ import annotations

from ...core.context import Context
from ...core.metric import Axis, HypothesisMetric, MetricResult
from ...core.models import EvaluationRun, Hypothesis
from ...core.registry import register
from ...services.relationship_judge import judge_relationship
from ._citation_resolution import resolve_citation

_SUPPORT_CHOICES = ["supports", "contradicts", "neutral"]

_SUPPORT_INSTRUCTIONS = """\
You are checking whether a cited source actually supports a specific claim from \
a scientific hypothesis. Choose exactly one:
- supports: the source confirms or directly implies the claim.
- contradicts: the source states something that conflicts with the claim.
- neutral: the source doesn't clearly confirm or deny the claim (e.g. it's about \
a related but different topic, or is too vague to tell).
Default to neutral unless the relationship is clear.
"""


def _judge_support(judge, source_abstract: str, claim_text: str) -> str:
    return judge_relationship(
        judge,
        label_a="Source abstract", text_a=source_abstract,
        label_b="Claim", text_b=claim_text,
        instructions=_SUPPORT_INSTRUCTIONS,
        choices=_SUPPORT_CHOICES,
    )


@register
class CitationSupport(HypothesisMetric):
    name = "citation_support"
    axis = Axis.ACCURACY

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
        if ctx.judge is None:
            return MetricResult(
                metric=self.name, axis=self.axis, score=None,
                evidence={"status": "no_judge"},
            )

        checked = []
        for claim, ref in claim_refs:
            resolved = resolve_citation(ref, ctx)
            support = None
            if resolved.exists and resolved.abstract:
                support = _judge_support(ctx.judge, resolved.abstract, claim.text)
            checked.append({"reference": ref.raw, "exists": resolved.exists, "support": support})

        verdicts = [c["support"] for c in checked if c["support"] is not None]
        if not verdicts:
            return MetricResult(
                metric=self.name, axis=self.axis, score=None,
                evidence={"status": "no_checkable_citations", "checked": checked},
            )

        score = round(verdicts.count("supports") / len(verdicts), 4)

        return MetricResult(
            metric=self.name, axis=self.axis, score=score,
            anchor=_anchor(score),
            evidence={"checked": checked, "n_checked": len(verdicts), "n_total": len(claim_refs)},
        )


def _anchor(score: float) -> str:
    if score >= 0.8:
        return "citations strongly support their claims"
    if score >= 0.5:
        return "citations partially support their claims"
    return "citations largely fail to support their claims"
