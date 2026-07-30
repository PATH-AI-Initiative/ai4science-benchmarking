"""Citation accuracy: do the sources a hypothesis cites actually exist, and do
they support the claims they're attached to?

Three layered checks:

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
3. **Support** (diagnostic, needs both a literature client that returns
   abstracts and a judge): does the source's abstract actually support this
   specific claim, via the same ``judge_relationship`` utility
   ``logical_consistency`` uses — or does it merely exist while saying
   something unrelated or contradictory? Reported as ``support_rate`` and
   per-reference verdicts in evidence, never folded into ``score``. A citation
   can be perfectly real and still misused, and conflating "exists" with
   "supports" into one number would make the score's meaning depend on which
   optional services happen to be attached for a given run.

Degrades gracefully at each layer: no literature client -> ``score`` is
``None``; no judge -> the title-match check is skipped (a fuzzy search hit is
trusted at face value, the old behavior) and ``support_rate`` is ``None``; no
abstract available for a given reference -> that reference's support stays
unchecked, but the existence score still computes.
"""

from __future__ import annotations

from ...core.context import Context
from ...core.metric import Axis, HypothesisMetric, MetricResult
from ...core.models import EvaluationRun, Hypothesis
from ...core.registry import register
from ...services.relationship_judge import judge_relationship

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

# "different_paper" listed last: judge_relationship's fallback-on-unparseable-
# verdict, and the conservative choice when a fuzzy bibliographic-search hit's
# actual relevance is ambiguous -- a citation isn't "verified" on a shrug.
_TITLE_MATCH_CHOICES = ["same_paper", "different_paper"]

_TITLE_MATCH_INSTRUCTIONS = """\
A bibliographic search for a cited reference returned this candidate. Search \
relevance ranking is based on shared wording, not topic -- a generic phrase \
like "phase 3 trial" or "meets primary endpoint" can make an unrelated paper \
(different drug, different disease) score as a strong textual match. Choose \
exactly one:
- same_paper: the candidate is clearly about the same subject as the cited \
reference (same drug/gene/finding), consistent with actually being that paper.
- different_paper: the candidate is about a different subject despite \
wording overlap, or there isn't enough shared subject matter to tell.
Default to different_paper unless clearly the same paper.
"""


def _judge_support(judge, source_abstract: str, claim_text: str) -> str:
    return judge_relationship(
        judge,
        label_a="Source abstract", text_a=source_abstract,
        label_b="Claim", text_b=claim_text,
        instructions=_SUPPORT_INSTRUCTIONS,
        choices=_SUPPORT_CHOICES,
    )


def _judge_title_match(judge, cited_reference: str, candidate_title: str) -> str:
    return judge_relationship(
        judge,
        label_a="Cited reference", text_a=cited_reference,
        label_b="Search result title", text_b=candidate_title,
        instructions=_TITLE_MATCH_INSTRUCTIONS,
        choices=_TITLE_MATCH_CHOICES,
    )


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
        support_verdicts = []
        for claim, ref in claim_refs:
            record = None
            is_exact_doi = False
            if ref.doi:
                record = ctx.literature.lookup_doi(ref.doi)
                is_exact_doi = record is not None
            if record is None:
                hits = ctx.literature.search(ref.title or ref.raw, limit=1)
                record = hits[0] if hits else None
            matched = record is not None and record.match_score >= 0.7

            title_match = None
            if matched and not is_exact_doi and ctx.judge is not None and record.title:
                title_match = _judge_title_match(ctx.judge, ref.title or ref.raw, record.title)
            exists = matched and (title_match is None or title_match == "same_paper")

            support = None
            if exists and ctx.judge is not None and record.abstract:
                support = _judge_support(ctx.judge, record.abstract, claim.text)
                support_verdicts.append(support)

            checked.append({
                "reference": ref.raw, "exists": exists,
                "matched_title": record.title if record else None,
                "title_match": title_match,
                "support": support,
            })

        n_exist = sum(1 for c in checked if c["exists"])
        score = n_exist / len(checked)
        support_rate = (
            round(support_verdicts.count("supports") / len(support_verdicts), 4)
            if support_verdicts else None
        )

        return MetricResult(
            metric=self.name, axis=self.axis, score=score,
            anchor=_anchor(score),
            evidence={
                "checked": checked, "n_exist": n_exist, "n_total": len(checked),
                "support_rate": support_rate, "n_support_checked": len(support_verdicts),
            },
        )


def _anchor(score: float) -> str:
    if score >= 0.95:
        return "all citations verifiable"
    if score >= 0.7:
        return "most citations verifiable"
    if score >= 0.4:
        return "many citations unverifiable"
    return "citations largely unverifiable"
