"""Epistemic calibration: does the tool distinguish established, contested, and
uncertain claims?

Two independent judgments per claim, then compared via an explicit scoring
table (asymmetric — overconfidence is penalised harder than underconfidence,
per the framework's own emphasis: "A poorly calibrated tool that presents
speculative claims with the same confidence as established facts is... just as
misleading as one that cites incorrectly"):

1. **Expressed confidence** — how confidently does the claim's own language
   state it (high / moderate / low)? A single-text judge classification, same
   shape as logical_consistency's role classification.
2. **Actual consensus** — searches the literature for the claim and asks the
   judge to classify its real status (established / contested / uncertain)
   given that evidence, via the shared ``judge_relationship`` utility.
   Grounded in a real search rather than the judge's own unverified background
   knowledge, consistent with the rest of this benchmark's philosophy of
   verifying rather than trusting an LLM's say-so.

Degrades gracefully: no judge -> score is None. No literature client, or no
search results for a given claim -> that claim's consensus can't be
determined, so it's excluded from the mean and reported separately, not
silently dropped and not scored via an unverified judge-only guess.
"""

from __future__ import annotations

from ...core.context import Context
from ...core.metric import Axis, HypothesisMetric, MetricResult
from ...core.models import EvaluationRun, Hypothesis
from ...core.registry import register
from ...services.relationship_judge import judge_relationship

_CONFIDENCE_CHOICES = ["high", "moderate", "low"]

_CONFIDENCE_PROMPT = """\
Classify how confidently this claim is stated, based on its language alone \
(not whether it's true) — choose exactly one:
- high: stated as settled fact, with no hedging ("shows", "demonstrates", "is").
- moderate: stated with some hedging ("likely", "suggests", "appears to").
- low: stated as speculative or tentative ("may", "could possibly", "preliminary \
evidence hints").

Claim: "{text}"
"""

_CONSENSUS_CHOICES = ["established", "contested", "uncertain"]

_CONSENSUS_INSTRUCTIONS = """\
You are assessing the real-world scientific status of a claim, given a sample \
of related literature search results. Choose exactly one:
- established: the literature search results consistently support this as settled science.
- contested: the literature search results show disagreement or conflicting findings.
- uncertain: the literature search results are sparse, inconclusive, or don't clearly \
address this claim either way.
Base your answer only on the evidence provided, not on prior knowledge.
"""

# (expressed, actual) -> score. Asymmetric: overconfidence (expressed > actual)
# is penalised harder than underconfidence (expressed < actual), matching the
# framework's explicit concern that overconfident speculation is the more
# dangerous failure mode. A placeholder anchor table -- tune via expert
# elicitation like any other rubric anchor, not a fixed formula.
_CALIBRATION_SCORES = {
    ("high", "established"): 1.0,
    ("high", "contested"): 0.4,
    ("high", "uncertain"): 0.0,
    ("moderate", "established"): 0.8,
    ("moderate", "contested"): 1.0,
    ("moderate", "uncertain"): 0.6,
    ("low", "established"): 0.6,
    ("low", "contested"): 0.8,
    ("low", "uncertain"): 1.0,
}


def _classify_expressed_confidence(judge, claim_text: str) -> str:
    verdict = judge.judge(
        _CONFIDENCE_PROMPT.format(text=claim_text), choices=_CONFIDENCE_CHOICES
    ).verdict
    return verdict if verdict in _CONFIDENCE_CHOICES else "moderate"


def _classify_actual_consensus(judge, claim_text: str, search_summary: str) -> str:
    return judge_relationship(
        judge,
        label_a="Claim", text_a=claim_text,
        label_b="Related literature", text_b=search_summary,
        instructions=_CONSENSUS_INSTRUCTIONS,
        choices=_CONSENSUS_CHOICES,
    )


def _search_summary(literature, claim_text: str, limit: int = 3) -> str | None:
    hits = literature.search(claim_text, limit=limit)
    if not hits:
        return None
    lines = []
    for hit in hits:
        line = hit.title or "(untitled)"
        if hit.abstract:
            line += f" — {hit.abstract[:300]}"
        lines.append(line)
    return "\n".join(lines)


@register
class EpistemicCalibration(HypothesisMetric):
    name = "epistemic_calibration"
    axis = Axis.ACCURACY

    def score(self, hypothesis: Hypothesis, run: EvaluationRun, ctx: Context) -> MetricResult:
        claims = hypothesis.claims
        if ctx.judge is None:
            return MetricResult(metric=self.name, axis=self.axis, score=None,
                                evidence={"status": "no_judge"})
        if not claims:
            return MetricResult(metric=self.name, axis=self.axis, score=None,
                                evidence={"status": "no_claims"})
        if ctx.literature is None:
            return MetricResult(metric=self.name, axis=self.axis, score=None,
                                evidence={"status": "no_literature_client"})

        assessed = []
        unassessed = []
        for i, claim in enumerate(claims):
            expressed = _classify_expressed_confidence(ctx.judge, claim.text)
            summary = _search_summary(ctx.literature, claim.text)
            if summary is None:
                unassessed.append({
                    "i": i, "claim": claim.text, "expressed": expressed,
                    "reason": "no_literature_found",
                })
                continue
            actual = _classify_actual_consensus(ctx.judge, claim.text, summary)
            assessed.append({
                "i": i, "claim": claim.text,
                "expressed": expressed, "actual": actual,
                "calibration": _CALIBRATION_SCORES[(expressed, actual)],
            })

        if not assessed:
            return MetricResult(
                metric=self.name, axis=self.axis, score=None,
                evidence={"status": "no_claims_could_be_assessed", "unassessed": unassessed},
            )

        score = sum(a["calibration"] for a in assessed) / len(assessed)

        return MetricResult(
            metric=self.name, axis=self.axis,
            score=round(score, 4),
            anchor=_anchor(score),
            evidence={
                "n_claims": len(claims),
                "n_assessed": len(assessed),
                "assessed": assessed,
                "unassessed": unassessed,
                "judge": ctx.judge.name,
            },
        )


def _anchor(score: float) -> str:
    if score >= 0.9:
        return "well calibrated"
    if score >= 0.6:
        return "mostly calibrated, some mismatches"
    if score >= 0.3:
        return "poorly calibrated"
    return "systematically overconfident"
