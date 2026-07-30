"""Tests for epistemic_calibration: expressed-confidence classification,
literature-grounded consensus classification, and the asymmetric calibration
scoring table (overconfidence penalised harder than underconfidence).
"""

from __future__ import annotations

from benchmarking_pipeline.axes.accuracy.epistemic_calibration import EpistemicCalibration
from benchmarking_pipeline.core.config import RunConfig
from benchmarking_pipeline.core.context import Context
from benchmarking_pipeline.core.models import Claim, EvaluationRun, Hypothesis, HypothesisSet, Tool
from benchmarking_pipeline.services.literature import PaperRecord
from benchmarking_pipeline.services.llm_judge import Judgement

METRIC = EpistemicCalibration()


class FakeJudge:
    """Distinguishes confidence-classification calls (choices include "high")
    from consensus-classification calls (choices include "established") via
    lookup tables keyed by claim text substring."""

    name = "fake"

    def __init__(self, confidence: dict[str, str] | None = None,
                consensus: dict[str, str] | None = None):
        self.confidence = confidence or {}
        self.consensus = consensus or {}

    def judge(self, prompt: str, *, choices: list[str] | None = None) -> Judgement:
        if choices and "high" in choices:
            for text, level in self.confidence.items():
                if text in prompt:
                    return Judgement(verdict=level, confidence=1.0)
            return Judgement(verdict="moderate", confidence=1.0)
        for text, level in self.consensus.items():
            if text in prompt:
                return Judgement(verdict=level, confidence=1.0)
        return Judgement(verdict="uncertain", confidence=1.0)


class FakeLiteratureClient:
    name = "fake"

    def __init__(self, results: dict[str, list[PaperRecord]]):
        self._results = results

    def lookup_doi(self, doi: str):
        return None

    def search(self, query: str, limit: int = 3):
        return self._results.get(query, [])[:limit]


def _hyp(claims: list[Claim]) -> tuple[Hypothesis, EvaluationRun]:
    hyp = Hypothesis(id="h1", text="...", rank=1, claims=claims)
    run = EvaluationRun(tool=Tool(name="t"), prompt="p", outputs=HypothesisSet(hypotheses=[hyp]))
    return hyp, run


_A_PAPER = [PaperRecord(title="A related paper", doi="10.1/x", year=2020, match_score=0.9,
                        abstract="Some relevant abstract content.")]


def test_no_judge_reports_not_assessed():
    hyp, run = _hyp([Claim(text="x")])
    ctx = Context(config=RunConfig(), judge=None)
    result = METRIC.score(hyp, run, ctx)
    assert result.score is None
    assert result.evidence["status"] == "no_judge"


def test_no_claims_reports_not_assessed():
    hyp, run = _hyp([])
    ctx = Context(config=RunConfig(), judge=FakeJudge())
    result = METRIC.score(hyp, run, ctx)
    assert result.score is None
    assert result.evidence["status"] == "no_claims"


def test_no_literature_client_reports_not_assessed():
    hyp, run = _hyp([Claim(text="x")])
    ctx = Context(config=RunConfig(), judge=FakeJudge(), literature=None)
    result = METRIC.score(hyp, run, ctx)
    assert result.score is None
    assert result.evidence["status"] == "no_literature_client"


def test_claim_with_no_search_results_is_unassessed_not_scored():
    claim = Claim(text="An unusual claim with no literature hits.")
    hyp, run = _hyp([claim])
    ctx = Context(config=RunConfig(), judge=FakeJudge(), literature=FakeLiteratureClient({}))
    result = METRIC.score(hyp, run, ctx)
    assert result.score is None
    assert result.evidence["status"] == "no_claims_could_be_assessed"
    assert result.evidence["unassessed"][0]["reason"] == "no_literature_found"


def test_well_calibrated_high_confidence_established_scores_perfectly():
    claim = Claim(text="X definitively causes Y.")
    hyp, run = _hyp([claim])
    judge = FakeJudge(confidence={claim.text: "high"}, consensus={claim.text: "established"})
    ctx = Context(config=RunConfig(), judge=judge,
                 literature=FakeLiteratureClient({claim.text: _A_PAPER}))

    result = METRIC.score(hyp, run, ctx)
    assert result.score == 1.0
    assert result.anchor == "well calibrated"


def test_overconfident_claim_scores_zero_the_worst_case():
    claim = Claim(text="X definitively causes Y.")
    hyp, run = _hyp([claim])
    judge = FakeJudge(confidence={claim.text: "high"}, consensus={claim.text: "uncertain"})
    ctx = Context(config=RunConfig(), judge=judge,
                 literature=FakeLiteratureClient({claim.text: _A_PAPER}))

    result = METRIC.score(hyp, run, ctx)
    assert result.score == 0.0
    assert result.anchor == "systematically overconfident"


def test_underconfidence_is_penalised_less_than_overconfidence():
    """Same magnitude of mismatch (one notch off), opposite direction: the
    framework explicitly treats overconfidence as more dangerous than
    underconfidence, so underconfident-but-established must score higher than
    overconfident-but-contested.
    """
    over_claim = Claim(text="Overconfident claim.")
    under_claim = Claim(text="Underconfident claim.")
    hyp_over, run_over = _hyp([over_claim])
    hyp_under, run_under = _hyp([under_claim])

    judge = FakeJudge(
        confidence={over_claim.text: "high", under_claim.text: "low"},
        consensus={over_claim.text: "contested", under_claim.text: "established"},
    )
    lit = FakeLiteratureClient({over_claim.text: _A_PAPER, under_claim.text: _A_PAPER})

    over_result = METRIC.score(hyp_over, run_over, Context(config=RunConfig(), judge=judge, literature=lit))
    under_result = METRIC.score(hyp_under, run_under, Context(config=RunConfig(), judge=judge, literature=lit))

    assert under_result.score > over_result.score


def test_well_hedged_uncertain_claim_scores_perfectly():
    claim = Claim(text="X may possibly influence Y.")
    hyp, run = _hyp([claim])
    judge = FakeJudge(confidence={claim.text: "low"}, consensus={claim.text: "uncertain"})
    ctx = Context(config=RunConfig(), judge=judge,
                 literature=FakeLiteratureClient({claim.text: _A_PAPER}))

    result = METRIC.score(hyp, run, ctx)
    assert result.score == 1.0


def test_mixed_hypothesis_scores_only_over_assessable_claims():
    assessable = Claim(text="Assessable claim.")
    unassessable = Claim(text="No literature exists for this one.")
    hyp, run = _hyp([assessable, unassessable])

    judge = FakeJudge(confidence={assessable.text: "high"},
                      consensus={assessable.text: "established"})
    ctx = Context(config=RunConfig(), judge=judge,
                 literature=FakeLiteratureClient({assessable.text: _A_PAPER}))

    result = METRIC.score(hyp, run, ctx)
    assert result.score == 1.0  # only the assessable claim counted
    assert result.evidence["n_claims"] == 2
    assert result.evidence["n_assessed"] == 1
    assert len(result.evidence["unassessed"]) == 1
