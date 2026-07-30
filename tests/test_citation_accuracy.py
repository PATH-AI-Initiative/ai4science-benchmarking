"""Tests for citation_accuracy: existence checking (the score) and the
support-checking diagnostic layer (never folded into the score).
"""

from __future__ import annotations

from benchmarking_pipeline.axes.accuracy.citation_accuracy import CitationAccuracy
from benchmarking_pipeline.core.config import RunConfig
from benchmarking_pipeline.core.context import Context
from benchmarking_pipeline.core.models import (
    Claim,
    EvaluationRun,
    Hypothesis,
    HypothesisSet,
    Reference,
    Tool,
)
from benchmarking_pipeline.services.literature import PaperRecord
from benchmarking_pipeline.services.llm_judge import Judgement

METRIC = CitationAccuracy()


class FakeLiteratureClient:
    """Keyed by reference.raw / doi -> PaperRecord (or None -> not found)."""

    name = "fake"

    def __init__(self, records: dict[str, PaperRecord | None]):
        self._records = records

    def lookup_doi(self, doi: str):
        return self._records.get(doi)

    def search(self, query: str, limit: int = 3):
        record = self._records.get(query)
        return [record] if record is not None else []


class FakeJudge:
    """Returns a fixed verdict per call type, and records call count.

    ``verdict`` answers the support check; ``title_match_verdict`` answers the
    title-match check (defaults to "same_paper" so tests that don't care about
    title-matching get the old permissive behavior -- a fuzzy search hit
    trusted at face value -- without having to configure it explicitly).
    """

    name = "fake"

    def __init__(self, verdict: str, title_match_verdict: str = "same_paper"):
        self.verdict = verdict
        self.title_match_verdict = title_match_verdict
        self.calls = 0

    def judge(self, prompt: str, *, choices=None) -> Judgement:
        self.calls += 1
        if choices == ["same_paper", "different_paper"]:
            return Judgement(verdict=self.title_match_verdict, confidence=1.0)
        return Judgement(verdict=self.verdict, confidence=1.0)


def _hyp(claims: list[Claim]) -> tuple[Hypothesis, EvaluationRun]:
    hyp = Hypothesis(id="h1", text="...", rank=1, claims=claims)
    run = EvaluationRun(tool=Tool(name="t"), prompt="p", outputs=HypothesisSet(hypotheses=[hyp]))
    return hyp, run


def test_no_references_reports_not_assessed():
    hyp, run = _hyp([Claim(text="unsupported claim")])
    result = METRIC.score(hyp, run, Context(config=RunConfig(), literature=FakeLiteratureClient({})))
    assert result.score is None
    assert result.evidence["status"] == "no_references"


def test_no_literature_client_reports_not_assessed():
    hyp, run = _hyp([Claim(text="x", references=[Reference(raw="Some Paper")])])
    result = METRIC.score(hyp, run, Context(config=RunConfig(), literature=None))
    assert result.score is None
    assert result.evidence["status"] == "no_literature_client"


def test_existence_only_without_judge():
    real = PaperRecord(title="Real Paper", doi="10.1/x", year=2020, match_score=1.0)
    hyp, run = _hyp([Claim(text="x", references=[Reference(raw="Real Paper")])])
    ctx = Context(config=RunConfig(), literature=FakeLiteratureClient({"Real Paper": real}),
                 judge=None)

    result = METRIC.score(hyp, run, ctx)
    assert result.score == 1.0
    assert result.evidence["checked"][0]["support"] is None
    assert result.evidence["support_rate"] is None  # nothing was checked


def test_nonexistent_reference_scores_zero_and_is_never_support_checked():
    hyp, run = _hyp([Claim(text="x", references=[Reference(raw="Fabricated Paper")])])
    judge = FakeJudge(verdict="supports")
    ctx = Context(config=RunConfig(), literature=FakeLiteratureClient({}), judge=judge)

    result = METRIC.score(hyp, run, ctx)
    assert result.score == 0.0
    assert result.evidence["checked"][0]["exists"] is False
    assert result.evidence["checked"][0]["support"] is None
    assert judge.calls == 0  # never asked the judge about a citation that doesn't exist


def test_existing_reference_with_abstract_gets_support_checked():
    real = PaperRecord(title="Real Paper", doi="10.1/x", year=2020, match_score=1.0,
                       abstract="This paper shows X causes Y in mice.")
    claim = Claim(text="X causes Y.", references=[Reference(raw="Real Paper")])
    hyp, run = _hyp([claim])
    judge = FakeJudge(verdict="supports")
    ctx = Context(config=RunConfig(), literature=FakeLiteratureClient({"Real Paper": real}),
                 judge=judge)

    result = METRIC.score(hyp, run, ctx)
    assert result.evidence["checked"][0]["support"] == "supports"
    assert result.evidence["support_rate"] == 1.0
    assert result.evidence["n_support_checked"] == 1
    assert judge.calls == 2  # title-match (fuzzy search hit) + support check


def test_existing_reference_without_abstract_is_not_support_checked():
    real = PaperRecord(title="Real Paper", doi="10.1/x", year=2020, match_score=1.0,
                       abstract=None)
    claim = Claim(text="X causes Y.", references=[Reference(raw="Real Paper")])
    hyp, run = _hyp([claim])
    judge = FakeJudge(verdict="supports")
    ctx = Context(config=RunConfig(), literature=FakeLiteratureClient({"Real Paper": real}),
                 judge=judge)

    result = METRIC.score(hyp, run, ctx)
    assert result.evidence["checked"][0]["support"] is None
    assert result.evidence["support_rate"] is None
    assert judge.calls == 1  # title-match check still runs; no abstract means no support check


def test_existence_score_unaffected_by_a_contradicting_source():
    """A citation can be perfectly real and still misused -- exists=True must
    hold regardless of what the support-check finds, since 'contradicts' is
    reported as a diagnostic, not folded back into the existence score.
    """
    real = PaperRecord(title="Real Paper", doi="10.1/x", year=2020, match_score=1.0,
                       abstract="This paper found no effect of X on Y.")
    claim = Claim(text="X causes Y.", references=[Reference(raw="Real Paper")])
    hyp, run = _hyp([claim])
    judge = FakeJudge(verdict="contradicts")
    ctx = Context(config=RunConfig(), literature=FakeLiteratureClient({"Real Paper": real}),
                 judge=judge)

    result = METRIC.score(hyp, run, ctx)
    assert result.score == 1.0  # existence, not support, drives the score
    assert result.evidence["checked"][0]["exists"] is True
    assert result.evidence["checked"][0]["support"] == "contradicts"
    assert result.evidence["support_rate"] == 0.0  # 0 of 1 checked references "supports"


def test_fuzzy_match_on_shared_wording_but_different_topic_is_rejected():
    """Regression test: bibliographic search relevance is text-overlap, not
    topical -- a press-release-style title can out-score-match an unrelated
    paper on generic shared phrasing (observed for real against CrossRef:
    a malaria drug trial announcement matched a diabetes drug trial
    announcement at match_score >= 0.7 purely on "phase N trial ... meets
    primary endpoint" overlap). The title-match check must catch this.
    """
    wrong_paper = PaperRecord(title="Saxagliptin meets primary safety endpoint in a phase IV trial",
                              doi="10.1/wrong", year=2019, match_score=0.85)
    claim = Claim(text="GanLum has completed Phase 3 trials.",
                 references=[Reference(raw="[65]",
                                       title="Phase 3 trial for GanLum meets primary endpoint")])
    hyp, run = _hyp([claim])
    judge = FakeJudge(verdict="supports", title_match_verdict="different_paper")
    ctx = Context(
        config=RunConfig(),
        literature=FakeLiteratureClient({"Phase 3 trial for GanLum meets primary endpoint": wrong_paper}),
        judge=judge,
    )

    result = METRIC.score(hyp, run, ctx)
    assert result.score == 0.0
    assert result.evidence["checked"][0]["exists"] is False
    assert result.evidence["checked"][0]["title_match"] == "different_paper"
    assert result.evidence["checked"][0]["support"] is None  # never support-checked a rejected match


def test_exact_doi_lookup_skips_title_match_check():
    """An exact DOI resolution is already authoritative -- no need to spend a
    judge call re-verifying it's the same paper."""
    real = PaperRecord(title="Real Paper", doi="10.1/x", year=2020, match_score=1.0)
    claim = Claim(text="x", references=[Reference(raw="Real Paper", doi="10.1/x")])
    hyp, run = _hyp([claim])
    judge = FakeJudge(verdict="supports")
    ctx = Context(config=RunConfig(), literature=FakeLiteratureClient({"10.1/x": real}), judge=judge)

    result = METRIC.score(hyp, run, ctx)
    assert result.score == 1.0
    assert result.evidence["checked"][0]["title_match"] is None
    assert judge.calls == 0  # no abstract to support-check, and no title-match call either


def test_fuzzy_match_trusted_at_face_value_without_a_judge():
    """No judge configured -> title-match check is skipped entirely (can't
    run it), so a fuzzy match is trusted as before -- graceful degradation,
    not a stricter default that would need a judge to ever pass."""
    maybe_wrong = PaperRecord(title="Some Match", doi="10.1/y", year=2020, match_score=0.85)
    claim = Claim(text="x", references=[Reference(raw="Some Match")])
    hyp, run = _hyp([claim])
    ctx = Context(config=RunConfig(), literature=FakeLiteratureClient({"Some Match": maybe_wrong}),
                 judge=None)

    result = METRIC.score(hyp, run, ctx)
    assert result.score == 1.0
    assert result.evidence["checked"][0]["title_match"] is None


def test_mixed_hypothesis_support_rate_over_checked_subset_only():
    supported = PaperRecord(title="A", doi="10.1/a", year=2020, match_score=1.0,
                            abstract="Confirms the claim.")
    no_abstract = PaperRecord(title="B", doi="10.1/b", year=2020, match_score=1.0,
                              abstract=None)
    claims = [
        Claim(text="claim A", references=[Reference(raw="Paper A")]),
        Claim(text="claim B", references=[Reference(raw="Paper B")]),
        Claim(text="claim C", references=[Reference(raw="Fabricated Paper C")]),
    ]
    hyp, run = _hyp(claims)
    judge = FakeJudge(verdict="supports")
    ctx = Context(
        config=RunConfig(),
        literature=FakeLiteratureClient({"Paper A": supported, "Paper B": no_abstract}),
        judge=judge,
    )

    result = METRIC.score(hyp, run, ctx)
    assert result.score == 2 / 3  # A and B exist, C is fabricated
    assert result.evidence["n_support_checked"] == 1  # only A had exists+abstract+judge
    assert result.evidence["support_rate"] == 1.0
