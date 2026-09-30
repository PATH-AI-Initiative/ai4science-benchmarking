"""Tests for citation_accuracy: existence + title-match checking (the score).

Support-checking lives in test_citation_support.py -- it's a sibling metric
now, not a layer within this one.
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
    """Returns a fixed title-match verdict, and records call count."""

    name = "fake"

    def __init__(self, title_match_verdict: str = "same_paper"):
        self.title_match_verdict = title_match_verdict
        self.calls = 0

    def judge(self, prompt: str, *, choices=None) -> Judgement:
        self.calls += 1
        return Judgement(verdict=self.title_match_verdict, confidence=1.0)


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
    assert result.evidence["checked"][0]["exists"] is True


def test_nonexistent_reference_scores_zero():
    hyp, run = _hyp([Claim(text="x", references=[Reference(raw="Fabricated Paper")])])
    judge = FakeJudge()
    ctx = Context(config=RunConfig(), literature=FakeLiteratureClient({}), judge=judge)

    result = METRIC.score(hyp, run, ctx)
    assert result.score == 0.0
    assert result.evidence["checked"][0]["exists"] is False
    assert judge.calls == 0  # no candidate title to check a nonexistent citation against


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
    judge = FakeJudge(title_match_verdict="different_paper")
    ctx = Context(
        config=RunConfig(),
        literature=FakeLiteratureClient({"Phase 3 trial for GanLum meets primary endpoint": wrong_paper}),
        judge=judge,
    )

    result = METRIC.score(hyp, run, ctx)
    assert result.score == 0.0
    assert result.evidence["checked"][0]["exists"] is False
    assert result.evidence["checked"][0]["title_match"] == "different_paper"


def test_exact_doi_lookup_skips_title_match_check():
    """An exact DOI resolution is already authoritative -- no need to spend a
    judge call re-verifying it's the same paper."""
    real = PaperRecord(title="Real Paper", doi="10.1/x", year=2020, match_score=1.0)
    claim = Claim(text="x", references=[Reference(raw="Real Paper", doi="10.1/x")])
    hyp, run = _hyp([claim])
    judge = FakeJudge()
    ctx = Context(config=RunConfig(), literature=FakeLiteratureClient({"10.1/x": real}), judge=judge)

    result = METRIC.score(hyp, run, ctx)
    assert result.score == 1.0
    assert result.evidence["checked"][0]["title_match"] is None
    assert judge.calls == 0


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


def test_mixed_hypothesis_existence_over_all_references():
    supported = PaperRecord(title="A", doi="10.1/a", year=2020, match_score=1.0)
    claims = [
        Claim(text="claim A", references=[Reference(raw="Paper A")]),
        Claim(text="claim B", references=[Reference(raw="Fabricated Paper B")]),
    ]
    hyp, run = _hyp(claims)
    judge = FakeJudge()
    ctx = Context(
        config=RunConfig(),
        literature=FakeLiteratureClient({"Paper A": supported}),
        judge=judge,
    )

    result = METRIC.score(hyp, run, ctx)
    assert result.score == 0.5  # A exists, B is fabricated
