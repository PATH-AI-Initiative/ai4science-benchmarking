"""Tests for citation_support: does an existing citation's abstract actually
support the claim it's attached to? A sibling of citation_accuracy, gated on
its existence/title-match resolution but scored independently.
"""

from __future__ import annotations

from benchmarking_pipeline.axes.accuracy.citation_support import CitationSupport
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

METRIC = CitationSupport()


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
    title-match check spent resolving existence (defaults to "same_paper" so
    tests that don't care about title-matching get the permissive behavior
    without configuring it explicitly).
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


def test_no_judge_reports_not_assessed():
    hyp, run = _hyp([Claim(text="x", references=[Reference(raw="Some Paper")])])
    result = METRIC.score(
        hyp, run, Context(config=RunConfig(), literature=FakeLiteratureClient({}), judge=None)
    )
    assert result.score is None
    assert result.evidence["status"] == "no_judge"


def test_nonexistent_reference_is_never_support_checked():
    hyp, run = _hyp([Claim(text="x", references=[Reference(raw="Fabricated Paper")])])
    judge = FakeJudge(verdict="supports")
    ctx = Context(config=RunConfig(), literature=FakeLiteratureClient({}), judge=judge)

    result = METRIC.score(hyp, run, ctx)
    assert result.score is None
    assert result.evidence["status"] == "no_checkable_citations"
    assert result.evidence["checked"][0]["exists"] is False
    assert result.evidence["checked"][0]["support"] is None


def test_existing_reference_with_abstract_gets_support_checked():
    real = PaperRecord(title="Real Paper", doi="10.1/x", year=2020, match_score=1.0,
                       abstract="This paper shows X causes Y in mice.")
    claim = Claim(text="X causes Y.", references=[Reference(raw="Real Paper")])
    hyp, run = _hyp([claim])
    judge = FakeJudge(verdict="supports")
    ctx = Context(config=RunConfig(), literature=FakeLiteratureClient({"Real Paper": real}),
                 judge=judge)

    result = METRIC.score(hyp, run, ctx)
    assert result.score == 1.0
    assert result.evidence["checked"][0]["support"] == "supports"
    assert result.evidence["n_checked"] == 1
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
    assert result.score is None
    assert result.evidence["status"] == "no_checkable_citations"
    assert judge.calls == 1  # title-match check still runs; no abstract means no support check


def test_contradicting_source_scores_against_support():
    real = PaperRecord(title="Real Paper", doi="10.1/x", year=2020, match_score=1.0,
                       abstract="This paper found no effect of X on Y.")
    claim = Claim(text="X causes Y.", references=[Reference(raw="Real Paper")])
    hyp, run = _hyp([claim])
    judge = FakeJudge(verdict="contradicts")
    ctx = Context(config=RunConfig(), literature=FakeLiteratureClient({"Real Paper": real}),
                 judge=judge)

    result = METRIC.score(hyp, run, ctx)
    assert result.score == 0.0  # 0 of 1 checked references "supports"
    assert result.evidence["checked"][0]["support"] == "contradicts"


def test_mixed_hypothesis_score_over_checked_subset_only():
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
    assert result.score == 1.0  # only A was checkable (exists + abstract), and it supports
    assert result.evidence["n_checked"] == 1
    assert result.evidence["n_total"] == 3


def test_citation_resolution_is_shared_with_citation_accuracy_via_cache():
    """The existence/title-match resolution should only be computed once per
    reference per Context, regardless of which metric asks for it first --
    otherwise adding this metric would double literature-API and
    title-match-judge calls for every citation citation_accuracy already
    resolves."""
    from benchmarking_pipeline.axes.accuracy.citation_accuracy import CitationAccuracy

    real = PaperRecord(title="Real Paper", doi="10.1/x", year=2020, match_score=1.0,
                       abstract="This paper shows X causes Y in mice.")
    claim = Claim(text="X causes Y.", references=[Reference(raw="Real Paper")])
    hyp, run = _hyp([claim])
    judge = FakeJudge(verdict="supports")
    ctx = Context(config=RunConfig(), literature=FakeLiteratureClient({"Real Paper": real}),
                 judge=judge)

    CitationAccuracy().score(hyp, run, ctx)
    assert judge.calls == 1  # title-match only

    METRIC.score(hyp, run, ctx)
    assert judge.calls == 2  # support check only -- title-match reused from cache
