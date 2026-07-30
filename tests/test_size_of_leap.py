"""Tests for size_of_leap: literature-grounded nearest-neighbour distance per
hypothesis, aggregated to a set-level mean.
"""

from __future__ import annotations

from benchmarking_pipeline.axes.novelty.size_of_leap import SizeOfLeap
from benchmarking_pipeline.core.config import RunConfig
from benchmarking_pipeline.core.context import Context
from benchmarking_pipeline.core.models import EvaluationRun, Hypothesis, HypothesisSet, Tool
from benchmarking_pipeline.services.embeddings import HashingEmbedding
from benchmarking_pipeline.services.literature import PaperRecord

METRIC = SizeOfLeap()


class FakeLiteratureClient:
    name = "fake"

    def __init__(self, results: dict[str, list[PaperRecord]]):
        self._results = results

    def lookup_doi(self, doi: str):
        return None

    def search(self, query: str, limit: int = 3):
        return self._results.get(query, [])[:limit]


def _run(hyps: list[Hypothesis]) -> EvaluationRun:
    return EvaluationRun(tool=Tool(name="t"), prompt="p", outputs=HypothesisSet(hypotheses=hyps))


def test_no_embeddings_reports_not_assessed():
    hyp = Hypothesis(id="h1", text="x", rank=1)
    run = _run([hyp])
    ctx = Context(config=RunConfig(), embeddings=None, literature=FakeLiteratureClient({}))
    result = METRIC.score(run.outputs, run, ctx)
    assert result.score is None
    assert result.evidence["status"] == "no_embedding_model"


def test_no_literature_client_reports_not_assessed():
    hyp = Hypothesis(id="h1", text="x", rank=1)
    run = _run([hyp])
    ctx = Context(config=RunConfig(), embeddings=HashingEmbedding(), literature=None)
    result = METRIC.score(run.outputs, run, ctx)
    assert result.score is None
    assert result.evidence["status"] == "no_literature_client"


def test_hypothesis_with_no_search_results_is_unassessed():
    hyp = Hypothesis(id="h1", text="an idea with no literature hits", rank=1)
    run = _run([hyp])
    ctx = Context(config=RunConfig(), embeddings=HashingEmbedding(), literature=FakeLiteratureClient({}))
    result = METRIC.score(run.outputs, run, ctx)
    assert result.score is None
    assert result.evidence["status"] == "no_hypotheses_could_be_assessed"
    assert result.evidence["unassessed"][0]["reason"] == "no_literature_found"


def test_near_identical_literature_gives_low_distance():
    text = "Kelch13 propeller domain mutations reduce artemisinin binding affinity"
    hyp = Hypothesis(id="h1", text=text, rank=1)
    run = _run([hyp])
    lit = FakeLiteratureClient({
        text: [PaperRecord(title=text, doi="10.1/x", year=2020, match_score=1.0)],
    })
    ctx = Context(config=RunConfig(), embeddings=HashingEmbedding(), literature=lit)

    result = METRIC.score(run.outputs, run, ctx)
    assert result.score < 0.1  # identical text -> ~0 distance
    assert result.anchor == "closely matches existing literature (restatement-like)"


def test_unrelated_literature_gives_high_distance():
    hyp_text = "Kelch13 propeller domain mutations reduce artemisinin binding affinity"
    hyp = Hypothesis(id="h1", text=hyp_text, rank=1)
    run = _run([hyp])
    lit = FakeLiteratureClient({
        hyp_text: [PaperRecord(
            title="Completely unrelated economic policy analysis of urban housing markets",
            doi="10.1/y", year=2020, match_score=0.3,
        )],
    })
    ctx = Context(config=RunConfig(), embeddings=HashingEmbedding(), literature=lit)

    result = METRIC.score(run.outputs, run, ctx)
    assert result.score > 0.8
    assert result.anchor == "far from existing literature (verify grounding via the Accuracy axis)"


def test_nearest_neighbor_is_the_best_match_not_the_first_result():
    hyp_text = "Kelch13 propeller domain mutations reduce artemisinin binding affinity"
    hyp = Hypothesis(id="h1", text=hyp_text, rank=1)
    run = _run([hyp])
    lit = FakeLiteratureClient({
        hyp_text: [
            PaperRecord(title="Unrelated urban housing economics", doi="10.1/a",
                       year=2020, match_score=0.3),
            PaperRecord(title=hyp_text, doi="10.1/b", year=2020, match_score=1.0),  # exact match
        ],
    })
    ctx = Context(config=RunConfig(), embeddings=HashingEmbedding(), literature=lit)

    result = METRIC.score(run.outputs, run, ctx)
    assert result.evidence["per_hypothesis"][0]["nearest_neighbor_title"] == hyp_text
    assert result.score < 0.1


def test_mixed_set_scores_only_over_assessed_hypotheses():
    close_text = "Kelch13 propeller domain mutations reduce artemisinin binding affinity"
    no_hits_text = "a totally unsearched idea"
    close_hyp = Hypothesis(id="h1", text=close_text, rank=1)
    unassessable_hyp = Hypothesis(id="h2", text=no_hits_text, rank=2)
    run = _run([close_hyp, unassessable_hyp])

    lit = FakeLiteratureClient({
        close_text: [PaperRecord(title=close_text, doi="10.1/x", year=2020, match_score=1.0)],
    })
    ctx = Context(config=RunConfig(), embeddings=HashingEmbedding(), literature=lit)

    result = METRIC.score(run.outputs, run, ctx)
    assert result.evidence["n_hypotheses"] == 2
    assert result.evidence["n_assessed"] == 1
    assert len(result.evidence["unassessed"]) == 1
    assert result.score < 0.1  # only the close-match hypothesis counted
