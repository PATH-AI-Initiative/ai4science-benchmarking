"""Tests for diversity: mean pairwise embedding distance across a hypothesis
set's own texts.

Uses a deterministic FakeEmbedding keyed by exact text -> vector, so pairwise
cosine similarities can be constructed exactly (identical, orthogonal,
opposite) rather than relying on HashingEmbedding's emergent token-overlap
behavior -- needed here to pin the anchor thresholds precisely.
"""

from __future__ import annotations

import numpy as np

from benchmarking_pipeline.axes.novelty.diversity import Diversity, _anchor
from benchmarking_pipeline.core.config import RunConfig
from benchmarking_pipeline.core.context import Context
from benchmarking_pipeline.core.models import EvaluationRun, Hypothesis, HypothesisSet, Tool

METRIC = Diversity()


class FakeEmbedding:
    """Deterministic embedding keyed by exact text -> vector. Looking up text
    with no mapping raises, so a test's intent about what it's embedding is
    always explicit rather than silently falling back to a zero vector."""

    name = "fake"

    def __init__(self, vectors: dict[str, list[float]]):
        self._vectors = vectors

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array([self._vectors[t] for t in texts], dtype=np.float64)


def _run(hyps: list[Hypothesis]) -> EvaluationRun:
    return EvaluationRun(tool=Tool(name="t"), prompt="p", outputs=HypothesisSet(hypotheses=hyps))


def test_no_embeddings_reports_not_assessed():
    hyps = [Hypothesis(id="h1", text="A", rank=1), Hypothesis(id="h2", text="B", rank=2)]
    run = _run(hyps)
    ctx = Context(config=RunConfig(), embeddings=None)

    result = METRIC.score(run.outputs, run, ctx)
    assert result.score is None
    assert result.evidence["status"] == "no_embedding_model"


def test_empty_set_reports_not_assessed():
    run = _run([])
    ctx = Context(config=RunConfig(), embeddings=FakeEmbedding({}))

    result = METRIC.score(run.outputs, run, ctx)
    assert result.score is None
    assert result.evidence["status"] == "need_at_least_two_hypotheses"
    assert result.evidence["n"] == 0


def test_single_hypothesis_reports_not_assessed():
    hyp = Hypothesis(id="h1", text="Only one idea", rank=1)
    run = _run([hyp])
    ctx = Context(config=RunConfig(), embeddings=FakeEmbedding({hyp.text: [1.0, 0.0]}))

    result = METRIC.score(run.outputs, run, ctx)
    assert result.score is None
    assert result.evidence["status"] == "need_at_least_two_hypotheses"
    assert result.evidence["n"] == 1


def test_identical_hypotheses_score_as_clustered():
    hyps = [Hypothesis(id="h1", text="A", rank=1), Hypothesis(id="h2", text="B", rank=2)]
    run = _run(hyps)
    same_vector = [1.0, 0.0]
    ctx = Context(config=RunConfig(),
                 embeddings=FakeEmbedding({"A": same_vector, "B": same_vector}))

    result = METRIC.score(run.outputs, run, ctx)
    assert result.score == 0.0
    assert result.anchor == "clustered / redundant"
    assert result.evidence["mean_pairwise_similarity"] == 1.0


def test_orthogonal_hypotheses_score_as_broadly_spread():
    hyps = [Hypothesis(id="h1", text="A", rank=1), Hypothesis(id="h2", text="B", rank=2)]
    run = _run(hyps)
    ctx = Context(config=RunConfig(),
                 embeddings=FakeEmbedding({"A": [1.0, 0.0], "B": [0.0, 1.0]}))

    result = METRIC.score(run.outputs, run, ctx)
    assert result.score == 1.0
    assert result.anchor == "broadly spread across conceptual space"
    assert result.evidence["mean_pairwise_similarity"] == 0.0


def test_opposite_hypotheses_distance_is_clamped_to_one():
    """Cosine similarity for antipodal vectors is -1, which would make the
    raw distance (1 - similarity) = 2.0 -- the score must clamp to [0, 1]
    rather than reporting a nonsensical distance greater than 1."""
    hyps = [Hypothesis(id="h1", text="A", rank=1), Hypothesis(id="h2", text="B", rank=2)]
    run = _run(hyps)
    ctx = Context(config=RunConfig(),
                 embeddings=FakeEmbedding({"A": [1.0, 0.0], "B": [-1.0, 0.0]}))

    result = METRIC.score(run.outputs, run, ctx)
    assert result.score == 1.0
    assert result.evidence["mean_pairwise_similarity"] == -1.0
    assert result.evidence["mean_pairwise_distance"] == 2.0  # unclamped value still reported


def test_mean_is_over_all_pairs_not_just_consecutive_ones():
    """Three hypotheses: A~B orthogonal, A~C identical, B~C orthogonal.
    Mean similarity must average all three pairs (0 + 1 + 0) / 3, not just
    adjacent pairs in list order."""
    hyps = [Hypothesis(id="h1", text="A", rank=1),
           Hypothesis(id="h2", text="B", rank=2),
           Hypothesis(id="h3", text="C", rank=3)]
    run = _run(hyps)
    ctx = Context(config=RunConfig(), embeddings=FakeEmbedding({
        "A": [1.0, 0.0], "B": [0.0, 1.0], "C": [1.0, 0.0],
    }))

    result = METRIC.score(run.outputs, run, ctx)
    # Evidence values are rounded to 4 decimal places, so compare loosely.
    assert abs(result.evidence["mean_pairwise_similarity"] - 1 / 3) < 1e-4
    assert abs(result.score - 2 / 3) < 1e-4


def test_score_is_independent_of_hypothesis_order():
    hyps_a = [Hypothesis(id="h1", text="A", rank=1),
             Hypothesis(id="h2", text="B", rank=2),
             Hypothesis(id="h3", text="C", rank=3)]
    hyps_b = list(reversed(hyps_a))
    embeddings = FakeEmbedding({"A": [1.0, 0.0], "B": [0.0, 1.0], "C": [0.5, 0.5]})

    result_a = METRIC.score(_run(hyps_a).outputs, _run(hyps_a), Context(config=RunConfig(), embeddings=embeddings))
    result_b = METRIC.score(_run(hyps_b).outputs, _run(hyps_b), Context(config=RunConfig(), embeddings=embeddings))

    assert result_a.score == result_b.score


def test_evidence_reports_n_hypotheses_and_embedding_model_name():
    hyps = [Hypothesis(id="h1", text="A", rank=1), Hypothesis(id="h2", text="B", rank=2)]
    run = _run(hyps)
    ctx = Context(config=RunConfig(),
                 embeddings=FakeEmbedding({"A": [1.0, 0.0], "B": [0.0, 1.0]}))

    result = METRIC.score(run.outputs, run, ctx)
    assert result.evidence["n_hypotheses"] == 2
    assert result.evidence["embedding_model"] == "fake"
    assert result.evidence["mean_pairwise_similarity"] + result.evidence["mean_pairwise_distance"] == 1.0


def test_anchor_thresholds():
    assert _anchor(0.6) == "broadly spread across conceptual space"
    assert _anchor(0.5999) == "moderately diverse"
    assert _anchor(0.3) == "moderately diverse"
    assert _anchor(0.2999) == "clustered / redundant"
    assert _anchor(0.0) == "clustered / redundant"
    assert _anchor(1.0) == "broadly spread across conceptual space"
