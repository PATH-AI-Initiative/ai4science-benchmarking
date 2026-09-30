"""Tests for size_of_leap: distance from each hypothesis to its nearest match
in a fixed set of reference anchor papers, resolved once per run (via the
run's prompt) and shared across every hypothesis, aggregated to a set-level
mean.
"""

from __future__ import annotations

from benchmarking_pipeline.axes.novelty.size_of_leap import SizeOfLeap
from benchmarking_pipeline.core.config import RunConfig
from benchmarking_pipeline.core.context import Context
from benchmarking_pipeline.core.models import Claim, EvaluationRun, Hypothesis, HypothesisSet, Tool
from benchmarking_pipeline.services.embeddings import HashingEmbedding
from benchmarking_pipeline.services.literature import PaperRecord
from benchmarking_pipeline.services.llm_judge import Judgement

METRIC = SizeOfLeap()

PROMPT = "novel hypotheses for artemisinin-resistant Plasmodium knowlesi malaria"


class FakeLiteratureClient:
    """Keyed by search query -> list of PaperRecord (the anchor candidates)."""

    name = "fake"

    def __init__(self, results: dict[str, list[PaperRecord]]):
        self._results = results

    def lookup_doi(self, doi: str):
        return None

    def search(self, query: str, limit: int = 3):
        return self._results.get(query, [])[:limit]


class FakeJudge:
    """Returns "same_topic" for any candidate title in ``same_topic_titles``,
    "different_topic" otherwise (the safe default), and records call count."""

    name = "fake"

    def __init__(self, same_topic_titles: set[str] = frozenset()):
        self.same_topic_titles = same_topic_titles
        self.calls = 0

    def judge(self, prompt: str, *, choices=None) -> Judgement:
        self.calls += 1
        verdict = "different_topic"
        for title in self.same_topic_titles:
            if title in prompt:
                verdict = "same_topic"
                break
        return Judgement(verdict=verdict, confidence=1.0)


def _run(hyps: list[Hypothesis], prompt: str = PROMPT) -> EvaluationRun:
    return EvaluationRun(tool=Tool(name="t"), prompt=prompt, outputs=HypothesisSet(hypotheses=hyps))


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


def test_no_anchors_found_for_the_prompt_reports_not_assessed():
    hyp = Hypothesis(id="h1", text="an idea", rank=1)
    run = _run([hyp])
    ctx = Context(config=RunConfig(), embeddings=HashingEmbedding(), literature=FakeLiteratureClient({}))
    result = METRIC.score(run.outputs, run, ctx)
    assert result.score is None
    assert result.evidence["status"] == "no_literature_found"


def test_near_identical_anchor_gives_low_distance():
    text = "Kelch13 propeller domain mutations reduce artemisinin binding affinity"
    hyp = Hypothesis(id="h1", text=text, rank=1)
    run = _run([hyp])
    lit = FakeLiteratureClient({
        PROMPT: [PaperRecord(title=text, doi="10.1/x", year=2020, match_score=1.0)],
    })
    ctx = Context(config=RunConfig(), embeddings=HashingEmbedding(), literature=lit)

    result = METRIC.score(run.outputs, run, ctx)
    assert result.score < 0.1  # identical text -> ~0 distance
    assert result.anchor == "closely matches the reference anchors (restatement-like)"


def test_unrelated_anchor_gives_high_distance():
    hyp_text = "Kelch13 propeller domain mutations reduce artemisinin binding affinity"
    hyp = Hypothesis(id="h1", text=hyp_text, rank=1)
    run = _run([hyp])
    lit = FakeLiteratureClient({
        PROMPT: [PaperRecord(
            title="Completely unrelated economic policy analysis of urban housing markets",
            doi="10.1/y", year=2020, match_score=0.3,
        )],
    })
    ctx = Context(config=RunConfig(), embeddings=HashingEmbedding(), literature=lit)

    result = METRIC.score(run.outputs, run, ctx)
    assert result.score > 0.8
    assert result.anchor == "far from the reference anchors (verify grounding via the Accuracy axis)"


def test_nearest_anchor_is_the_best_match_not_the_first_result():
    hyp_text = "Kelch13 propeller domain mutations reduce artemisinin binding affinity"
    hyp = Hypothesis(id="h1", text=hyp_text, rank=1)
    run = _run([hyp])
    lit = FakeLiteratureClient({
        PROMPT: [
            PaperRecord(title="Unrelated urban housing economics", doi="10.1/a",
                       year=2020, match_score=0.3),
            PaperRecord(title=hyp_text, doi="10.1/b", year=2020, match_score=1.0),  # exact match
        ],
    })
    ctx = Context(config=RunConfig(), embeddings=HashingEmbedding(), literature=lit)

    result = METRIC.score(run.outputs, run, ctx)
    assert result.evidence["per_hypothesis"][0]["nearest_anchor_title"] == hyp_text
    assert result.score < 0.1


def test_anchor_set_is_shared_across_hypotheses_each_finding_its_own_match():
    """Every hypothesis in a run is read against the identical anchor set --
    but distinct hypotheses should still find their own best-matching anchor
    within that shared pool, not all collapse onto the same one."""
    hyp_a = Hypothesis(id="a", text="A gauge equivalence hypothesis", rank=1)
    hyp_b = Hypothesis(id="b", text="A conformal prediction hypothesis", rank=2)
    run = _run([hyp_a, hyp_b])

    paper_a = "A gauge equivalence hypothesis"  # exact match to hyp_a
    paper_b = "A conformal prediction hypothesis"  # exact match to hyp_b
    lit = FakeLiteratureClient({
        PROMPT: [
            PaperRecord(title=paper_a, doi="10.1/a", year=2024, match_score=1.0),
            PaperRecord(title=paper_b, doi="10.1/b", year=2024, match_score=1.0),
        ],
    })
    ctx = Context(config=RunConfig(), embeddings=HashingEmbedding(), literature=lit)

    result = METRIC.score(run.outputs, run, ctx)
    titles = {a["hypothesis_id"]: a["nearest_anchor_title"] for a in result.evidence["per_hypothesis"]}
    assert titles == {"a": paper_a, "b": paper_b}
    assert len(result.evidence["anchors"]) == 2


def test_anchor_search_runs_once_and_is_cached_across_hypotheses():
    """Resolving the anchor set is one literature search per run (keyed by
    prompt), not one per hypothesis -- otherwise every hypothesis in a run,
    and every run in a bundle sharing the same prompt, would re-search for
    what should be the identical fixed reference set."""
    calls = []

    class CountingLiteratureClient(FakeLiteratureClient):
        def search(self, query, limit=3):
            calls.append(query)
            return super().search(query, limit)

    hyp_a = Hypothesis(id="a", text="idea A", rank=1)
    hyp_b = Hypothesis(id="b", text="idea B", rank=2)
    run = _run([hyp_a, hyp_b])
    lit = CountingLiteratureClient({
        PROMPT: [PaperRecord(title="idea A", doi="10.1/a", year=2024, match_score=1.0)],
    })
    ctx = Context(config=RunConfig(), embeddings=HashingEmbedding(), literature=lit)

    METRIC.score(run.outputs, run, ctx)
    assert calls == [PROMPT]  # one search total, not one per hypothesis


def test_mixed_set_scores_only_over_topically_assessed_hypotheses():
    on_topic_hyp = Hypothesis(id="h1", text="malaria drug resistance mechanism", rank=1)
    off_topic_hyp = Hypothesis(id="h2", text="unrelated tangent", rank=2)
    run = _run([on_topic_hyp, off_topic_hyp])

    anchor_title = "Drug resistance pathways in Plasmodium species"
    lit = FakeLiteratureClient({
        PROMPT: [PaperRecord(title=anchor_title, doi="10.1/x", year=2020, match_score=1.0)],
    })
    judge = FakeJudge(same_topic_titles={"malaria drug resistance mechanism"})
    ctx = Context(config=RunConfig(), embeddings=HashingEmbedding(), literature=lit, judge=judge)

    result = METRIC.score(run.outputs, run, ctx)
    assert result.evidence["n_hypotheses"] == 2
    assert result.evidence["n_assessed"] == 1
    assert len(result.evidence["unassessed"]) == 1
    assert result.evidence["unassessed"][0]["reason"] == "no_topically_relevant_anchor"


def test_topically_unrelated_high_scoring_match_is_rejected_with_a_judge():
    """Regression test: a bibliographic search can return a paper that shares
    generic ML vocabulary (e.g. "meta-learning") with the prompt while being
    from an unrelated field entirely -- observed for real via
    citation_accuracy's Saxagliptin false positive, and the same failure mode
    applies here since anchor selection uses the same literature.search(). A
    high embedding similarity alone must not be enough to trust the match.
    """
    hyp_text = "Meta-learning a shared prior across tissue types for fast model selection"
    hyp = Hypothesis(id="h1", text=hyp_text, rank=1)
    run = _run([hyp])
    wrong_paper_title = "Meta-learning for few-shot meningioma segmentation"
    lit = FakeLiteratureClient({
        PROMPT: [PaperRecord(title=wrong_paper_title, doi="10.1/wrong",
                             year=2020, match_score=0.9)],
    })
    judge = FakeJudge(same_topic_titles=set())  # everything is "different_topic"
    ctx = Context(config=RunConfig(), embeddings=HashingEmbedding(), literature=lit, judge=judge)

    result = METRIC.score(run.outputs, run, ctx)
    assert result.score is None
    assert result.evidence["status"] == "no_hypotheses_could_be_assessed"
    assert result.evidence["unassessed"][0]["reason"] == "no_topically_relevant_anchor"


def test_topically_relevant_match_still_scores_with_a_judge():
    hyp_text = "Kelch13 propeller domain mutations reduce artemisinin binding affinity"
    hyp = Hypothesis(id="h1", text=hyp_text, rank=1)
    run = _run([hyp])
    lit = FakeLiteratureClient({
        PROMPT: [PaperRecord(title=hyp_text, doi="10.1/x", year=2020, match_score=1.0)],
    })
    judge = FakeJudge(same_topic_titles={hyp_text})
    ctx = Context(config=RunConfig(), embeddings=HashingEmbedding(), literature=lit, judge=judge)

    result = METRIC.score(run.outputs, run, ctx)
    assert result.score < 0.1
    assert judge.calls == 1


def test_falls_through_to_next_best_anchor_when_top_match_is_off_topic():
    hyp_text = "Conformal prediction sets for diffusion MRI model selection"
    hyp = Hypothesis(id="h1", text=hyp_text, rank=1)
    run = _run([hyp])
    off_topic = "Conformal prediction sets for tabular regression under covariate shift"
    on_topic = "Distribution-free coverage guarantees for diffusion MRI compartment models"
    lit = FakeLiteratureClient({
        # off_topic embeds closer under a naive hashing scheme by sharing more
        # tokens with hyp_text; the judge should skip it and fall through.
        PROMPT: [
            PaperRecord(title=off_topic, doi="10.1/off", year=2024, match_score=0.9),
            PaperRecord(title=on_topic, doi="10.1/on", year=2024, match_score=0.7),
        ],
    })
    judge = FakeJudge(same_topic_titles={on_topic})
    ctx = Context(config=RunConfig(), embeddings=HashingEmbedding(), literature=lit, judge=judge)

    result = METRIC.score(run.outputs, run, ctx)
    assert result.score is not None
    assert result.evidence["per_hypothesis"][0]["nearest_anchor_title"] == on_topic


def test_no_judge_trusts_best_embedding_match_at_face_value():
    """No judge configured -> topical-relevance check is skipped entirely
    (can't run it), so the best embedding match is trusted as before --
    graceful degradation, matching the old behavior exactly."""
    hyp_text = "Kelch13 propeller domain mutations reduce artemisinin binding affinity"
    hyp = Hypothesis(id="h1", text=hyp_text, rank=1)
    run = _run([hyp])
    lit = FakeLiteratureClient({
        PROMPT: [PaperRecord(title="Something unrelated", doi="10.1/y",
                             year=2020, match_score=0.9)],
    })
    ctx = Context(config=RunConfig(), embeddings=HashingEmbedding(), literature=lit, judge=None)

    result = METRIC.score(run.outputs, run, ctx)
    assert result.score is not None  # trusted at face value, not rejected


def test_claims_enrich_the_comparison_text():
    """A hypothesis's own title is often domain-free on its own (e.g. never
    says "diffusion MRI"); its claims are naturally domain-grounded and
    should be folded into both the embedding comparison and the topical-match
    judge call, via the same ``hypothesis_text`` helper reproducibility/
    robustness use -- not compared on the bare title alone."""
    hyp_text = "Conformalized model selection with finite-sample coverage guarantees"
    claim = Claim(text="Existing model selection methods for diffusion MRI lack "
                       "finite-sample coverage guarantees.")
    hyp = Hypothesis(id="h1", text=hyp_text, rank=1, claims=[claim])
    run = _run([hyp])
    anchor_title = "Finite-sample coverage guarantees for high-dimensional model selection"
    lit = FakeLiteratureClient({
        PROMPT: [PaperRecord(title=anchor_title, doi="10.1/x", year=2024, match_score=1.0)],
    })
    judge = FakeJudge(same_topic_titles={"diffusion MRI"})
    ctx = Context(config=RunConfig(), embeddings=HashingEmbedding(), literature=lit, judge=judge)

    result = METRIC.score(run.outputs, run, ctx)
    # The judge only recognises "same_topic" when "diffusion MRI" appears in
    # the prompt it's shown -- which only happens if the claim (not the bare
    # domain-free title) was folded into the comparison text.
    assert result.score is not None
    assert result.evidence["per_hypothesis"][0]["nearest_anchor_title"] == anchor_title
