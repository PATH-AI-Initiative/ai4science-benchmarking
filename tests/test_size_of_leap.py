"""Tests for size_of_leap: literature-grounded nearest-neighbour distance per
hypothesis, aggregated to a set-level mean.
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


class FakeLiteratureClient:
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


def _run(hyps: list[Hypothesis], prompt: str = "") -> EvaluationRun:
    # Empty prompt by default: most tests here are about the scoring/matching
    # logic and key FakeLiteratureClient by the bare hypothesis text, which a
    # non-empty prompt would change the search query away from (see
    # test_prompt_is_folded_into_the_search_query for that behavior itself).
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


def test_topically_unrelated_high_scoring_match_is_rejected_with_a_judge():
    """Regression test: a bibliographic search can return a paper that shares
    generic ML vocabulary (e.g. "meta-learning") with the hypothesis while
    being from an unrelated field entirely -- observed for real via
    citation_accuracy's Saxagliptin false positive, and the same failure mode
    applies here since size_of_leap uses the same literature.search(). A
    high embedding similarity alone must not be enough to trust the match.
    """
    hyp_text = "Meta-learning a shared prior across tissue types for fast model selection"
    hyp = Hypothesis(id="h1", text=hyp_text, rank=1)
    run = _run([hyp])
    wrong_paper_title = "Meta-learning for few-shot meningioma segmentation"
    lit = FakeLiteratureClient({
        hyp_text: [PaperRecord(title=wrong_paper_title, doi="10.1/wrong",
                               year=2020, match_score=0.9)],
    })
    judge = FakeJudge(same_topic_titles=set())  # everything is "different_topic"
    ctx = Context(config=RunConfig(), embeddings=HashingEmbedding(), literature=lit, judge=judge)

    result = METRIC.score(run.outputs, run, ctx)
    assert result.score is None
    assert result.evidence["status"] == "no_hypotheses_could_be_assessed"
    assert result.evidence["unassessed"][0]["reason"] == "no_topically_relevant_match"


def test_topically_relevant_match_still_scores_with_a_judge():
    hyp_text = "Kelch13 propeller domain mutations reduce artemisinin binding affinity"
    hyp = Hypothesis(id="h1", text=hyp_text, rank=1)
    run = _run([hyp])
    lit = FakeLiteratureClient({
        hyp_text: [PaperRecord(title=hyp_text, doi="10.1/x", year=2020, match_score=1.0)],
    })
    judge = FakeJudge(same_topic_titles={hyp_text})
    ctx = Context(config=RunConfig(), embeddings=HashingEmbedding(), literature=lit, judge=judge)

    result = METRIC.score(run.outputs, run, ctx)
    assert result.score < 0.1
    assert judge.calls == 1


def test_falls_through_to_next_best_candidate_when_top_match_is_off_topic():
    hyp_text = "Conformal prediction sets for diffusion MRI model selection"
    hyp = Hypothesis(id="h1", text=hyp_text, rank=1)
    run = _run([hyp])
    off_topic = "Conformal prediction sets for tabular regression under covariate shift"
    on_topic = "Distribution-free coverage guarantees for diffusion MRI compartment models"
    lit = FakeLiteratureClient({
        # off_topic embeds closer under a naive hashing scheme by sharing more
        # tokens with hyp_text; the judge should skip it and fall through.
        hyp_text: [
            PaperRecord(title=off_topic, doi="10.1/off", year=2024, match_score=0.9),
            PaperRecord(title=on_topic, doi="10.1/on", year=2024, match_score=0.7),
        ],
    })
    judge = FakeJudge(same_topic_titles={on_topic})
    ctx = Context(config=RunConfig(), embeddings=HashingEmbedding(), literature=lit, judge=judge)

    result = METRIC.score(run.outputs, run, ctx)
    assert result.score is not None
    assert result.evidence["per_hypothesis"][0]["nearest_neighbor_title"] == on_topic


def test_no_judge_trusts_best_embedding_match_at_face_value():
    """No judge configured -> topical-relevance check is skipped entirely
    (can't run it), so the best embedding match is trusted as before --
    graceful degradation, matching the old behavior exactly."""
    hyp_text = "Kelch13 propeller domain mutations reduce artemisinin binding affinity"
    hyp = Hypothesis(id="h1", text=hyp_text, rank=1)
    run = _run([hyp])
    lit = FakeLiteratureClient({
        hyp_text: [PaperRecord(title="Something unrelated", doi="10.1/y",
                               year=2020, match_score=0.9)],
    })
    ctx = Context(config=RunConfig(), embeddings=HashingEmbedding(), literature=lit, judge=None)

    result = METRIC.score(run.outputs, run, ctx)
    assert result.score is not None  # trusted at face value, not rejected


def test_claims_enrich_the_search_query_when_available():
    """A hypothesis's own title is often domain-free (e.g. never says
    "diffusion MRI"), which starves the bibliographic search of the one word
    that would actually distinguish real prior art from generically-worded
    papers in unrelated fields. Its own claim text is naturally domain-
    grounded (a premise like "existing methods for diffusion MRI optimize
    ...") and should be folded into the query."""
    hyp_text = "Conformalized model selection with finite-sample coverage guarantees"
    claim = Claim(text="Existing model selection methods for diffusion MRI lack "
                       "finite-sample coverage guarantees.")
    hyp = Hypothesis(id="h1", text=hyp_text, rank=1, claims=[claim])
    run = _run([hyp])  # no prompt -- claims alone should be enough
    enriched_query = f"{hyp_text}. {claim.text}"
    lit = FakeLiteratureClient({
        enriched_query: [PaperRecord(title=hyp_text, doi="10.1/x", year=2024, match_score=1.0)],
        hyp_text: [PaperRecord(title="a decoy the bare-title query would have hit",
                               doi="10.1/decoy", year=2024, match_score=0.9)],
    })
    ctx = Context(config=RunConfig(), embeddings=HashingEmbedding(), literature=lit, judge=None)

    result = METRIC.score(run.outputs, run, ctx)
    assert result.evidence["per_hypothesis"][0]["nearest_neighbor_title"] == hyp_text


def test_prompt_is_only_a_fallback_when_a_hypothesis_has_no_claims():
    """A hand-authored/undecomposed capture (no claims) has nothing
    hypothesis-specific to draw on, so the run's prompt is used instead --
    better than searching the bare, possibly domain-free title alone."""
    hyp_text = "Conformalized model selection with finite-sample coverage guarantees"
    hyp = Hypothesis(id="h1", text=hyp_text, rank=1)  # no claims
    prompt = "novel hypotheses for efficient multi-compartment model selection in diffusion MRI"
    run = _run([hyp], prompt=prompt)
    enriched_query = f"{hyp_text}. {prompt}"
    lit = FakeLiteratureClient({
        enriched_query: [PaperRecord(title=hyp_text, doi="10.1/x", year=2024, match_score=1.0)],
    })
    ctx = Context(config=RunConfig(), embeddings=HashingEmbedding(), literature=lit, judge=None)

    result = METRIC.score(run.outputs, run, ctx)
    assert result.evidence["per_hypothesis"][0]["nearest_neighbor_title"] == hyp_text


def test_different_hypotheses_sharing_a_prompt_get_different_queries():
    """The bug this whole design avoids: if every hypothesis's query were
    dominated by the same shared prompt, distinct hypotheses could all
    converge on whatever generic paper best matches the prompt's own
    wording, losing the ability to tell them apart. Claim text (different
    per hypothesis) prevents that; this pins the property directly.
    """
    prompt = "novel hypotheses for efficient multi-compartment model selection in diffusion MRI"
    hyp_a = Hypothesis(id="a", text="Hypothesis A", rank=1,
                       claims=[Claim(text="A involves gauge equivalence theory.")])
    hyp_b = Hypothesis(id="b", text="Hypothesis B", rank=2,
                       claims=[Claim(text="B involves conformal prediction sets.")])
    run = _run([hyp_a, hyp_b], prompt=prompt)

    paper_a = "A gauge equivalence paper"
    paper_b = "A conformal prediction paper"
    lit = FakeLiteratureClient({
        f"{hyp_a.text}. {hyp_a.claims[0].text}": [
            PaperRecord(title=paper_a, doi="10.1/a", year=2024, match_score=1.0)],
        f"{hyp_b.text}. {hyp_b.claims[0].text}": [
            PaperRecord(title=paper_b, doi="10.1/b", year=2024, match_score=1.0)],
    })
    ctx = Context(config=RunConfig(), embeddings=HashingEmbedding(), literature=lit, judge=None)

    result = METRIC.score(run.outputs, run, ctx)
    titles = {a["hypothesis_id"]: a["nearest_neighbor_title"] for a in result.evidence["per_hypothesis"]}
    assert titles == {"a": paper_a, "b": paper_b}


def test_embedding_similarity_still_uses_bare_hypothesis_text():
    """Query enrichment (claims/prompt) must NOT leak into the embedding-
    similarity computation -- that's meant to measure the hypothesis's own
    semantic distance from the nearest paper, and extending it with extra
    query text would change what's actually being compared."""
    hyp_text = "Kelch13 propeller domain mutations reduce artemisinin binding affinity"
    hyp = Hypothesis(id="h1", text=hyp_text, rank=1)
    prompt = "artemisinin resistance in Plasmodium knowlesi"
    run = _run([hyp], prompt=prompt)
    enriched_query = f"{hyp_text}. {prompt}"  # no claims -> falls back to the prompt
    lit = FakeLiteratureClient({
        enriched_query: [PaperRecord(title=hyp_text, doi="10.1/x", year=2020, match_score=1.0)],
    })
    ctx = Context(config=RunConfig(), embeddings=HashingEmbedding(), literature=lit, judge=None)

    result = METRIC.score(run.outputs, run, ctx)
    # Bare hyp_text embedded against an identical-text candidate -> ~0 distance,
    # exactly as in test_near_identical_literature_gives_low_distance. If the
    # prompt had leaked into the embedding step, this would no longer hold
    # since the candidate text doesn't include the prompt prefix.
    assert result.score < 0.1
