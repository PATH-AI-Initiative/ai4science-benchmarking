"""Tests for the ``multi_run_comparison`` config option (whole_set vs.
top_hypothesis) that reproducibility/robustness use to decide what gets
compared across runs.

The motivating case: a tool whose rank-1 idea flips to something unrelated
between runs, while a lower-ranked idea happens to stay identical. Mean-
pooling the whole set (the default) dilutes that instability with the stable
lower-ranked idea; comparing only the rank-1 hypothesis surfaces it directly.
"""

from __future__ import annotations

from benchmarking_pipeline.axes.quality.reproducibility import Reproducibility
from benchmarking_pipeline.axes.quality.robustness import Robustness
from benchmarking_pipeline.core.config import RunConfig
from benchmarking_pipeline.core.context import Context
from benchmarking_pipeline.core.models import (
    Claim,
    EvaluationRun,
    Hypothesis,
    HypothesisSet,
    RunBundle,
    Tool,
)
from benchmarking_pipeline.services.embeddings import HashingEmbedding
from benchmarking_pipeline.services.llm_judge import Judgement

REPRO = Reproducibility()
ROBUST = Robustness()


class FakeJudge:
    """Returns "same_mechanism" when one of ``same_mechanism_texts`` appears
    *twice* in the prompt (i.e. both compared hypotheses are that exact
    text), "different_mechanism" otherwise. Requiring two occurrences (not
    just one) matters: the prompt always contains both compared texts, so a
    marker matching only one side must not count as a match -- otherwise a
    mixed pair (one matching text, one not) would incorrectly read as the
    same mechanism just because one side happened to contain the marker."""

    name = "fake"

    def __init__(self, same_mechanism_texts: set[str] = frozenset()):
        self.same_mechanism_texts = same_mechanism_texts
        self.calls = 0

    def judge(self, prompt: str, *, choices=None) -> Judgement:
        self.calls += 1
        verdict = "different_mechanism"
        for text in self.same_mechanism_texts:
            if prompt.count(text) >= 2:
                verdict = "same_mechanism"
                break
        return Judgement(verdict=verdict, confidence=1.0)


def _run(top_text: str, second_text: str) -> EvaluationRun:
    hyps = [
        Hypothesis(id="h1", text=top_text, rank=1),
        Hypothesis(id="h2", text=second_text, rank=2),
    ]
    return EvaluationRun(tool=Tool(name="t"), prompt="p", outputs=HypothesisSet(hypotheses=hyps))


def _run_with_hypotheses(hyps: list[Hypothesis]) -> EvaluationRun:
    return EvaluationRun(tool=Tool(name="t"), prompt="p", outputs=HypothesisSet(hypotheses=hyps))


def _ctx(comparison: str, judge=None) -> Context:
    return Context(
        config=RunConfig(multi_run_comparison=comparison),
        embeddings=HashingEmbedding(),
        judge=judge,
    )


TOP_A = "Kelch13 propeller domain mutations reduce artemisinin binding affinity"
TOP_B = "Host reticulocyte membrane composition alters merozoite invasion efficiency"
STABLE_SECOND = "Lipid metabolism pathways modulate ring stage drug tolerance"


def test_whole_set_dilutes_top_hypothesis_instability():
    base = _run(TOP_A, STABLE_SECOND)
    repeat = _run(TOP_B, STABLE_SECOND)  # rank-1 completely different; rank-2 identical
    bundle = RunBundle(base=base, repeats=[repeat])

    result = REPRO.score(bundle, _ctx("whole_set"))
    # The identical rank-2 hypothesis pulls the mean-pooled similarity up,
    # masking how different the actual top idea was between runs.
    assert result.score > 0.5


def test_top_hypothesis_surfaces_the_same_instability():
    base = _run(TOP_A, STABLE_SECOND)
    repeat = _run(TOP_B, STABLE_SECOND)
    bundle = RunBundle(base=base, repeats=[repeat])

    result = REPRO.score(bundle, _ctx("top_hypothesis"))
    # Only rank-1 is compared: two unrelated ideas -> low similarity.
    assert result.score < 0.3
    assert result.evidence["comparison_unit"] == "top_hypothesis"


def test_top_hypothesis_identical_top_idea_scores_high_regardless_of_others():
    base = _run(TOP_A, STABLE_SECOND)
    # rank-1 identical, rank-2 completely different this time
    repeat = _run(TOP_A, "Completely unrelated tangent about vaccine cold-chain logistics")
    bundle = RunBundle(base=base, repeats=[repeat])

    result = REPRO.score(bundle, _ctx("top_hypothesis"))
    assert result.score > 0.9  # rank-1 text is literally identical across runs
    assert result.evidence["score_basis"] == "mean_pairwise_similarity"  # no judge -> embedding fallback


def test_robustness_top_hypothesis_mode():
    base = _run(TOP_A, STABLE_SECOND)
    reworded = _run(TOP_B, STABLE_SECOND)
    reworded.metadata["perturbation"] = "reword"
    bundle = RunBundle(base=base, perturbations=[reworded])

    result = ROBUST.score(bundle, _ctx("top_hypothesis"))
    assert result.score < 0.3
    assert result.evidence["comparison_unit"] == "top_hypothesis"
    assert result.evidence["score_basis"] == "reword_stability"  # no judge -> embedding fallback


def test_default_config_is_whole_set():
    assert RunConfig().multi_run_comparison == "whole_set"


def test_mechanism_match_disagrees_with_embedding_similarity_when_it_should():
    """Regression test: a real 4-run test found two hypotheses that read as
    highly similar by embedding (same broad research area) but that a judge
    correctly identifies as different core proposals. Measured directly,
    embedding similarity barely discriminates "same narrow research question"
    from "same specific proposal" -- so when a judge is available,
    mechanism_match_rate is the *scored* signal, not embedding similarity;
    embedding is still reported in full as context.
    """
    base = _run(TOP_A, STABLE_SECOND)
    repeat = _run(TOP_A, STABLE_SECOND)  # identical text -> embedding similarity is ~1.0
    bundle = RunBundle(base=base, repeats=[repeat])
    judge = FakeJudge(same_mechanism_texts=set())  # judge insists they're different anyway

    result = REPRO.score(bundle, _ctx("top_hypothesis", judge=judge))

    assert result.score == 0.0  # scored on mechanism_match_rate, not embedding
    assert result.evidence["score_basis"] == "mechanism_match_rate"
    assert result.evidence["mean_pairwise_similarity"] > 0.9  # embedding: "basically identical"
    assert result.evidence["mechanism_match_rate"] == 0.0  # judge disagrees
    assert result.evidence["mechanism_matches"][0]["credit"] == 0.0
    # The anchor must surface that embedding still reads high, not hide it.
    assert "mechanism-verified" in result.anchor
    assert "shared topic" in result.anchor


def test_mechanism_match_rate_over_all_pairs_not_just_vs_base():
    """3 runs -> 3 pairs. Only the base/repeat1 pair is judged the same
    mechanism; the rate must reflect all 3 pairs, not just comparisons
    against base. Each run has 2 hypotheses (rank 1 and 2), so each pair
    costs 3 judge calls (A0-vs-B0, A0-vs-B1, B0-vs-A1 -- A0-vs-B0 is reused
    for both directions), not 1 -- 9 calls total across 3 pairs."""
    base = _run(TOP_A, STABLE_SECOND)
    repeat1 = _run(TOP_A, STABLE_SECOND)  # same mechanism as base
    repeat2 = _run(TOP_B, STABLE_SECOND)  # different mechanism from both
    bundle = RunBundle(base=base, repeats=[repeat1, repeat2])
    judge = FakeJudge(same_mechanism_texts={TOP_A})

    result = REPRO.score(bundle, _ctx("top_hypothesis", judge=judge))

    # Pairs: (base,repeat1)=same [both TOP_A, full credit], (base,repeat2)=different,
    # (repeat1,repeat2)=different -- STABLE_SECOND never matches, so no partial credit.
    assert result.evidence["mechanism_match_rate"] == round(1 / 3, 4)
    assert judge.calls == 9


def test_mechanism_match_skipped_in_whole_set_mode():
    """The mechanism check only makes sense for a single hypothesis pair --
    whole_set mode compares mean-pooled sets, so there's no single pair of
    texts to hand the judge."""
    base = _run(TOP_A, STABLE_SECOND)
    repeat = _run(TOP_B, STABLE_SECOND)
    bundle = RunBundle(base=base, repeats=[repeat])
    judge = FakeJudge(same_mechanism_texts={TOP_A})

    result = REPRO.score(bundle, _ctx("whole_set", judge=judge))

    assert result.evidence["mechanism_match_rate"] is None
    assert judge.calls == 0


def test_mechanism_match_skipped_without_a_judge():
    base = _run(TOP_A, STABLE_SECOND)
    repeat = _run(TOP_A, STABLE_SECOND)
    bundle = RunBundle(base=base, repeats=[repeat])

    result = REPRO.score(bundle, _ctx("top_hypothesis", judge=None))

    assert result.evidence["mechanism_match_rate"] is None
    assert result.evidence["mechanism_matches"] is None


def test_robustness_mechanism_match_by_perturbation():
    base = _run(TOP_A, STABLE_SECOND)
    reworded = _run(TOP_A, STABLE_SECOND)  # embedding-identical to base
    reworded.metadata["perturbation"] = "reword"
    bundle = RunBundle(base=base, perturbations=[reworded])
    judge = FakeJudge(same_mechanism_texts=set())  # judge says different anyway

    result = ROBUST.score(bundle, _ctx("top_hypothesis", judge=judge))

    assert result.score == 0.0  # scored on reword_mechanism_match_rate, not embedding
    assert result.evidence["score_basis"] == "reword_mechanism_match_rate"
    assert result.evidence["reword_stability"] > 0.9  # embedding: "basically identical"
    assert result.evidence["reword_mechanism_match_rate"] == 0.0  # judge disagrees
    assert result.evidence["mechanism_match_by_perturbation"]["reword"][0]["credit"] == 0.0
    assert "mechanism-verified" in result.anchor
    assert "shared topic" in result.anchor


def test_mechanism_match_uses_claims_not_just_the_bare_title():
    """Regression test for a real Biomni capture: extraction left each
    hypothesis's ``text`` as just a short title (e.g. "Secretory Pathway +
    Blood Schizonticide" vs. "A Non-Artemisinin Combination for Pk"), with the
    two runs actually proposing the same intervention -- only visible in their
    claims. Before ``hypothesis_text`` enrichment, mechanism-match only ever
    saw the bare, differently-worded titles and had no way to see the shared
    claim underneath."""
    shared_claim_text = "Both target the Ganaplacide-Lumefantrine drug combination."
    hyp_a = Hypothesis(
        id="h1", text="Secretory Pathway + Blood Schizonticide", rank=1,
        claims=[Claim(text=shared_claim_text)],
    )
    hyp_b = Hypothesis(
        id="h1", text="A Non-Artemisinin Combination for Pk", rank=1,
        claims=[Claim(text=shared_claim_text)],
    )
    base = _run_with_hypotheses([hyp_a])
    repeat = _run_with_hypotheses([hyp_b])
    bundle = RunBundle(base=base, repeats=[repeat])
    judge = FakeJudge(same_mechanism_texts={shared_claim_text})

    result = REPRO.score(bundle, _ctx("top_hypothesis", judge=judge))

    assert result.evidence["mechanism_match_rate"] == 1.0
    assert result.evidence["mechanism_matches"][0]["credit"] == 1.0


def test_idea_demoted_not_vanished_gets_partial_credit():
    """The actual new capability: base's rank-1 idea (X) reappears at rank 2
    in the other run instead of rank 1 -- a strict rank-1-vs-rank-1 check
    would score this as a flat miss (0.0), same as if X had vanished
    entirely. It should instead get partial credit (found at rank 2 -> 0.5),
    since "demoted" and "gone" are genuinely different outcomes."""
    X, Y, Z, W, V = "Drug X combo", "Drug Y combo", "Drug Z combo", "Drug W combo", "Drug V combo"
    base = _run_with_hypotheses([
        Hypothesis(id="b1", text=X, rank=1),
        Hypothesis(id="b2", text=Y, rank=2),
        Hypothesis(id="b3", text=Z, rank=3),
    ])
    other = _run_with_hypotheses([
        Hypothesis(id="o1", text=W, rank=1),  # base's rank-1 (X) is NOT here
        Hypothesis(id="o2", text=X, rank=2),  # ...it's here instead, demoted to rank 2
        Hypothesis(id="o3", text=V, rank=3),
    ])
    bundle = RunBundle(base=base, repeats=[other])
    judge = FakeJudge(same_mechanism_texts={X})  # only X-vs-X reads as same_mechanism

    result = REPRO.score(bundle, _ctx("top_hypothesis", judge=judge))

    match = result.evidence["mechanism_matches"][0]
    assert match["credit_a_to_b"] == 0.5  # base's X found at rank 2 in other -> half credit
    assert match["credit_b_to_a"] == 0.0  # other's rank-1 (W) not found anywhere in base
    assert match["credit"] == 0.25  # averaged
    assert result.evidence["mechanism_match_rate"] == 0.25
    # And this is strictly more information than a rank-1-only check would
    # give: W-vs-X (the old comparison) is "different_mechanism" on its own,
    # which would have scored this pair identically to X vanishing outright.
    assert match["a_rank1_vs_b"][0] == "different_mechanism"  # X vs W: not a match
    assert match["a_rank1_vs_b"][1] == "same_mechanism"  # X vs X: the match that saves it
