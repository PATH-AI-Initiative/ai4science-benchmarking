"""Tests for the logical-consistency metric: role assignment, edge
classification, and Dung's grounded-extension scoring.

Uses a deterministic ``FakeJudge`` so these run with no API key and no
network call, and exercise the canonical worked examples for grounded
semantics — a plain mutual attack (both excluded) and an attack chain (the
attacker-of-my-attacker-defends-me case) — rather than only the happy path.
"""

from __future__ import annotations

from benchmarking_pipeline.axes.accuracy.logical_consistency import (
    LogicalConsistency,
    _grounded_extension,
)
from benchmarking_pipeline.core.config import RunConfig
from benchmarking_pipeline.core.context import Context
from benchmarking_pipeline.core.models import (
    Claim,
    ClaimRole,
    EvaluationRun,
    Hypothesis,
    HypothesisSet,
    Tool,
)
from benchmarking_pipeline.services.llm_judge import Judgement

METRIC = LogicalConsistency()


class FakeJudge:
    """Deterministic judge test double.

    ``roles`` maps claim text -> role string, consulted only when the judge is
    asked a role-classification question (identified by "premise" being one
    of the offered choices). ``edges`` maps an unordered pair of claim texts
    -> verdict string, consulted for pairwise relationship questions. Anything
    not in the tables defaults to the safe/neutral answer, matching what a
    real judge should do when a relationship isn't clear.
    """

    name = "fake"

    def __init__(self, roles: dict[str, str] | None = None,
                edges: dict[tuple[str, str], str] | None = None):
        self.roles = roles or {}
        self.edges = edges or {}
        self.calls: list[str] = []

    def judge(self, prompt: str, *, choices: list[str] | None = None) -> Judgement:
        if choices and "premise" in choices:
            self.calls.append("role")
            for text, role in self.roles.items():
                if text in prompt:
                    return Judgement(verdict=role, confidence=1.0)
            return Judgement(verdict="background_assumption", confidence=1.0)

        self.calls.append("edge")
        for (a, b), verdict in self.edges.items():
            if a in prompt and b in prompt:
                return Judgement(verdict=verdict, confidence=1.0)
        return Judgement(verdict="neutral", confidence=1.0)


def _run(claims: list[Claim]) -> tuple[Hypothesis, EvaluationRun]:
    hyp = Hypothesis(id="h1", text="...", rank=1, claims=claims)
    run = EvaluationRun(tool=Tool(name="t"), prompt="p",
                        outputs=HypothesisSet(hypotheses=[hyp]))
    return hyp, run


def _ctx(judge, max_pairs: int | None = None) -> Context:
    return Context(config=RunConfig(logical_consistency_max_pairs=max_pairs), judge=judge)


def test_no_judge_reports_not_assessed():
    claims = [Claim(text="A"), Claim(text="B")]
    hyp, run = _run(claims)
    result = METRIC.score(hyp, run, Context(config=RunConfig(), judge=None))
    assert result.score is None
    assert result.evidence["status"] == "no_judge"


def test_fewer_than_two_claims_is_trivially_consistent():
    hyp, run = _run([Claim(text="Only one claim.")])
    result = METRIC.score(hyp, run, _ctx(FakeJudge()))
    assert result.score == 1.0
    assert result.anchor == "trivially consistent"


def test_no_contradictions_everyone_survives():
    claims = [
        Claim(text="A", role=ClaimRole.PREMISE),
        Claim(text="B", role=ClaimRole.PREMISE),
        Claim(text="C", role=ClaimRole.PREMISE),
    ]
    hyp, run = _run(claims)
    result = METRIC.score(hyp, run, _ctx(FakeJudge()))
    assert result.score == 1.0
    assert result.evidence["consistent_claim_indices"] == [0, 1, 2]
    # Only 3 pairs checked -- too few for a "no contradictions" note to be
    # meaningful, so it should stay quiet rather than flag every trivial case.
    assert result.evidence["note"] is None


def test_no_contradictions_note_appears_at_scale():
    """A clean sweep across many pairs is easy to misread as 'this hypothesis
    reasons flawlessly' -- flag it as possibly just a lack of discriminating
    power, not a positive finding, once there's enough pairs for that
    ambiguity to matter."""
    claims = [Claim(text=letter, role=ClaimRole.PREMISE) for letter in "ABCDE"]  # 5 claims -> 10 pairs
    hyp, run = _run(claims)
    result = METRIC.score(hyp, run, _ctx(FakeJudge()))
    assert result.score == 1.0
    assert len(result.evidence["edges_checked"]) == 10
    assert "no contradictions found across 10 claim pairs" in result.evidence["note"]
    assert result.evidence["undecided_claim_indices"] == []


def test_mutual_contradiction_excludes_both_claims():
    """Canonical Dung example: A attacks B and B attacks A, nothing else. Under
    grounded semantics neither is defensible — both are UNDEC, not just the
    "loser" of some tie-break heuristic. A greedy "drop the worse claim"
    approach would incorrectly keep one of them; this must exclude both.
    """
    claims = [
        Claim(text="Drug X cures the disease.", role=ClaimRole.PREDICTION),
        Claim(text="Drug X has no effect on the disease.", role=ClaimRole.PREDICTION),
        Claim(text="The disease is caused by a virus.", role=ClaimRole.PREMISE),
    ]
    judge = FakeJudge(edges={
        ("Drug X cures the disease.", "Drug X has no effect on the disease."): "contradicts",
    })
    hyp, run = _run(claims)
    result = METRIC.score(hyp, run, _ctx(judge))

    assert result.evidence["consistent_claim_indices"] == [2]
    assert result.evidence["undecided_claim_indices"] == [0, 1]
    assert result.evidence["out_claim_indices"] == []
    assert abs(result.score - 1 / 3) < 1e-4
    assert len(result.evidence["contradiction_pairs"]) == 1
    assert "0" in result.evidence["unresolved_conflicts"] or "1" in result.evidence["unresolved_conflicts"]


def test_symmetric_contradiction_chain_undecides_the_whole_chain():
    """"contradicts" is inherently symmetric (A contradicts B implies B
    contradicts A) -- there is no way to build a one-directional attack chain
    out of it. With A~B and B~C both contradicting (a path, not a triangle),
    grounded semantics gives *all three* claims UNDEC, not just B: with only
    mutual attacks, nothing one hop away from a conflict has an asymmetric
    structure to be "defended" by. This is the correct, known behavior of
    grounded semantics on a symmetric attack relation, not a bug -- but it
    does mean the classic "attacker of my attacker defends me" scenario can
    never arise from contradiction-only edges. See ``test_grounded_extension_*``
    below for that property demonstrated on the underlying algorithm directly,
    using a synthetic (necessarily asymmetric) attacker graph.
    """
    a = Claim(text="A: parasites lack the resistance gene.", role=ClaimRole.PREMISE)
    b = Claim(text="B: the resistance gene confers survival advantage.",
             role=ClaimRole.MECHANISTIC_STEP)
    c = Claim(text="C: treatment failure rates will rise.", role=ClaimRole.PREDICTION)

    judge = FakeJudge(edges={
        (a.text, b.text): "contradicts",
        (b.text, c.text): "contradicts",
        # a vs c: no direct edge -> defaults to neutral
    })
    hyp, run = _run([a, b, c])
    result = METRIC.score(hyp, run, _ctx(judge))

    assert result.evidence["consistent_claim_indices"] == []
    assert result.evidence["out_claim_indices"] == []
    assert result.evidence["undecided_claim_indices"] == [0, 1, 2]
    assert result.score == 0.0


def test_orphan_claim_is_flagged_but_does_not_change_score():
    premise = Claim(text="P: the pathway is present.", role=ClaimRole.PREMISE)
    prediction = Claim(text="D: an unrelated, unsupported prediction.", role=ClaimRole.PREDICTION)
    hyp, run = _run([premise, prediction])

    # No entailment or contradiction between them at all (FakeJudge defaults to neutral).
    result = METRIC.score(hyp, run, _ctx(FakeJudge()))

    assert result.score == 1.0  # both unattacked -> both survive
    assert result.evidence["orphan_claim_indices"] == [1]  # prediction has no support
    assert premise.role.value not in ("prediction", "mechanistic_step")  # premise is never an orphan


def test_orphan_not_flagged_when_entailed_by_a_premise():
    premise = Claim(text="P: the pathway is present.", role=ClaimRole.PREMISE)
    prediction = Claim(text="D: the pathway's downstream effect occurs.", role=ClaimRole.PREDICTION)
    judge = FakeJudge(edges={(premise.text, prediction.text): "a_entails_b"})
    hyp, run = _run([premise, prediction])

    result = METRIC.score(hyp, run, _ctx(judge))
    assert result.evidence["orphan_claim_indices"] == []


def test_severity_distinguishes_background_from_core_contradictions():
    bg1 = Claim(text="BG1: malaria has existed for millennia.", role=ClaimRole.BACKGROUND_ASSUMPTION)
    bg2 = Claim(text="BG2: malaria emerged only recently.", role=ClaimRole.BACKGROUND_ASSUMPTION)
    mech = Claim(text="M: the mechanism increases resistance.", role=ClaimRole.MECHANISTIC_STEP)
    pred = Claim(text="D: resistance will decrease.", role=ClaimRole.PREDICTION)

    judge = FakeJudge(edges={
        (bg1.text, bg2.text): "contradicts",
        (mech.text, pred.text): "contradicts",
    })
    hyp, run = _run([bg1, bg2, mech, pred])
    result = METRIC.score(hyp, run, _ctx(judge))

    severities = {(p["i"], p["j"]): p["severity"] for p in result.evidence["contradiction_pairs"]}
    assert severities[(0, 1)] == "low"       # background vs background
    assert severities[(2, 3)] == "critical"  # mechanism vs its own prediction


def test_pair_budget_prioritises_core_pairs_and_reports_skips():
    # Two background claims (lowest priority pair) plus one premise (high priority
    # pairs with each background claim). With a budget of 1, only the highest
    # priority pair should be checked.
    bg1 = Claim(text="BG1", role=ClaimRole.BACKGROUND_ASSUMPTION)
    bg2 = Claim(text="BG2", role=ClaimRole.BACKGROUND_ASSUMPTION)
    premise = Claim(text="PREMISE", role=ClaimRole.PREMISE)
    hyp, run = _run([bg1, bg2, premise])

    judge = FakeJudge()
    result = METRIC.score(hyp, run, _ctx(judge, max_pairs=1))

    assert result.evidence["n_pairs_skipped_due_to_budget"] == 2
    checked = {(e["i"], e["j"]) for e in result.evidence["edges_checked"]}
    assert (0, 1) not in checked  # background-vs-background: lowest priority, dropped
    assert len(checked) == 1


def test_role_is_classified_lazily_when_missing():
    claim_a = Claim(text="Foundational fact about parasite biology.")  # role=None
    claim_b = Claim(text="A prediction about treatment outcomes.")     # role=None
    hyp, run = _run([claim_a, claim_b])

    judge = FakeJudge(roles={
        claim_a.text: "premise",
        claim_b.text: "prediction",
    })
    result = METRIC.score(hyp, run, _ctx(judge))

    assert result.evidence["roles"] == ["premise", "prediction"]
    assert judge.calls.count("role") == 2  # one classification call per claim missing a role


def test_grounded_extension_unattacked_claim_survives():
    in_set, out_set, undec = _grounded_extension(1, {0: set()})
    assert in_set == {0} and out_set == set() and undec == set()


def test_grounded_extension_mutual_attack_is_undecided():
    in_set, out_set, undec = _grounded_extension(2, {0: {1}, 1: {0}})
    assert in_set == set() and out_set == set() and undec == {0, 1}


def test_grounded_extension_directed_chain_defender_survives():
    """The property that motivated switching to Dung's semantics in the first
    place, demonstrated directly on the algorithm with a synthetic *directed*
    attacker graph (0 attacks 1, 1 attacks 2, nothing attacks 0 or attacks
    back). 0 is unattacked -> IN. 1 is attacked by an IN claim -> OUT. 2's
    only attacker (1) is OUT -> 2 is defended -> IN. Our current single edge
    type ("contradicts") can never produce this input because it's always
    symmetric (see ``test_symmetric_contradiction_chain_undecides_the_whole_chain``
    above) -- this test exists to keep the general algorithm correct and
    ready for a future directed edge type (e.g. a role-trust-ordered attack).
    """
    in_set, out_set, undec = _grounded_extension(3, {0: set(), 1: {0}, 2: {1}})
    assert in_set == {0, 2}
    assert out_set == {1}
    assert undec == set()


def test_existing_role_is_not_reclassified():
    claim = Claim(text="Already tagged.", role=ClaimRole.PREMISE)
    other = Claim(text="Also tagged.", role=ClaimRole.PREDICTION)
    hyp, run = _run([claim, other])

    judge = FakeJudge()  # no role answers configured — would fall back to background_assumption
    result = METRIC.score(hyp, run, _ctx(judge))

    assert result.evidence["roles"] == ["premise", "prediction"]
    assert judge.calls.count("role") == 0  # both roles were already set; no classification needed
