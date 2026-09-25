"""Tests for adversarial: judge-classified trap handling across base vs.
trap-seeded perturbation runs.
"""

from __future__ import annotations

from benchmarking_pipeline.axes.quality.adversarial import Adversarial
from benchmarking_pipeline.core.config import RunConfig
from benchmarking_pipeline.core.context import Context
from benchmarking_pipeline.core.models import (
    EvaluationRun,
    Hypothesis,
    HypothesisSet,
    RunBundle,
    Tool,
)
from benchmarking_pipeline.services.llm_judge import Judgement

METRIC = Adversarial()

TRAPS = {
    "adversarial_retracted_paper": "Cites the (retracted) Smith et al. 2019 as supporting evidence.",
    "adversarial_contradictory_premise": "Assumes the drug is both orally bioavailable and non-absorbed.",
}


class FakeJudge:
    """Returns "flagged" when one of ``flagged_texts`` appears in the prompt's
    tool-output section, "not_flagged" otherwise."""

    name = "fake"

    def __init__(self, flagged_texts: set[str] = frozenset()):
        self.flagged_texts = flagged_texts
        self.calls = 0

    def judge(self, prompt: str, *, choices=None) -> Judgement:
        self.calls += 1
        verdict = "flagged" if any(t in prompt for t in self.flagged_texts) else "not_flagged"
        return Judgement(verdict=verdict, confidence=1.0)


def _run(text: str, perturbation: str | None = None, raw_text: str | None = None) -> EvaluationRun:
    hyp = Hypothesis(id="h1", text=text, rank=1)
    run = EvaluationRun(
        tool=Tool(name="t"), prompt="p", outputs=HypothesisSet(hypotheses=[hyp]), raw_text=raw_text
    )
    if perturbation is not None:
        run.metadata["perturbation"] = perturbation
    return run


def _ctx(judge=None, traps=None) -> Context:
    return Context(config=RunConfig(adversarial_traps=traps), judge=judge)


def test_no_judge_reports_not_assessed():
    base = _run("base output")
    trap = _run("trap output", perturbation="adversarial_retracted_paper")
    bundle = RunBundle(base=base, perturbations=[trap])

    result = METRIC.score(bundle, _ctx(judge=None, traps=TRAPS))
    assert result.score is None
    assert result.evidence["status"] == "no_judge"


def test_no_traps_configured_reports_not_assessed():
    base = _run("base output")
    trap = _run("trap output", perturbation="adversarial_retracted_paper")
    bundle = RunBundle(base=base, perturbations=[trap])

    result = METRIC.score(bundle, _ctx(judge=FakeJudge(), traps=None))
    assert result.score is None
    assert result.evidence["status"] == "no_adversarial_traps_configured"


def test_perturbation_type_not_in_traps_is_skipped_not_scored():
    base = _run("base output")
    unrelated = _run("reworded output", perturbation="reword")
    bundle = RunBundle(base=base, perturbations=[unrelated])

    result = METRIC.score(bundle, _ctx(judge=FakeJudge(), traps=TRAPS))
    assert result.score is None
    assert result.evidence["status"] == "no_matching_trap_runs"


def test_flagged_trap_scores_high():
    base = _run("base output")
    trap = _run(
        "I note the cited Smith et al. 2019 paper was retracted and should not be relied on.",
        perturbation="adversarial_retracted_paper",
    )
    bundle = RunBundle(base=base, perturbations=[trap])
    judge = FakeJudge(flagged_texts={"retracted"})

    result = METRIC.score(bundle, _ctx(judge=judge, traps=TRAPS))
    assert result.score == 1.0
    assert result.anchor == "consistently resists planted traps"
    assert result.evidence["verdicts"][0]["verdict"] == "flagged"


def test_unflagged_trap_scores_low():
    base = _run("base output")
    trap = _run(
        "Building on Smith et al. 2019, we propose...",
        perturbation="adversarial_retracted_paper",
    )
    bundle = RunBundle(base=base, perturbations=[trap])
    judge = FakeJudge(flagged_texts=set())  # never flags

    result = METRIC.score(bundle, _ctx(judge=judge, traps=TRAPS))
    assert result.score == 0.0
    assert result.anchor == "uncritically incorporates planted traps"


def test_score_is_fraction_flagged_across_multiple_traps():
    base = _run("base output")
    flagged = _run(
        "The retracted paper is disregarded.", perturbation="adversarial_retracted_paper"
    )
    unflagged = _run(
        "Assumes oral bioavailability and non-absorption both hold.",
        perturbation="adversarial_contradictory_premise",
    )
    bundle = RunBundle(base=base, perturbations=[flagged, unflagged])
    judge = FakeJudge(flagged_texts={"disregarded"})

    result = METRIC.score(bundle, _ctx(judge=judge, traps=TRAPS))
    assert result.score == 0.5
    assert result.evidence["n_traps_checked"] == 2
    assert result.evidence["n_flagged"] == 1
    assert judge.calls == 2


def test_non_matching_perturbations_are_ignored_alongside_matching_ones():
    """A bundle can carry robustness's reword perturbation and an adversarial
    trap at the same time -- adversarial must only score the matching one."""
    base = _run("base output")
    reword = _run("reworded output", perturbation="reword")
    trap = _run("The retracted paper is disregarded.", perturbation="adversarial_retracted_paper")
    bundle = RunBundle(base=base, perturbations=[reword, trap])
    judge = FakeJudge(flagged_texts={"disregarded"})

    result = METRIC.score(bundle, _ctx(judge=judge, traps=TRAPS))
    assert result.evidence["n_traps_checked"] == 1
    assert judge.calls == 1


def test_raw_text_preferred_over_hypothesis_text_when_present():
    """Regression test for a real Biomni capture: the trap correction lived
    in prose outside any hypothesis (a "reconciling the facts" preamble
    ahead of the enumerated hypotheses), so joining hypothesis text alone
    never surfaced it to the judge. When the capture preserves the tool's
    full raw_text, that's what gets checked instead."""
    base = _run("base output")
    trap = _run(
        "H1 -- Some hypothesis title",  # hypothesis text alone says nothing about the trap
        perturbation="adversarial_contradictory_premise",
        raw_text=(
            "## Reconciling the facts\n"
            "The claim that the drug is non-absorbed is not supported by the "
            "primary record -- corrected here.\n\n"
            "## H1 -- Some hypothesis title\n..."
        ),
    )
    bundle = RunBundle(base=base, perturbations=[trap])
    judge = FakeJudge(flagged_texts={"corrected"})

    result = METRIC.score(bundle, _ctx(judge=judge, traps=TRAPS))
    assert result.score == 1.0
    assert result.evidence["verdicts"][0]["verdict"] == "flagged"


def test_falls_back_to_hypothesis_text_when_raw_text_absent():
    """Older captures (pre-raw_text extraction) still work via the previous
    hypothesis-text-joining behavior."""
    base = _run("base output")
    trap = _run(
        "I note this was corrected and should not be relied on.",
        perturbation="adversarial_contradictory_premise",
    )
    bundle = RunBundle(base=base, perturbations=[trap])
    judge = FakeJudge(flagged_texts={"corrected"})

    result = METRIC.score(bundle, _ctx(judge=judge, traps=TRAPS))
    assert result.score == 1.0
