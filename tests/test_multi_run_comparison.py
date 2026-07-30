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
    EvaluationRun,
    Hypothesis,
    HypothesisSet,
    RunBundle,
    Tool,
)
from benchmarking_pipeline.services.embeddings import HashingEmbedding

REPRO = Reproducibility()
ROBUST = Robustness()


def _run(top_text: str, second_text: str) -> EvaluationRun:
    hyps = [
        Hypothesis(id="h1", text=top_text, rank=1),
        Hypothesis(id="h2", text=second_text, rank=2),
    ]
    return EvaluationRun(tool=Tool(name="t"), prompt="p", outputs=HypothesisSet(hypotheses=hyps))


def _ctx(comparison: str) -> Context:
    return Context(
        config=RunConfig(multi_run_comparison=comparison),
        embeddings=HashingEmbedding(),
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


def test_robustness_top_hypothesis_mode():
    base = _run(TOP_A, STABLE_SECOND)
    reworded = _run(TOP_B, STABLE_SECOND)
    reworded.metadata["perturbation"] = "reword"
    bundle = RunBundle(base=base, perturbations=[reworded])

    result = ROBUST.score(bundle, _ctx("top_hypothesis"))
    assert result.score < 0.3
    assert result.evidence["comparison_unit"] == "top_hypothesis"


def test_default_config_is_whole_set():
    assert RunConfig().multi_run_comparison == "whole_set"
