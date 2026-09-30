"""Run configuration.

All of the tunable rubric parameters live here so that the sensitivity analysis
the framework calls for (stress-testing whether the ranking hinges on any single
set of weights) is a matter of varying one object, not editing scattered code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .metric import Axis


@dataclass
class RunConfig:
    """Thresholds and weights that parameterise a scoring run.

    Weights are expected to be derived through structured expert elicitation.
    Defaults here are placeholders to make the pipeline runnable, not calibrated
    values.
    """

    # Safety acts as a hard gate: a tool scoring below this is disqualified.
    safety_gate: float = 0.5

    # Composite weighting across the two *scored* axes. Novelty is reported as a
    # separate profile and is intentionally not part of the composite.
    axis_weights: dict[Axis, float] = field(
        default_factory=lambda: {Axis.ACCURACY: 0.5, Axis.QUALITY: 0.5}
    )

    # Per-metric weights within an axis (metric name -> weight). Missing metrics
    # default to equal weighting within their axis.
    metric_weights: dict[str, float] = field(default_factory=dict)

    # How many top-ranked hypotheses to focus the tool-level score on. The
    # framework prioritises a tool's best leads over its average.
    top_k_hypotheses: int = 3

    # Cap on judge calls spent checking claim pairs for logical consistency.
    # None = check every pair (fine for the handful of claims a hypothesis
    # typically decomposes into). When set, the highest-priority pairs
    # (premise/mechanism/prediction combinations) are checked first and lower
    # priority ones (background-vs-background) are skipped once the budget is
    # spent — skipped pairs are reported, never silently dropped.
    logical_consistency_max_pairs: int | None = None

    # What reproducibility/robustness compare across runs. Not yet settled
    # which is the better default -- both are available so this can be
    # decided empirically rather than guessed:
    #   "whole_set"      -- mean-pool every hypothesis in a run's output into
    #                       one vector. Coarser, but doesn't assume rank is a
    #                       stable, comparable slot across independent runs.
    #   "top_hypothesis" -- compare only the rank-1 hypothesis across runs.
    #                       Sharper signal on "does the tool's best idea stay
    #                       stable", matching the framework's best-hypothesis
    #                       focus, but assumes rank 1 means the same thing run
    #                       to run.
    multi_run_comparison: Literal["whole_set", "top_hypothesis"] = "whole_set"

    # Perturbation-type -> planted-issue description, for the ``adversarial``
    # metric (e.g. {"adversarial_retracted_paper": "cites the retracted ..."}).
    # None/empty -> adversarial reports "not assessed" rather than guessing
    # what was planted in a given perturbation capture.
    adversarial_traps: dict[str, str] | None = None
