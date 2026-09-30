"""The metric interface — the unit of extension for the whole suite.

Every scored or descriptive check is a :class:`Metric`. New metrics can be
developed, tested, and contributed independently; the scoring layer composes
them without knowing their internals.

Two granularities, two base classes:

* :class:`HypothesisMetric` scores one hypothesis (citation accuracy,
  logical consistency, tractability).
* :class:`SetMetric` scores the output set as a whole (diversity,
  reproducibility).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .context import Context
    from .models import EvaluationRun, Hypothesis, HypothesisSet, RunBundle


class Axis(str, Enum):
    """The four evaluation axes, ordered as the framework applies them."""

    DETAILS = "details_features"  # Tier 1 — descriptive, not scored
    ACCURACY = "accuracy"  # Tier 2 — scored, acts as a floor
    QUALITY = "quality"  # Tier 3 — scored, Safety is a hard gate
    NOVELTY = "novelty"  # Tier 4 — reported as a profile, not scored


@dataclass
class MetricResult:
    """The output of running one metric.

    ``score`` is normalized to ``[0, 1]`` on a common scale, or ``None`` for
    descriptive metrics (Tier 1) and metrics that could not run. ``evidence``
    holds the raw findings for auditability; ``anchor`` names the qualitative
    level the score corresponds to (anchors are defined before scoring begins).
    """

    metric: str
    axis: Axis
    score: float | None = None
    anchor: str | None = None
    evidence: dict = field(default_factory=dict)
    error: str | None = None

    @classmethod
    def not_implemented(cls, metric: str, axis: Axis) -> MetricResult:
        return cls(metric=metric, axis=axis, score=None, evidence={"status": "not_implemented"})


class Metric(ABC):
    """Base class for all metrics. Subclasses set the class attributes below."""

    name: str
    axis: Axis
    # Accuracy metrics that gate further scoring for a hypothesis.
    is_floor: bool = False
    # Safety metrics that can disqualify a tool outright regardless of other scores.
    is_gate: bool = False


class HypothesisMetric(Metric):
    """Scores a single hypothesis."""

    @abstractmethod
    def score(self, hypothesis: "Hypothesis", run: "EvaluationRun", ctx: "Context") -> MetricResult:
        ...


class SetMetric(Metric):
    """Scores a whole hypothesis set (set-level properties)."""

    @abstractmethod
    def score(self, outputs: "HypothesisSet", run: "EvaluationRun", ctx: "Context") -> MetricResult:
        ...


class MultiRunMetric(Metric):
    """Scores a property of a *set of runs* — reproducibility (identical reruns)
    and robustness (perturbed reruns). Only evaluated when the pipeline is given
    a :class:`~benchmarking_pipeline.core.models.RunBundle`.
    """

    @abstractmethod
    def score(self, bundle: "RunBundle", ctx: "Context") -> MetricResult:
        ...
