"""Safety — the hard gate.

Planned: expert biosafety review of outputs against DURC criteria (dual-use
research of concern), supplemented by a red-team prompt set scored on
detection rate. A tool that raises safety concerns is disqualified regardless
of performance elsewhere.

The gate is enforced in ``core.scoring``; this metric signals gate membership
via ``evidence["gate"] = True`` so the scorer treats a low score as
disqualifying, not merely weighted. Stub for now.
"""

from __future__ import annotations

from ...core.context import Context
from ...core.metric import Axis, MetricResult, SetMetric
from ...core.models import EvaluationRun, HypothesisSet
from ...core.registry import register


@register
class Safety(SetMetric):
    name = "safety"
    axis = Axis.QUALITY
    is_gate = True

    def score(self, outputs: HypothesisSet, run: EvaluationRun, ctx: Context) -> MetricResult:
        # TODO: expert DURC/biosafety review + red-team detection rate.
        # When implemented, return a real score in [0, 1] with evidence["gate"] = True.
        return MetricResult(
            metric=self.name, axis=self.axis, score=None, evidence={"gate": True,
            "status": "not_implemented"},
        )
