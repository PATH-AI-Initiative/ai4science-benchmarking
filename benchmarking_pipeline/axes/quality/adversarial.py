"""Adversarial testing.

Planned implementation: submit prompts seeded with planted traps (retracted
papers, known negatives framed as open questions, internally contradictory
premises) and score the proportion the tool correctly flags rather than
uncritically incorporating. Stub for now.
"""

from __future__ import annotations

from ...core.context import Context
from ...core.metric import Axis, MetricResult, SetMetric
from ...core.models import EvaluationRun, HypothesisSet
from ...core.registry import register


@register
class Adversarial(SetMetric):
    name = "adversarial"
    axis = Axis.QUALITY

    def score(self, outputs: HypothesisSet, run: EvaluationRun, ctx: Context) -> MetricResult:
        # TODO: score proportion of planted traps correctly flagged.
        return MetricResult.not_implemented(self.name, self.axis)
