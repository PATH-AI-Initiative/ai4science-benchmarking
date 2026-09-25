"""Downstream tractability.

Planned: a structured SME review scoring each hypothesis on specificity,
feasibility, and falsifiability -- through an LMIC lens (does it need
equipment only well-resourced labs have?) and an ethics check (human
subjects, animal welfare, biosafety). Rubric anchors are agreed before
scoring. ``ctx.judge`` may assist, but this metric is human-in-the-loop by
design. Stub for now.
"""

from __future__ import annotations

from ...core.context import Context
from ...core.metric import Axis, HypothesisMetric, MetricResult
from ...core.models import EvaluationRun, Hypothesis
from ...core.registry import register


@register
class Tractability(HypothesisMetric):
    name = "tractability"
    axis = Axis.QUALITY

    def score(self, hypothesis: Hypothesis, run: EvaluationRun, ctx: Context) -> MetricResult:
        # TODO: SME-scored specificity / feasibility / falsifiability with LMIC + ethics lens.
        return MetricResult.not_implemented(self.name, self.axis)
