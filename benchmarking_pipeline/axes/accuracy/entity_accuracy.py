"""Entity accuracy: are named biological entities real and correctly related?

Planned implementation: extract entities per claim, resolve each against curated
databases (UniProt, KEGG) via ``ctx.biodb``, and score the fraction that resolve
and are used consistently. Stub for now.
"""

from __future__ import annotations

from ...core.context import Context
from ...core.metric import Axis, HypothesisMetric, MetricResult
from ...core.models import EvaluationRun, Hypothesis
from ...core.registry import register


@register
class EntityAccuracy(HypothesisMetric):
    name = "entity_accuracy"
    axis = Axis.ACCURACY
    is_floor = True

    def score(self, hypothesis: Hypothesis, run: EvaluationRun, ctx: Context) -> MetricResult:
        # TODO: extract entities, resolve via ctx.biodb, score resolution + relational correctness.
        return MetricResult.not_implemented(self.name, self.axis)
