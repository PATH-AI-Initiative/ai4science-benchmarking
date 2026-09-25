"""Tier 2 — Accuracy. Is the factual content of the output correct?

Acts as a floor for the whole rubric: a hypothesis below the accuracy threshold
contributes no further scores. Metrics:

* ``citation_accuracy`` — do cited sources actually exist?
* ``citation_support`` — do those sources actually support the claims they're
  attached to?
* ``entity_accuracy`` — are genes/proteins/pathways real and correctly related?
* ``epistemic_calibration`` — does the tool distinguish established, contested,
  and uncertain claims?
* ``logical_consistency`` — do conclusions follow from premises without internal
  contradiction?
"""

from . import (  # noqa: F401,E501
    citation_accuracy,
    citation_support,
    entity_accuracy,
    epistemic_calibration,
    logical_consistency,
)
