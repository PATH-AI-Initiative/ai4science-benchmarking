"""Tier 3 — Quality. Are the outputs reliable enough to act on, and would acting
on them be responsible?

* ``robustness`` — do outputs converge under prompt / knowledge-base perturbation?
* ``reproducibility`` — does the same prompt yield substantively similar outputs?
* ``adversarial`` — does the tool resist misleading inputs (retracted papers,
  known negatives, contradictory premises)?
* ``tractability`` — are hypotheses actionable, feasible, ethical (LMIC lens)?
* ``safety`` — do outputs pose biorisk / DURC concerns? Acts as a hard gate.
"""

from . import adversarial, reproducibility, robustness, safety, tractability  # noqa: F401
