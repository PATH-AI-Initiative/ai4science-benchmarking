"""Tier 4 — Scientific Novelty. Characterises what a tool's outputs look like in
relation to existing knowledge, rather than assigning a definitive novelty score.

Reported as a separate profile, never folded into the composite (novelty has no
ground truth). Metrics:

* ``size_of_leap`` — distance of each hypothesis from the nearest known idea in
  a literature embedding space.
* ``diversity`` — how spread the hypotheses are across conceptual space.
"""

from . import diversity, size_of_leap  # noqa: F401
