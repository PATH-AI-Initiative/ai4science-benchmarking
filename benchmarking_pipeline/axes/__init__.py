"""The four evaluation axes, one subpackage each. Importing this package
imports every axis so that all metrics register themselves.
"""

from . import accuracy, details_features, novelty, quality  # noqa: F401
