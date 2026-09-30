"""Metric registry.

Metrics register themselves with the ``@register`` decorator; axes and the
pipeline discover them by axis. This is what lets contributors drop a new metric
into an axis package without editing the scoring layer.
"""

from __future__ import annotations

from .metric import Axis, HypothesisMetric, Metric, MultiRunMetric, SetMetric

_REGISTRY: dict[str, Metric] = {}


def register(metric_cls: type[Metric]) -> type[Metric]:
    """Class decorator: instantiate a metric and add it to the registry."""
    instance = metric_cls()
    if instance.name in _REGISTRY:
        raise ValueError(f"Duplicate metric name: {instance.name!r}")
    _REGISTRY[instance.name] = instance
    return metric_cls


def all_metrics() -> list[Metric]:
    return list(_REGISTRY.values())


def metrics_for_axis(axis: Axis) -> list[Metric]:
    return [m for m in _REGISTRY.values() if m.axis is axis]


def hypothesis_metrics(axis: Axis | None = None) -> list[HypothesisMetric]:
    return [
        m
        for m in _REGISTRY.values()
        if isinstance(m, HypothesisMetric) and (axis is None or m.axis is axis)
    ]


def set_metrics(axis: Axis | None = None) -> list[SetMetric]:
    return [
        m
        for m in _REGISTRY.values()
        if isinstance(m, SetMetric) and (axis is None or m.axis is axis)
    ]


def multi_run_metrics(axis: Axis | None = None) -> list[MultiRunMetric]:
    return [
        m
        for m in _REGISTRY.values()
        if isinstance(m, MultiRunMetric) and (axis is None or m.axis is axis)
    ]
