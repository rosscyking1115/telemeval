"""Metric registry.

New metrics (e.g. ADTQC at v1.x) plug in here rather than being hardcoded
into the facade. A metric is a callable taking the validated metric inputs
(from :func:`telemeval.contract.build_metric_inputs`) plus keyword options,
returning a JSON-serializable mapping of scores.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from telemeval.errors import SchemaError
from telemeval.metrics.adtqc import score_adtqc
from telemeval.metrics.affiliation import score_affiliation
from telemeval.metrics.event_wise import score_event_wise

MetricFn = Callable[..., Mapping[str, Any]]

_REGISTRY: dict[str, MetricFn] = {}

__all__ = ["available_metrics", "get_metric", "register_metric"]


def register_metric(name: str, fn: MetricFn, *, overwrite: bool = False) -> None:
    """Register a metric under ``name`` for use with ``evaluate(metrics=...)``."""

    if not overwrite and name in _REGISTRY:
        raise SchemaError(f"metric {name!r} is already registered; pass overwrite=True")
    _REGISTRY[name] = fn


def get_metric(name: str) -> MetricFn:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise SchemaError(
            f"unknown metric {name!r}; available: {sorted(_REGISTRY)}"
        ) from None


def available_metrics() -> list[str]:
    return sorted(_REGISTRY)


register_metric("event_wise", score_event_wise)
register_metric("affiliation", score_affiliation)
# Detection-timing quality (ESA-ADB ADTQC). Not in DEFAULT_METRICS yet:
# timing is only meaningful once detection quality is understood, so it is
# explicit opt-in via evaluate(metrics=(..., "adtqc")).
register_metric("adtqc", score_adtqc)
