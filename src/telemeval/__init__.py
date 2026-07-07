"""telemeval: leakage-safe, event-wise and affiliation-based evaluation
for spacecraft-telemetry anomaly detection."""

from __future__ import annotations

from telemeval.contract import (
    assert_window_consistent,
    build_metric_inputs,
    intervals_to_mask,
    validate_labels,
    validate_predictions,
)
from telemeval.errors import (
    AlignmentError,
    BinaryDomainError,
    ContractError,
    IntervalError,
    MonotonicityError,
    SchemaError,
    TypeJoinError,
    WindowLeakageError,
)
from telemeval.evaluate import DEFAULT_METRICS, EvaluationResult, evaluate
from telemeval.events import group_binary_events, positive_runs
from telemeval.metrics.affiliation import score_affiliation
from telemeval.metrics.event_wise import score_event_wise
from telemeval.registry import available_metrics, get_metric, register_metric
from telemeval.report import build_report, render_markdown, write_report

__version__ = "0.1.2"

__all__ = [
    "DEFAULT_METRICS",
    "AlignmentError",
    "BinaryDomainError",
    "ContractError",
    "EvaluationResult",
    "IntervalError",
    "MonotonicityError",
    "SchemaError",
    "TypeJoinError",
    "WindowLeakageError",
    "__version__",
    "assert_window_consistent",
    "available_metrics",
    "build_metric_inputs",
    "build_report",
    "evaluate",
    "get_metric",
    "group_binary_events",
    "intervals_to_mask",
    "positive_runs",
    "register_metric",
    "render_markdown",
    "score_affiliation",
    "score_event_wise",
    "validate_labels",
    "validate_predictions",
    "write_report",
]
