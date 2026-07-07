"""scikit-learn-convention metric wrappers.

Thin ``score(y_true, y_pred) -> float`` functions over 1-D binary arrays, so
telemeval's metrics drop into existing sklearn-style scoring code
(``sklearn.metrics.make_scorer`` compatible via ``needs_proba=False``
defaults). The rich, guarded API remains :func:`telemeval.evaluate`.

The arrays are interpreted on a synthetic uniform grid (one sample per step);
ground-truth runs become interval events. This is the standard point-mask
setting of research benchmarks — window-leakage guarding does not apply here
because y_true and y_pred cover the same samples by construction.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from telemeval.contract import build_metric_inputs, validate_labels
from telemeval.errors import SchemaError
from telemeval.metrics.affiliation import score_affiliation
from telemeval.metrics.event_wise import score_event_wise

__all__ = [
    "affiliation_fbeta_score",
    "affiliation_precision_score",
    "affiliation_recall_score",
    "event_wise_fbeta_score",
    "event_wise_precision_score",
    "event_wise_recall_score",
]

_CHANNEL = "series"
_BASE = pd.Timestamp("2000-01-01")
_STEP = pd.Timedelta(minutes=1)


def event_wise_precision_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return _event_wise(y_true, y_pred, 0.5)["event_wise_precision"]


def event_wise_recall_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return _event_wise(y_true, y_pred, 0.5)["event_wise_recall"]


def event_wise_fbeta_score(
    y_true: Sequence[int], y_pred: Sequence[int], *, beta: float = 0.5
) -> float:
    return _event_wise(y_true, y_pred, beta)["event_wise_fbeta"]


def affiliation_precision_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    value = _affiliation(y_true, y_pred, 0.5)["affiliation_precision"]
    return float("nan") if value is None else value


def affiliation_recall_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    value = _affiliation(y_true, y_pred, 0.5)["affiliation_recall"]
    return float("nan") if value is None else value


def affiliation_fbeta_score(
    y_true: Sequence[int], y_pred: Sequence[int], *, beta: float = 0.5
) -> float:
    return _affiliation(y_true, y_pred, beta)["affiliation_fbeta"]


def _event_wise(y_true: Sequence[int], y_pred: Sequence[int], beta: float) -> dict:
    inputs = _to_metric_inputs(y_true, y_pred)
    return score_event_wise(inputs, beta=beta)


def _affiliation(y_true: Sequence[int], y_pred: Sequence[int], beta: float) -> dict:
    inputs = _to_metric_inputs(y_true, y_pred)
    return score_affiliation(inputs, beta=beta)


def _to_metric_inputs(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    true_array = _validated_binary(y_true, "y_true")
    pred_array = _validated_binary(y_pred, "y_pred")
    if len(true_array) != len(pred_array):
        raise SchemaError(
            f"y_true (len {len(true_array)}) and y_pred (len {len(pred_array)}) "
            "must have the same length"
        )
    if not true_array.any():
        raise SchemaError(
            "y_true contains no anomalous samples; event metrics require at "
            "least one ground-truth event"
        )

    timestamps = pd.DatetimeIndex(_BASE + _STEP * np.arange(len(true_array)))
    labels = _mask_to_labels(true_array, timestamps)
    predictions = {_CHANNEL: pd.DataFrame({"Timestamp": timestamps, "Score": pred_array})}
    return build_metric_inputs(validate_labels(labels), predictions)


def _mask_to_labels(mask: np.ndarray, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    padded = np.concatenate(([0], mask, [0]))
    edges = np.flatnonzero(np.diff(padded))
    starts, ends = edges[::2], edges[1::2] - 1
    return pd.DataFrame(
        {
            "ID": [f"event_{i}" for i in range(len(starts))],
            "Channel": _CHANNEL,
            "StartTime": timestamps[starts],
            "EndTime": timestamps[ends],
        }
    )


def _validated_binary(values: Sequence[int], name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise SchemaError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise SchemaError(f"{name} must not be empty")
    if not np.isin(array, (0, 1, False, True)).all():
        raise SchemaError(f"{name} must be binary (0/1)")
    return array.astype("uint8")
