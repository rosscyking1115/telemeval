"""The telemeval ingestion contract.

Validates interval labels and channel-keyed predictions before any metric is
computed, raising typed :mod:`telemeval.errors` instead of silently producing
a number from a malformed or leaky evaluation.

The label/prediction semantics are ported from the evaluator-contract layer of
the aerospace-prognostics project (ESA-ADB-compatible), generalized so the
core is not bound to ESA file layouts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from telemeval.errors import (
    AlignmentError,
    BinaryDomainError,
    IntervalError,
    MonotonicityError,
    SchemaError,
    TypeJoinError,
    WindowLeakageError,
)

LABEL_COLUMNS = ("ID", "Channel", "StartTime", "EndTime")
PREDICTION_COLUMNS = ("Timestamp", "Score")
INTERVAL_PREDICTION_COLUMNS = ("StartTime", "EndTime")

__all__ = [
    "INTERVAL_PREDICTION_COLUMNS",
    "LABEL_COLUMNS",
    "PREDICTION_COLUMNS",
    "assert_window_consistent",
    "build_metric_inputs",
    "intervals_to_mask",
    "validate_labels",
    "validate_predictions",
]


def validate_labels(
    labels: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
    *,
    metadata_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Validate and normalise interval labels, optionally joining event metadata.

    ``labels`` must contain :data:`LABEL_COLUMNS`. ``metadata`` (optional) is
    joined by ``ID`` — every label ID must have exactly one metadata row.
    """

    _require_columns(labels, LABEL_COLUMNS, "labels")
    labels = labels.loc[:, list(LABEL_COLUMNS)].copy()
    labels["ID"] = labels["ID"].astype(str)
    labels["Channel"] = labels["Channel"].astype(str)
    labels["StartTime"] = _parse_timestamps(labels["StartTime"], "StartTime")
    labels["EndTime"] = _parse_timestamps(labels["EndTime"], "EndTime")
    if (labels["StartTime"] > labels["EndTime"]).any():
        bad = labels.loc[labels["StartTime"] > labels["EndTime"], "ID"].tolist()
        raise IntervalError(f"labels contain StartTime values after EndTime for IDs: {bad}")

    if metadata is None:
        return labels

    if "ID" not in metadata.columns:
        raise SchemaError("event metadata is missing the required 'ID' column")
    if metadata_columns is None:
        metadata_columns = [column for column in metadata.columns if column != "ID"]
    _require_columns(metadata, ("ID", *metadata_columns), "event metadata")

    metadata = metadata.loc[:, ["ID", *metadata_columns]].copy()
    metadata["ID"] = metadata["ID"].astype(str)

    duplicate_ids = sorted(metadata.loc[metadata["ID"].duplicated(), "ID"].unique())
    if duplicate_ids:
        raise TypeJoinError(f"event metadata contains duplicate ID rows: {duplicate_ids}")
    missing_ids = sorted(set(labels["ID"]) - set(metadata["ID"]))
    if missing_ids:
        raise TypeJoinError(f"missing event metadata rows for label IDs: {missing_ids}")

    return labels.merge(metadata, on="ID", how="left", validate="many_to_one")


def validate_predictions(
    predictions_by_channel: Mapping[str, pd.DataFrame],
    *,
    threshold: float | None = None,
) -> dict[str, pd.DataFrame]:
    """Validate channel-keyed predictions into aligned binary detection frames.

    Two accepted per-channel forms:

    - **Binary mask** (``threshold is None``): ``Timestamp, Score`` with
      Score in {0, 1}.
    - **Continuous scores** (``threshold`` given): ``Timestamp, Score`` floats;
      binarized as ``Score > threshold``. Thresholds are caller inputs — never
      fitted here.
    """

    if not predictions_by_channel:
        raise SchemaError("predictions require at least one channel")
    if threshold is not None and not np.isfinite(threshold):
        raise SchemaError("threshold must be a finite number")

    normalised = {
        str(channel): _normalise_prediction_frame(frame, str(channel), threshold)
        for channel, frame in predictions_by_channel.items()
    }
    _validate_timestamp_alignment(normalised)
    return normalised


def intervals_to_mask(
    intervals: pd.DataFrame,
    timestamps: pd.DatetimeIndex | Sequence[Any],
) -> pd.DataFrame:
    """Convert interval predictions to a binary ``Timestamp, Score`` mask.

    ``intervals`` must contain :data:`INTERVAL_PREDICTION_COLUMNS`; membership
    is inclusive of both interval bounds, matching the label semantics.
    """

    _require_columns(intervals, INTERVAL_PREDICTION_COLUMNS, "interval predictions")
    index = pd.DatetimeIndex(pd.to_datetime(list(timestamps)))
    starts = _parse_timestamps(intervals["StartTime"], "StartTime")
    ends = _parse_timestamps(intervals["EndTime"], "EndTime")
    if (starts > ends).any():
        raise IntervalError("interval predictions contain StartTime values after EndTime")

    grid = index.to_numpy("datetime64[ns]")
    mask = np.zeros(len(grid), dtype="uint8")
    for start, end in zip(
        starts.to_numpy("datetime64[ns]"), ends.to_numpy("datetime64[ns]"), strict=True
    ):
        lo = int(np.searchsorted(grid, start, side="left"))
        hi = int(np.searchsorted(grid, end, side="right"))
        if hi > lo:
            mask[lo:hi] = 1
    return pd.DataFrame({"Timestamp": index, "Score": mask})


def assert_window_consistent(
    labels: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    *,
    clip_to_window: bool = False,
) -> pd.DataFrame:
    """Guard against the train/test-window leakage class of evaluation bug.

    Events lying entirely outside ``[window_start, window_end]`` can never be
    detected by predictions covering that window; counting them corrupts
    recall. This guard raises :class:`WindowLeakageError` unless the caller
    explicitly opts into ``clip_to_window=True``, in which case the offending
    events are removed and the filtered labels are returned. Silent filtering
    is deliberately not performed.
    """

    _require_columns(labels, ("ID", "StartTime", "EndTime"), "labels")
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)
    if window_start > window_end:
        raise IntervalError("window_start must not be after window_end")

    starts = pd.to_datetime(labels["StartTime"])
    ends = pd.to_datetime(labels["EndTime"])
    # An event overlaps the window when any part of it falls inside.
    per_row_outside = (ends < window_start) | (starts > window_end)
    outside_ids = set(labels.loc[per_row_outside, "ID"].astype(str))
    inside_ids = set(labels.loc[~per_row_outside, "ID"].astype(str))
    fully_outside = sorted(outside_ids - inside_ids)

    if not fully_outside:
        return labels.copy()
    if not clip_to_window:
        raise WindowLeakageError(
            f"{len(fully_outside)} labelled event(s) lie entirely outside the "
            f"prediction window [{window_start} .. {window_end}]: "
            f"{fully_outside[:10]}{'...' if len(fully_outside) > 10 else ''}. "
            "These events can never be detected by the supplied predictions, "
            "which corrupts recall. Pass window-consistent labels, or opt in "
            "explicitly with clip_to_window=True."
        )
    keep = ~labels["ID"].astype(str).isin(fully_outside)
    return labels.loc[keep].copy()


def build_metric_inputs(
    labels: pd.DataFrame,
    predictions_by_channel: Mapping[str, pd.DataFrame],
    *,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Build validated global and per-channel metric inputs.

    Returns a mapping with ``global_labels`` (channel column dropped),
    ``global_predictions`` (max score across channels), ``channel_labels``,
    ``channel_predictions``, and ``target_channels`` — the shape consumed by
    the metric implementations. Only labels on predicted channels are scored.
    """

    labels = _require_validated_labels(labels)
    channel_predictions = validate_predictions(predictions_by_channel, threshold=threshold)

    target_channels = tuple(channel_predictions)
    metric_labels = labels[labels["Channel"].isin(target_channels)].copy()
    global_labels = metric_labels.drop(columns=["Channel"])

    first_channel = target_channels[0]
    global_predictions = channel_predictions[first_channel][["Timestamp"]].copy()
    score_frame = pd.DataFrame(
        {channel: frame["Score"].to_numpy() for channel, frame in channel_predictions.items()}
    )
    global_predictions["Score"] = score_frame.max(axis=1).astype("uint8")

    return {
        "global_labels": global_labels.reset_index(drop=True),
        "global_predictions": global_predictions.reset_index(drop=True),
        "channel_labels": metric_labels.reset_index(drop=True),
        "channel_predictions": {
            channel: frame.reset_index(drop=True)
            for channel, frame in channel_predictions.items()
        },
        "target_channels": list(target_channels),
    }


def _require_validated_labels(labels: pd.DataFrame) -> pd.DataFrame:
    _require_columns(labels, LABEL_COLUMNS, "labels")
    labels = labels.copy()
    labels["ID"] = labels["ID"].astype(str)
    labels["Channel"] = labels["Channel"].astype(str)
    labels["StartTime"] = _parse_timestamps(labels["StartTime"], "StartTime")
    labels["EndTime"] = _parse_timestamps(labels["EndTime"], "EndTime")
    if (labels["StartTime"] > labels["EndTime"]).any():
        raise IntervalError("labels contain StartTime values after EndTime")
    return labels


def _normalise_prediction_frame(
    predictions: pd.DataFrame,
    channel_name: str,
    threshold: float | None,
) -> pd.DataFrame:
    _require_columns(predictions, PREDICTION_COLUMNS, f"predictions for {channel_name}")
    predictions = predictions.loc[:, list(PREDICTION_COLUMNS)].copy()
    if predictions.empty:
        raise SchemaError(f"predictions for {channel_name} must contain at least one row")
    predictions["Timestamp"] = _parse_timestamps(predictions["Timestamp"], "Timestamp")
    if predictions["Timestamp"].duplicated().any():
        raise MonotonicityError(f"predictions for {channel_name} contain duplicate timestamps")
    if not predictions["Timestamp"].is_monotonic_increasing:
        raise MonotonicityError(f"predictions for {channel_name} timestamps must be increasing")

    scores = predictions["Score"]
    if threshold is None:
        if not scores.isin([0, 1, False, True]).all():
            raise BinaryDomainError(
                f"predictions for {channel_name} must be binary; pass threshold=... "
                "to binarize continuous scores"
            )
        predictions["Score"] = scores.astype("uint8")
    else:
        numeric = pd.to_numeric(scores, errors="coerce")
        if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype="float64")).all():
            raise SchemaError(
                f"predictions for {channel_name} contain non-numeric or non-finite scores"
            )
        predictions["Score"] = (numeric.to_numpy(dtype="float64") > threshold).astype("uint8")
    return predictions


def _validate_timestamp_alignment(
    predictions_by_channel: Mapping[str, pd.DataFrame],
) -> None:
    timestamps: pd.Series | None = None
    for channel, predictions in predictions_by_channel.items():
        current = predictions["Timestamp"]
        if timestamps is None:
            timestamps = current
            continue
        if len(current) != len(timestamps) or not current.reset_index(drop=True).equals(
            timestamps.reset_index(drop=True)
        ):
            raise AlignmentError(
                "prediction timestamps must align across channels; "
                f"channel {channel} differs from the first channel"
            )


def _parse_timestamps(values: pd.Series, column_name: str) -> pd.Series:
    try:
        parsed = pd.to_datetime(values, errors="raise", utc=True).dt.tz_convert(None)
    except (ValueError, TypeError) as exc:
        raise SchemaError(f"column {column_name} contains unparseable timestamps: {exc}") from exc
    if parsed.isna().any():
        raise SchemaError(f"timestamp column {column_name} contains missing values")
    return parsed


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], source_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise SchemaError(f"{source_name} is missing required column(s): {missing}")
