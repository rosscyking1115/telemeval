"""TimeEval canonical dataset format reader.

TimeEval's canonical layout is the de-facto research CSV for time-series
anomaly detection: a ``timestamp`` column, one or more value columns, and a
binary ``is_anomaly`` column (see
https://timeeval.readthedocs.io/en/latest/concepts/datasets.html). This
reader converts that layout into the telemeval label contract so existing
benchmark corpora can be evaluated directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from telemeval.errors import BinaryDomainError, SchemaError

__all__ = ["read_dataset"]


def read_dataset(
    csv_path: str | Path,
    *,
    timestamp_column: str = "timestamp",
    label_column: str = "is_anomaly",
    channel: str = "series",
    event_id_prefix: str = "event",
) -> dict[str, Any]:
    """Read a TimeEval-canonical CSV into telemeval contract inputs.

    Returns a mapping with:

    - ``labels`` — interval labels (``ID, Channel, StartTime, EndTime``)
      derived from runs of ``is_anomaly == 1``;
    - ``timestamps`` — the parsed :class:`pandas.DatetimeIndex` grid;
    - ``values`` — the value column(s) as a DataFrame indexed by timestamp
      (for running detectors; telemeval itself does not consume them);
    - ``ground_truth_mask`` — the raw binary vector.

    Numeric timestamps are interpreted as **seconds since the Unix epoch**
    (TimeEval's common convention); datetime strings are parsed as such.
    """

    frame = pd.read_csv(csv_path)
    for column in (timestamp_column, label_column):
        if column not in frame.columns:
            raise SchemaError(
                f"TimeEval dataset {csv_path} is missing required column {column!r}"
            )

    raw_timestamps = frame[timestamp_column]
    if pd.api.types.is_numeric_dtype(raw_timestamps):
        timestamps = pd.to_datetime(raw_timestamps, unit="s")
    else:
        timestamps = pd.to_datetime(raw_timestamps, errors="raise", utc=True).dt.tz_convert(
            None
        )
    index = pd.DatetimeIndex(timestamps)
    if index.duplicated().any() or not index.is_monotonic_increasing:
        raise SchemaError(
            f"TimeEval dataset {csv_path} timestamps must be strictly increasing"
        )

    mask = frame[label_column]
    if not mask.isin([0, 1, False, True]).all():
        raise BinaryDomainError(
            f"TimeEval dataset {csv_path} column {label_column!r} must be binary"
        )
    mask_array = mask.to_numpy(dtype="uint8")

    padded = np.concatenate(([0], mask_array, [0]))
    edges = np.flatnonzero(np.diff(padded))
    starts, ends = edges[::2], edges[1::2] - 1
    labels = pd.DataFrame(
        {
            "ID": [f"{event_id_prefix}_{i}" for i in range(len(starts))],
            "Channel": channel,
            "StartTime": index[starts],
            "EndTime": index[ends],
        }
    )

    value_columns = [
        column for column in frame.columns if column not in (timestamp_column, label_column)
    ]
    values = frame[value_columns].copy()
    values.index = index

    return {
        "labels": labels,
        "timestamps": index,
        "values": values,
        "ground_truth_mask": mask_array,
    }
