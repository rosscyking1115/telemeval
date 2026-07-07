"""Event grouping over binary detection series.

Ported from the aerospace-prognostics evaluator-contract layer; the grouping
semantics mirror the official ESA-ADB TimeEval run handling.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from telemeval.contract import PREDICTION_COLUMNS
from telemeval.errors import SchemaError

__all__ = ["group_binary_events", "positive_runs"]


def group_binary_events(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    """Group a binary ``Timestamp, Score`` series into detection events.

    Each positive sample covers the interval from its timestamp to the next
    timestamp (closed-open), except a run reaching the final sample, which
    ends closed at the final timestamp — matching official ESA-ADB TimeEval
    run semantics.
    """

    _require_prediction_columns(predictions)
    scores = predictions["Score"].to_list()
    timestamps = predictions["Timestamp"].to_list()
    events: list[dict[str, Any]] = []
    index = 0
    n_rows = len(predictions)

    while index < n_rows:
        if int(scores[index]) <= 0:
            index += 1
            continue

        start_index = index
        while index < n_rows and int(scores[index]) > 0:
            index += 1

        reaches_final_sample = index == n_rows
        end_index = index - 1 if reaches_final_sample else index
        events.append(
            {
                "start_time": timestamps[start_index],
                "end_time": timestamps[end_index],
                "end_inclusive": reaches_final_sample,
            }
        )

    return events


def positive_runs(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    """Return maximal runs of consecutive positive samples with their samples.

    Unlike :func:`group_binary_events` (which extends runs to the next
    timestamp), each run here carries its own positive sample timestamps —
    the form consumed by event-wise alarm counting.
    """

    _require_prediction_columns(predictions)
    timestamps = predictions["Timestamp"].to_list()
    scores = predictions["Score"].to_list()

    runs: list[dict[str, Any]] = []
    index = 0
    n_rows = len(scores)
    while index < n_rows:
        if int(scores[index]) <= 0:
            index += 1
            continue
        start_index = index
        while index < n_rows and int(scores[index]) > 0:
            index += 1
        samples = timestamps[start_index:index]
        runs.append(
            {
                "start_time": samples[0],
                "end_time": samples[-1],
                "samples": samples,
            }
        )
    return runs


def _require_prediction_columns(predictions: pd.DataFrame) -> None:
    missing = [column for column in PREDICTION_COLUMNS if column not in predictions.columns]
    if missing:
        raise SchemaError(f"binary predictions are missing required column(s): {missing}")
