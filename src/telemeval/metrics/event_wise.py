"""Corrected event-wise precision / recall / F-beta.

Ported from the aerospace-prognostics event-wise scorer; semantics align with
ESA-ADB's corrected event-wise scoring (every event weighs equally regardless
of length; precision-weighted F0.5 by default).

Detection is sample-based and unambiguous: a labelled event is detected when
any positive prediction sample falls inside its inclusive
``[StartTime, EndTime]`` interval; a predicted positive run is a true alarm
when any of its samples falls inside any event, otherwise a false alarm.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from telemeval.errors import SchemaError
from telemeval.events import positive_runs

__all__ = ["score_event_wise"]


def score_event_wise(
    metric_inputs: Mapping[str, Any],
    *,
    beta: float = 0.5,
    exclude_categories: Sequence[str] = (),
) -> dict[str, Any]:
    """Score event-wise detection quality from validated metric inputs.

    ``metric_inputs`` is the mapping returned by
    :func:`telemeval.contract.build_metric_inputs`. ``beta=0.5`` weights
    precision above recall, matching ESA-ADB's false-alarm sensitivity.
    ``exclude_categories`` drops events by their ``Category`` metadata (e.g.
    communication gaps) — requesting exclusions without a Category column is
    a contract error, never a silent no-op.
    """

    if beta <= 0:
        raise SchemaError("beta must be positive")

    global_labels = metric_inputs["global_labels"]
    global_predictions = metric_inputs["global_predictions"]
    excluded = set(exclude_categories)
    if excluded and "Category" not in global_labels.columns:
        raise SchemaError(
            "exclude_categories was requested but labels carry no 'Category' "
            "metadata column; join event metadata first"
        )

    events = _collapse_events(global_labels, excluded)
    runs = positive_runs(global_predictions)

    per_event: list[dict[str, Any]] = []
    detected_events = 0
    for event in events:
        detected = any(_run_detects_event(run, event) for run in runs)
        detected_events += int(detected)
        per_event.append(
            {
                "id": event["id"],
                "start_time": event["start_time"],
                "end_time": event["end_time"],
                "category": event["category"],
                "detected": detected,
            }
        )

    true_alarm_runs = sum(
        1 for run in runs if any(_run_detects_event(run, event) for event in events)
    )
    total_events = len(events)
    total_runs = len(runs)
    false_alarm_runs = total_runs - true_alarm_runs

    recall = detected_events / total_events if total_events else 1.0
    if total_runs:
        precision = true_alarm_runs / total_runs
    elif total_events == 0:
        precision = 1.0
    else:
        precision = 0.0
    fbeta = _fbeta_score(precision, recall, beta)

    return {
        "beta": beta,
        "excluded_categories": sorted(excluded),
        "total_events": total_events,
        "detected_events": detected_events,
        "missed_events": total_events - detected_events,
        "predicted_alarms": total_runs,
        "true_alarms": true_alarm_runs,
        "false_alarms": false_alarm_runs,
        "event_wise_precision": precision,
        "event_wise_recall": recall,
        "event_wise_fbeta": fbeta,
        "per_event": per_event,
    }


def _collapse_events(
    global_labels: pd.DataFrame,
    excluded_categories: set[str],
) -> list[dict[str, Any]]:
    if global_labels.empty:
        return []

    has_category = "Category" in global_labels.columns
    events: list[dict[str, Any]] = []
    for event_id, group in global_labels.groupby("ID", sort=False):
        category = str(group["Category"].iloc[0]) if has_category else None
        if category is not None and category in excluded_categories:
            continue
        events.append(
            {
                "id": str(event_id),
                "start_time": group["StartTime"].min(),
                "end_time": group["EndTime"].max(),
                "category": category,
            }
        )
    events.sort(key=lambda event: (event["start_time"], event["id"]))
    return events


def _run_detects_event(run: Mapping[str, Any], event: Mapping[str, Any]) -> bool:
    start = event["start_time"]
    end = event["end_time"]
    # Cheap interval reject before scanning samples.
    if run["end_time"] < start or run["start_time"] > end:
        return False
    return any(start <= sample <= end for sample in run["samples"])


def _fbeta_score(precision: float, recall: float, beta: float) -> float:
    beta_sq = beta * beta
    denominator = beta_sq * precision + recall
    if denominator <= 0.0:
        return 0.0
    return (1.0 + beta_sq) * precision * recall / denominator
