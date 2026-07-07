"""ADTQC — Anomaly Detection Timing Quality Curve (ESA-ADB).

Scores *when* each ground-truth event was first detected, not just whether.
Semantics follow the ESA-ADB reference (`timeeval/metrics/latency_metrics.py`
in https://github.com/kplabs-pl/ESA-ADB, MIT; introduced in arXiv:2406.17826):

- For each event, the detection latency ``x`` is the start of the earliest
  predicted positive run that **overlaps** the event, minus the event start —
  negative for early (before-onset) detections.
- The timing curve ``f(x, a, b)`` maps latency to a [0, 1] quality score,
  where ``b`` is the event length and ``a`` (the early-detection allowance) is
  ``min(length, gap to the previous event's start)`` — an early detection is
  only credible if it fired after the previous anomaly began. For the first
  event ``a`` equals its length.
- ``x == 0`` with a zero-length allowance or event scores 1 (a point anomaly
  caught exactly at onset is a perfect detection).
- Detections before the allowance (``x <= -a``) or after the event ends
  (``x >= b``) score 0.
- Aggregates are computed **over detected events only** (undetected events are
  the recall metric's business, not the timing metric's): ``nb_before`` /
  ``nb_after`` (latency < 0 vs >= 0), ``after_rate``, and ``total`` (mean
  curve score).

Deviation from the reference, recorded honestly: telemeval computes latencies
in **seconds on the prediction timestamp grid** of the merged global timeline
(consistent with its other metrics), rather than on ESA-ADB's per-channel
resampled index. Channel-aware ADTQC arrives with the subsystem work (see
specs/roadmap.md).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from telemeval.errors import SchemaError
from telemeval.events import positive_runs

DEFAULT_EXPONENT = math.e

__all__ = ["score_adtqc", "timing_curve"]


def timing_curve(x: float, a: float, b: float, *, exponent: float = DEFAULT_EXPONENT) -> float:
    """The ESA-ADB timing quality curve: latency -> [0, 1] score.

    ``x`` is the detection latency, ``a`` the early-detection allowance,
    ``b`` the event length (all in the same time unit; ``a, b >= 0``).
    """

    if a < 0 or b < 0:
        raise SchemaError("timing_curve allowance and length must be non-negative")
    if x == 0 and (a == 0 or b == 0):
        return 1.0
    if x <= -a or x >= b:
        return 0.0
    if x <= 0:
        return ((x + a) / a) ** exponent
    return 1.0 / (1.0 + (x / (b - x)) ** exponent)


def score_adtqc(
    metric_inputs: Mapping[str, Any],
    *,
    beta: float = 0.5,  # accepted for registry uniformity; ADTQC does not use it
    exclude_categories: Sequence[str] = (),
    exponent: float = DEFAULT_EXPONENT,
) -> dict[str, Any]:
    """Score detection timing from validated metric inputs.

    ``metric_inputs`` is the mapping returned by
    :func:`telemeval.contract.build_metric_inputs`. Events with no overlapping
    detection are skipped (timing is only defined for detected events);
    ``after_rate`` and ``total`` are ``None`` when nothing was detected.
    """

    del beta  # timing has no precision/recall trade-off to weight

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
    curve_scores: list[float] = []
    nb_before = 0
    nb_after = 0

    previous_start: pd.Timestamp | None = None
    for event in events:
        start, end = event["start_time"], event["end_time"]
        length = (end - start).total_seconds()
        if previous_start is None:
            allowance = length
        else:
            allowance = min(length, (start - previous_start).total_seconds())
        previous_start = start

        first_overlapping = next(
            (run for run in runs if run["start_time"] <= end and run["end_time"] >= start),
            None,
        )
        if first_overlapping is None:
            per_event.append(
                {
                    "id": event["id"],
                    "detected": False,
                    "latency_seconds": None,
                    "curve_score": None,
                    "timing": None,
                }
            )
            continue

        latency = (first_overlapping["start_time"] - start).total_seconds()
        score = timing_curve(latency, allowance, length, exponent=exponent)
        if latency < 0:
            nb_before += 1
        else:
            nb_after += 1
        curve_scores.append(score)
        per_event.append(
            {
                "id": event["id"],
                "detected": True,
                "latency_seconds": latency,
                "curve_score": score,
                "timing": "before" if latency < 0 else "after",
            }
        )

    detected = len(curve_scores)
    return {
        "exponent": exponent,
        "excluded_categories": sorted(excluded),
        "total_events": len(events),
        "detected_events": detected,
        "nb_before": nb_before,
        "nb_after": nb_after,
        "after_rate": (nb_after / detected) if detected else None,
        "adtqc_total": (sum(curve_scores) / detected) if detected else None,
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
            }
        )
    events.sort(key=lambda event: (event["start_time"], event["id"]))
    return events
