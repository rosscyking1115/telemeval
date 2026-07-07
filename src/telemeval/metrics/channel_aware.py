"""Channel-aware and subsystem-aware F-scores (ESA-ADB).

"Did you point at the right *source*?" — for each event, every predicted
channel is judged: a channel labelled for the event counts as a true positive
when any of its detections overlap the event's full interval, a false
negative when none do, and an unlabelled-but-alarming channel is a false
positive *unless* its detection (restricted to this event's interval) overlaps
another event's labelled intervals on that same channel — one alarm spanning
two overlapping anomalies is not punished twice. Precision/recall/F-beta are
computed per event and averaged over events. The subsystem variant applies the
same logic to channel groups (an ESA-ADB ``channels.csv`` mapping).

Semantics implemented from and fixture-verified against the ESA-ADB reference
(`timeeval/metrics/ranking_metrics.py::ChannelAwareFScore`, MIT). Interval
arithmetic uses closed intervals on the prediction grid (the reference's
change-point series become sample runs here; equivalent at grid resolution).
Point-anomaly labels are widened by 1 ms exactly as the reference does.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from telemeval.errors import SchemaError
from telemeval.events import positive_runs

Interval = tuple[pd.Timestamp, pd.Timestamp]

__all__ = ["score_channel_aware"]


def score_channel_aware(
    metric_inputs: Mapping[str, Any],
    *,
    beta: float = 0.5,
    exclude_categories: Sequence[str] = (),
    subsystems_mapping: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Score affected-source diagnosis from validated metric inputs.

    ``metric_inputs`` is the mapping returned by
    :func:`telemeval.contract.build_metric_inputs`. Pass ``subsystems_mapping``
    (``{subsystem: [channel, ...]}``, e.g. from
    :func:`telemeval.formats.esa_adb.read_channels`) to also get
    subsystem-level scores.
    """

    if beta <= 0:
        raise SchemaError("beta must be positive")

    labels = metric_inputs["channel_labels"]
    channel_predictions = metric_inputs["channel_predictions"]
    channels = list(metric_inputs["target_channels"])

    excluded = set(exclude_categories)
    if excluded:
        if "Category" not in labels.columns:
            raise SchemaError(
                "exclude_categories was requested but labels carry no 'Category' "
                "metadata column; join event metadata first"
            )
        labels = labels[~labels["Category"].isin(excluded)]

    # Reference point-anomaly fix: widen zero-length labels by 1 ms.
    labels = labels.copy()
    is_point = labels["StartTime"] == labels["EndTime"]
    labels.loc[is_point, "EndTime"] = labels.loc[is_point, "StartTime"] + pd.Timedelta(
        milliseconds=1
    )

    predicted_intervals: dict[str, list[Interval]] = {
        channel: _merge(
            [(run["start_time"], run["end_time"]) for run in positive_runs(frame)]
        )
        for channel, frame in channel_predictions.items()
    }

    event_ids = list(dict.fromkeys(labels["ID"].astype(str)))
    event_channel_intervals: dict[str, dict[str, list[Interval]]] = {}
    for event_id in event_ids:
        event_rows = labels[labels["ID"].astype(str) == event_id]
        event_channel_intervals[event_id] = {
            channel: _merge(
                [
                    (row.StartTime, row.EndTime)
                    for row in event_rows[event_rows["Channel"] == channel][
                        ["StartTime", "EndTime"]
                    ].itertuples()
                ]
            )
            for channel in channels
        }

    channel_scores: list[tuple[float, float, float]] = []
    subsystem_scores: list[tuple[float, float, float]] = []
    per_event: list[dict[str, Any]] = []

    for event_id in event_ids:
        channels_intervals = event_channel_intervals[event_id]
        full_interval = _merge(
            [interval for parts in channels_intervals.values() for interval in parts]
        )

        tp = fp = fn = 0
        fp_channels: list[str] = []
        fn_channels: list[str] = []
        for channel in channels:
            affected = bool(channels_intervals[channel])
            detection = _intersect(full_interval, predicted_intervals[channel])
            detected = bool(detection)
            if affected and detected:
                tp += 1
            elif affected:
                fn += 1
                fn_channels.append(channel)
            elif detected and not _excused(
                detection, channel, event_id, event_channel_intervals
            ):
                fp += 1
                fp_channels.append(channel)

        precision, recall, f_score = _pr_re_f(tp, fp, fn, beta)
        channel_scores.append((precision, recall, f_score))
        detail: dict[str, Any] = {
            "id": event_id,
            "channel_precision": precision,
            "channel_recall": recall,
            "channel_fbeta": f_score,
            "false_positive_channels": fp_channels,
            "missed_channels": fn_channels,
        }

        if subsystems_mapping is not None:
            tp = fp = fn = 0
            for _sid, subsystem_channels in subsystems_mapping.items():
                members = [c for c in subsystem_channels if c in channels]
                if not members:
                    continue

                affected = any(bool(channels_intervals[c]) for c in members)
                detections = {
                    c: _intersect(full_interval, predicted_intervals[c]) for c in members
                }
                detected = any(bool(d) for d in detections.values())

                if affected and detected:
                    tp += 1
                elif affected:
                    fn += 1
                elif detected:
                    # Zero out detections excused by other events' labels on the
                    # same channel, then re-check (reference semantics).
                    for other_id, other_intervals in event_channel_intervals.items():
                        if other_id == event_id:
                            continue
                        for c in members:
                            if other_intervals[c] and _intersect(
                                detections[c], other_intervals[c]
                            ):
                                detections[c] = []
                    if any(bool(d) for d in detections.values()):
                        fp += 1

            precision, recall, f_score = _pr_re_f(tp, fp, fn, beta)
            subsystem_scores.append((precision, recall, f_score))
            detail["subsystem_precision"] = precision
            detail["subsystem_recall"] = recall
            detail["subsystem_fbeta"] = f_score

        per_event.append(detail)

    result: dict[str, Any] = {
        "beta": beta,
        "excluded_categories": sorted(excluded),
        "total_events": len(event_ids),
        "channel_precision": _mean(channel_scores, 0),
        "channel_recall": _mean(channel_scores, 1),
        "channel_fbeta": _mean(channel_scores, 2),
        "subsystem_precision": _mean(subsystem_scores, 0),
        "subsystem_recall": _mean(subsystem_scores, 1),
        "subsystem_fbeta": _mean(subsystem_scores, 2),
        "per_event": per_event,
    }
    return result


def _excused(
    detection: list[Interval],
    channel: str,
    event_id: str,
    event_channel_intervals: Mapping[str, Mapping[str, list[Interval]]],
) -> bool:
    """A false alarm is excused when its detection (restricted to this event's
    interval) overlaps another event's labelled intervals on the same channel."""

    for other_id, other_intervals in event_channel_intervals.items():
        if other_id == event_id or not other_intervals[channel]:
            continue
        if _intersect(detection, other_intervals[channel]):
            return True
    return False


def _pr_re_f(tp: int, fp: int, fn: int, beta: float) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    denominator = (beta**2) * precision + recall
    f_score = ((1 + beta**2) * precision * recall) / denominator if denominator else 0.0
    return precision, recall, f_score


def _mean(scores: list[tuple[float, float, float]], index: int) -> float | None:
    if not scores:
        return None
    return sum(score[index] for score in scores) / len(scores)


def _merge(intervals: list[Interval]) -> list[Interval]:
    """Union of closed intervals, returned sorted and non-overlapping."""

    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _intersect(a: list[Interval], b: list[Interval]) -> list[Interval]:
    """Intersection of two closed-interval unions (two-pointer sweep)."""

    result: list[Interval] = []
    i = j = 0
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if start <= end:
            result.append((start, end))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return result
