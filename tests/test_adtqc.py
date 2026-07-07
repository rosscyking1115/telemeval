from __future__ import annotations

import math

import pandas as pd
import pytest

from telemeval import build_metric_inputs, evaluate, score_adtqc, timing_curve, validate_labels
from telemeval.errors import SchemaError


def _inputs(label_rows: list[list[object]], scores: list[int]):
    labels = pd.DataFrame(
        label_rows, columns=["ID", "Channel", "StartTime", "EndTime"]
    )
    timestamps = pd.to_datetime(
        [pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=m) for m in range(len(scores))]
    )
    predictions = {"ch1": pd.DataFrame({"Timestamp": timestamps, "Score": scores})}
    return build_metric_inputs(validate_labels(labels), predictions)


# --- the curve itself (reference semantics) ---


def test_curve_perfect_point_anomaly_special_case() -> None:
    # x == 0 with zero-length event or zero allowance scores 1 (reference rule).
    assert timing_curve(0.0, 0.0, 0.0) == 1.0
    assert timing_curve(0.0, 0.0, 60.0) == 1.0
    assert timing_curve(0.0, 60.0, 0.0) == 1.0


def test_curve_zero_outside_bounds() -> None:
    assert timing_curve(-60.0, 60.0, 120.0) == 0.0   # x <= -a
    assert timing_curve(120.0, 60.0, 120.0) == 0.0   # x >= b
    assert timing_curve(500.0, 60.0, 120.0) == 0.0


def test_curve_before_branch_ramps_up_to_one() -> None:
    # -a < x <= 0: ((x + a)/a)^e — rises toward 1 at onset.
    a, b = 60.0, 120.0
    assert timing_curve(0.0, a, b) == 1.0
    mid = timing_curve(-30.0, a, b)
    assert 0.0 < mid < 1.0
    assert mid == pytest.approx(0.5**math.e)
    assert timing_curve(-59.0, a, b) < mid


def test_curve_after_branch_decays_from_one() -> None:
    # 0 < x < b: 1/(1 + (x/(b-x))^e) — halfway through the event gives 0.5.
    b = 120.0
    assert timing_curve(60.0, 60.0, b) == pytest.approx(0.5)
    assert timing_curve(30.0, 60.0, b) > 0.5
    assert timing_curve(90.0, 60.0, b) < 0.5


def test_curve_rejects_negative_allowance_or_length() -> None:
    with pytest.raises(SchemaError):
        timing_curve(0.0, -1.0, 10.0)


# --- allowance (alpha) semantics ---


def test_first_event_allowance_equals_its_length() -> None:
    # Event of 2 min starting at minute 4; detection 1 min early is inside the
    # allowance (a = length = 120s) and must score the before-branch value.
    inputs = _inputs(
        [["A-1", "ch1", "2024-01-01T00:04:00", "2024-01-01T00:06:00"]],
        [0, 0, 0, 1, 1, 1, 1, 0],
    )

    scores = score_adtqc(inputs)

    event = scores["per_event"][0]
    assert event["timing"] == "before"
    assert event["latency_seconds"] == -60.0
    assert event["curve_score"] == pytest.approx(timing_curve(-60.0, 120.0, 120.0))


def test_second_event_allowance_capped_by_gap_to_previous_start() -> None:
    # Two events starting 3 min apart; second event is 10 min long, so its
    # allowance is capped by the 180s gap, not its own 600s length. A detection
    # 4 min early (240s) falls outside that allowance -> curve score 0.
    inputs = _inputs(
        [
            ["A-1", "ch1", "2024-01-01T00:02:00", "2024-01-01T00:03:00"],
            ["A-2", "ch1", "2024-01-01T00:05:00", "2024-01-01T00:15:00"],
        ],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    )
    # The run at minutes 2-3 overlaps BOTH events? No: event 2 starts at min 5,
    # run ends min 3 -> overlaps only event 1. Extend a second early run:
    inputs = _inputs(
        [
            ["A-1", "ch1", "2024-01-01T00:02:00", "2024-01-01T00:03:00"],
            ["A-2", "ch1", "2024-01-01T00:05:00", "2024-01-01T00:15:00"],
        ],
        # run1 detects event1 at onset; run2 starts min 1... must not overlap
        # event1. Use a run starting at min 1? It would overlap event 1 too.
        # Instead: second run starts at minute 1 + is separate — simpler to
        # give event 2 a detection starting exactly 240s early via a run that
        # begins after event 1 ends (min 4) -> latency -60s, inside 180s gap.
        [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    )

    scores = score_adtqc(inputs)

    second = scores["per_event"][1]
    # run spans minutes 2-5, overlaps event 2 (starts min 5): latency = 2min-5min = -180s
    assert second["latency_seconds"] == -180.0
    # allowance = min(600s length, 180s gap) = 180s -> x <= -a -> score 0
    assert second["curve_score"] == 0.0
    assert second["timing"] == "before"


# --- detection assignment & aggregates ---


def test_undetected_events_are_skipped_not_zero_scored() -> None:
    inputs = _inputs(
        [
            ["hit", "ch1", "2024-01-01T00:01:00", "2024-01-01T00:02:00"],
            ["miss", "ch1", "2024-01-01T00:05:00", "2024-01-01T00:06:00"],
        ],
        [0, 1, 0, 0, 0, 0, 0, 0],
    )

    scores = score_adtqc(inputs)

    assert scores["total_events"] == 2
    assert scores["detected_events"] == 1
    assert scores["per_event"][1]["detected"] is False
    assert scores["per_event"][1]["curve_score"] is None
    # aggregates over detected events only
    assert scores["adtqc_total"] == scores["per_event"][0]["curve_score"]
    assert scores["after_rate"] == 1.0  # latency 0 counts as "after"


def test_no_detections_yield_none_aggregates() -> None:
    inputs = _inputs(
        [["A-1", "ch1", "2024-01-01T00:01:00", "2024-01-01T00:02:00"]],
        [0, 0, 0, 0],
    )

    scores = score_adtqc(inputs)

    assert scores["detected_events"] == 0
    assert scores["after_rate"] is None
    assert scores["adtqc_total"] is None


def test_detection_at_onset_scores_one_and_counts_after() -> None:
    inputs = _inputs(
        [["A-1", "ch1", "2024-01-01T00:02:00", "2024-01-01T00:04:00"]],
        [0, 0, 1, 1, 1, 0],
    )

    scores = score_adtqc(inputs)

    event = scores["per_event"][0]
    assert event["latency_seconds"] == 0.0
    assert event["curve_score"] == 1.0
    assert scores["nb_after"] == 1
    assert scores["nb_before"] == 0
    assert scores["adtqc_total"] == 1.0


def test_late_detection_scores_below_half() -> None:
    # 4-minute event detected in its final minute.
    inputs = _inputs(
        [["A-1", "ch1", "2024-01-01T00:00:00", "2024-01-01T00:04:00"]],
        [0, 0, 0, 1, 1, 0],
    )

    scores = score_adtqc(inputs)

    event = scores["per_event"][0]
    assert event["latency_seconds"] == 180.0
    assert 0.0 < event["curve_score"] < 0.5


# --- integration ---


def test_adtqc_runs_through_evaluate_registry() -> None:
    labels = pd.DataFrame(
        [["A-1", "ch1", "2024-01-01T00:02:00", "2024-01-01T00:03:00"]],
        columns=["ID", "Channel", "StartTime", "EndTime"],
    )
    timestamps = pd.to_datetime(
        [pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=m) for m in range(6)]
    )
    predictions = {"ch1": pd.DataFrame({"Timestamp": timestamps, "Score": [0, 0, 1, 1, 0, 0]})}

    result = evaluate(labels, predictions, metrics=("event_wise", "adtqc"))

    assert result.metrics["adtqc"]["adtqc_total"] == 1.0
    report = result.to_report()
    assert "adtqc" in report["metrics"]


def test_exclusions_require_category_metadata() -> None:
    inputs = _inputs(
        [["A-1", "ch1", "2024-01-01T00:01:00", "2024-01-01T00:02:00"]],
        [0, 1, 0, 0],
    )

    with pytest.raises(SchemaError, match="no 'Category' metadata"):
        score_adtqc(inputs, exclude_categories=("Communication Gap",))
