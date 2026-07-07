from __future__ import annotations

import pandas as pd
import pytest

from telemeval import (
    SchemaError,
    build_metric_inputs,
    group_binary_events,
    score_event_wise,
    validate_labels,
)


def _validated_labels(rows: list[list[object]], with_metadata: bool = True) -> pd.DataFrame:
    labels = pd.DataFrame(
        [row[:4] for row in rows], columns=["ID", "Channel", "StartTime", "EndTime"]
    )
    if not with_metadata:
        return validate_labels(labels)
    metadata = pd.DataFrame(
        [[row[0], row[4], "Univariate", "Local", "Subsequence"] for row in rows],
        columns=["ID", "Category", "Dimensionality", "Locality", "Length"],
    ).drop_duplicates("ID")
    return validate_labels(labels, metadata)


def test_group_binary_events_matches_official_timeeval_run_semantics() -> None:
    predictions = pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(
                [
                    "2024-01-01T00:00:00",
                    "2024-01-01T00:01:00",
                    "2024-01-01T00:02:00",
                    "2024-01-01T00:03:00",
                    "2024-01-01T00:04:00",
                ]
            ),
            "Score": [0, 1, 1, 0, 1],
        }
    )

    events = group_binary_events(predictions)

    assert events == [
        {
            "start_time": pd.Timestamp("2024-01-01T00:01:00"),
            "end_time": pd.Timestamp("2024-01-01T00:03:00"),
            "end_inclusive": False,
        },
        {
            "start_time": pd.Timestamp("2024-01-01T00:04:00"),
            "end_time": pd.Timestamp("2024-01-01T00:04:00"),
            "end_inclusive": True,
        },
    ]


def test_score_event_wise_detects_hit_and_penalises_false_alarm() -> None:
    timestamps = pd.to_datetime([f"2024-01-01T00:0{m}:00" for m in range(6)])
    labels = _validated_labels(
        [["A-1", "ch1", "2024-01-01T00:01:00", "2024-01-01T00:02:00", "Anomaly"]]
    )
    predictions = {"ch1": pd.DataFrame({"Timestamp": timestamps, "Score": [0, 1, 0, 0, 1, 0]})}
    inputs = build_metric_inputs(labels, predictions)

    scores = score_event_wise(inputs, beta=0.5)

    assert scores["total_events"] == 1
    assert scores["detected_events"] == 1
    assert scores["predicted_alarms"] == 2
    assert scores["true_alarms"] == 1
    assert scores["false_alarms"] == 1
    assert scores["event_wise_recall"] == 1.0
    assert scores["event_wise_precision"] == 0.5
    # F0.5 weights precision above recall: (1.25 * 0.5 * 1.0) / (0.25*0.5 + 1.0)
    assert round(scores["event_wise_fbeta"], 6) == round(0.625 / 1.125, 6)


def test_score_event_wise_reports_missed_event() -> None:
    timestamps = pd.to_datetime(
        ["2024-01-01T00:00:00", "2024-01-01T00:01:00", "2024-01-01T00:02:00"]
    )
    labels = _validated_labels(
        [["A-1", "ch1", "2024-01-01T00:02:00", "2024-01-01T00:02:00", "Anomaly"]]
    )
    predictions = {"ch1": pd.DataFrame({"Timestamp": timestamps, "Score": [0, 0, 0]})}
    inputs = build_metric_inputs(labels, predictions)

    scores = score_event_wise(inputs)

    assert scores["detected_events"] == 0
    assert scores["missed_events"] == 1
    assert scores["event_wise_recall"] == 0.0
    assert scores["event_wise_fbeta"] == 0.0


def test_score_event_wise_excludes_requested_categories() -> None:
    timestamps = pd.to_datetime(
        ["2024-01-01T00:00:00", "2024-01-01T00:01:00", "2024-01-01T00:02:00"]
    )
    labels = _validated_labels(
        [
            ["A-1", "ch1", "2024-01-01T00:00:00", "2024-01-01T00:00:00", "Anomaly"],
            ["G-1", "ch1", "2024-01-01T00:02:00", "2024-01-01T00:02:00", "Communication Gap"],
        ]
    )
    predictions = {"ch1": pd.DataFrame({"Timestamp": timestamps, "Score": [1, 0, 0]})}
    inputs = build_metric_inputs(labels, predictions)

    scores = score_event_wise(inputs, exclude_categories=("Communication Gap",))

    assert scores["total_events"] == 1
    assert scores["excluded_categories"] == ["Communication Gap"]
    assert scores["detected_events"] == 1


def test_score_event_wise_rejects_exclusions_without_category_metadata() -> None:
    timestamps = pd.to_datetime(["2024-01-01T00:00:00", "2024-01-01T00:01:00"])
    labels = _validated_labels(
        [["A-1", "ch1", "2024-01-01T00:00:00", "2024-01-01T00:00:00", "Anomaly"]],
        with_metadata=False,
    )
    predictions = {"ch1": pd.DataFrame({"Timestamp": timestamps, "Score": [1, 0]})}
    inputs = build_metric_inputs(labels, predictions)

    with pytest.raises(SchemaError, match="no 'Category' metadata"):
        score_event_wise(inputs, exclude_categories=("Communication Gap",))


def test_score_event_wise_works_without_metadata_when_no_exclusions() -> None:
    timestamps = pd.to_datetime(["2024-01-01T00:00:00", "2024-01-01T00:01:00"])
    labels = _validated_labels(
        [["A-1", "ch1", "2024-01-01T00:00:00", "2024-01-01T00:00:00", "Anomaly"]],
        with_metadata=False,
    )
    predictions = {"ch1": pd.DataFrame({"Timestamp": timestamps, "Score": [1, 0]})}
    inputs = build_metric_inputs(labels, predictions)

    scores = score_event_wise(inputs)

    assert scores["detected_events"] == 1
    assert scores["per_event"][0]["category"] is None


def test_score_event_wise_rejects_non_positive_beta() -> None:
    timestamps = pd.to_datetime(["2024-01-01T00:00:00"])
    labels = _validated_labels(
        [["A-1", "ch1", "2024-01-01T00:00:00", "2024-01-01T00:00:00", "Anomaly"]]
    )
    predictions = {"ch1": pd.DataFrame({"Timestamp": timestamps, "Score": [1]})}
    inputs = build_metric_inputs(labels, predictions)

    with pytest.raises(SchemaError, match="beta must be positive"):
        score_event_wise(inputs, beta=0.0)
