from __future__ import annotations

import pandas as pd
import pytest

from telemeval import (
    AlignmentError,
    BinaryDomainError,
    IntervalError,
    MonotonicityError,
    SchemaError,
    TypeJoinError,
    WindowLeakageError,
    assert_window_consistent,
    build_metric_inputs,
    intervals_to_mask,
    validate_labels,
    validate_predictions,
)


def _labels_frame(rows: list[list[object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["ID", "Channel", "StartTime", "EndTime"])


def _metadata_frame(rows: list[list[object]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["ID", "Category", "Dimensionality", "Locality", "Length"]
    )


def test_validate_labels_joins_event_metadata() -> None:
    labels = _labels_frame(
        [
            ["A-1", "ch1", "2024-01-01T00:00:00Z", "2024-01-01T00:05:00Z"],
            ["A-1", "ch2", "2024-01-01T00:02:00Z", "2024-01-01T00:05:00Z"],
            ["R-2", "ch1", "2024-01-01T00:07:00Z", "2024-01-01T00:07:00Z"],
        ]
    )
    metadata = _metadata_frame(
        [
            ["A-1", "Anomaly", "Multivariate", "Global", "Subsequence"],
            ["R-2", "Rare Event", "Univariate", "Local", "Point"],
        ]
    )

    validated = validate_labels(labels, metadata)

    assert validated.columns.tolist() == [
        "ID",
        "Channel",
        "StartTime",
        "EndTime",
        "Category",
        "Dimensionality",
        "Locality",
        "Length",
    ]
    assert validated.loc[validated["ID"] == "A-1", "Category"].tolist() == [
        "Anomaly",
        "Anomaly",
    ]
    assert validated.loc[validated["ID"] == "R-2", "Length"].item() == "Point"
    assert validated["StartTime"].dt.tz is None
    assert validated["EndTime"].dt.tz is None


def test_validate_labels_rejects_missing_metadata_ids() -> None:
    labels = _labels_frame(
        [["MISSING", "ch1", "2024-01-01T00:00:00", "2024-01-01T00:05:00"]]
    )
    metadata = _metadata_frame(
        [["A-1", "Anomaly", "Multivariate", "Global", "Subsequence"]]
    )

    with pytest.raises(TypeJoinError, match="missing event metadata rows"):
        validate_labels(labels, metadata)


def test_validate_labels_rejects_duplicate_metadata_ids() -> None:
    labels = _labels_frame(
        [["A-1", "ch1", "2024-01-01T00:00:00", "2024-01-01T00:05:00"]]
    )
    metadata = _metadata_frame(
        [
            ["A-1", "Anomaly", "Multivariate", "Global", "Subsequence"],
            ["A-1", "Anomaly", "Univariate", "Local", "Point"],
        ]
    )

    with pytest.raises(TypeJoinError, match="duplicate ID rows"):
        validate_labels(labels, metadata)


def test_validate_labels_rejects_reversed_intervals() -> None:
    labels = _labels_frame(
        [["A-1", "ch1", "2024-01-01T00:05:00", "2024-01-01T00:00:00"]]
    )

    with pytest.raises(IntervalError, match="StartTime values after EndTime"):
        validate_labels(labels)


def test_validate_labels_rejects_missing_columns() -> None:
    with pytest.raises(SchemaError, match="missing required column"):
        validate_labels(pd.DataFrame({"ID": ["A-1"]}))


def test_validate_predictions_accepts_binary_masks() -> None:
    timestamps = pd.to_datetime(["2024-01-01T00:00:00", "2024-01-01T00:01:00"])
    validated = validate_predictions(
        {"ch1": pd.DataFrame({"Timestamp": timestamps, "Score": [0, 1]})}
    )

    assert validated["ch1"]["Score"].dtype.name == "uint8"
    assert validated["ch1"]["Score"].tolist() == [0, 1]


def test_validate_predictions_binarizes_scores_with_threshold() -> None:
    timestamps = pd.to_datetime(
        ["2024-01-01T00:00:00", "2024-01-01T00:01:00", "2024-01-01T00:02:00"]
    )
    validated = validate_predictions(
        {"ch1": pd.DataFrame({"Timestamp": timestamps, "Score": [0.2, 5.7, 4.9]})},
        threshold=5.0,
    )

    assert validated["ch1"]["Score"].tolist() == [0, 1, 0]


def test_validate_predictions_rejects_non_binary_without_threshold() -> None:
    timestamps = pd.to_datetime(["2024-01-01T00:00:00", "2024-01-01T00:01:00"])

    with pytest.raises(BinaryDomainError, match="pass threshold="):
        validate_predictions(
            {"ch1": pd.DataFrame({"Timestamp": timestamps, "Score": [0, 0.7]})}
        )


def test_validate_predictions_rejects_misaligned_channels() -> None:
    with pytest.raises(AlignmentError, match="align across channels"):
        validate_predictions(
            {
                "ch1": pd.DataFrame(
                    {
                        "Timestamp": pd.to_datetime(
                            ["2024-01-01T00:00:00", "2024-01-01T00:01:00"]
                        ),
                        "Score": [0, 1],
                    }
                ),
                "ch2": pd.DataFrame(
                    {
                        "Timestamp": pd.to_datetime(
                            ["2024-01-01T00:00:00", "2024-01-01T00:02:00"]
                        ),
                        "Score": [0, 1],
                    }
                ),
            }
        )


def test_validate_predictions_rejects_duplicate_and_unsorted_timestamps() -> None:
    with pytest.raises(MonotonicityError, match="duplicate timestamps"):
        validate_predictions(
            {
                "ch1": pd.DataFrame(
                    {
                        "Timestamp": pd.to_datetime(
                            ["2024-01-01T00:00:00", "2024-01-01T00:00:00"]
                        ),
                        "Score": [0, 1],
                    }
                )
            }
        )
    with pytest.raises(MonotonicityError, match="must be increasing"):
        validate_predictions(
            {
                "ch1": pd.DataFrame(
                    {
                        "Timestamp": pd.to_datetime(
                            ["2024-01-01T00:01:00", "2024-01-01T00:00:00"]
                        ),
                        "Score": [0, 1],
                    }
                )
            }
        )


def test_validate_predictions_rejects_empty_channel() -> None:
    with pytest.raises(SchemaError, match="at least one row"):
        validate_predictions({"ch1": pd.DataFrame({"Timestamp": [], "Score": []})})


def test_intervals_to_mask_marks_inclusive_bounds() -> None:
    timestamps = pd.to_datetime(
        [f"2024-01-01T00:0{minute}:00" for minute in range(6)]
    )
    intervals = pd.DataFrame(
        {
            "StartTime": ["2024-01-01T00:01:00"],
            "EndTime": ["2024-01-01T00:03:00"],
        }
    )

    mask = intervals_to_mask(intervals, timestamps)

    assert mask["Score"].tolist() == [0, 1, 1, 1, 0, 0]


def test_window_guard_raises_on_out_of_window_events() -> None:
    labels = _labels_frame(
        [
            ["train-only", "ch1", "2024-01-01T00:00:00", "2024-01-01T00:02:00"],
            ["spans", "ch1", "2024-01-01T00:04:00", "2024-01-01T00:06:00"],
            ["inside", "ch1", "2024-01-01T00:08:00", "2024-01-01T00:09:00"],
        ]
    )
    labels = validate_labels(labels)

    with pytest.raises(WindowLeakageError, match="train-only"):
        assert_window_consistent(
            labels,
            pd.Timestamp("2024-01-01T00:05:00"),
            pd.Timestamp("2024-01-01T00:10:00"),
        )


def test_window_guard_clips_only_when_explicitly_requested() -> None:
    labels = validate_labels(
        _labels_frame(
            [
                ["train-only", "ch1", "2024-01-01T00:00:00", "2024-01-01T00:02:00"],
                ["spans", "ch1", "2024-01-01T00:04:00", "2024-01-01T00:06:00"],
                ["inside", "ch1", "2024-01-01T00:08:00", "2024-01-01T00:09:00"],
            ]
        )
    )

    clipped = assert_window_consistent(
        labels,
        pd.Timestamp("2024-01-01T00:05:00"),
        pd.Timestamp("2024-01-01T00:10:00"),
        clip_to_window=True,
    )

    assert sorted(clipped["ID"].unique()) == ["inside", "spans"]


def test_window_guard_passes_consistent_labels_through() -> None:
    labels = validate_labels(
        _labels_frame(
            [["inside", "ch1", "2024-01-01T00:08:00", "2024-01-01T00:09:00"]]
        )
    )

    result = assert_window_consistent(
        labels,
        pd.Timestamp("2024-01-01T00:05:00"),
        pd.Timestamp("2024-01-01T00:10:00"),
    )

    assert result["ID"].tolist() == ["inside"]


def test_build_metric_inputs_returns_global_and_channel_shapes() -> None:
    labels = validate_labels(
        _labels_frame(
            [
                ["A-1", "ch1", "2024-01-01T00:01:00", "2024-01-01T00:02:00"],
                ["A-1", "ch2", "2024-01-01T00:01:00", "2024-01-01T00:02:00"],
            ]
        )
    )
    timestamps = pd.to_datetime(
        ["2024-01-01T00:00:00", "2024-01-01T00:01:00", "2024-01-01T00:02:00"]
    )
    predictions = {
        "ch1": pd.DataFrame({"Timestamp": timestamps, "Score": [0, 1, 0]}),
        "ch2": pd.DataFrame({"Timestamp": timestamps, "Score": [0, 0, 1]}),
    }

    inputs = build_metric_inputs(labels, predictions)

    assert "Channel" not in inputs["global_labels"].columns
    assert inputs["global_predictions"]["Score"].tolist() == [0, 1, 1]
    assert inputs["target_channels"] == ["ch1", "ch2"]
    assert inputs["channel_predictions"]["ch1"]["Score"].tolist() == [0, 1, 0]
