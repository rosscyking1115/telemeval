from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from telemeval import SchemaError, build_metric_inputs, validate_labels
from telemeval.metrics._affiliation_vendor.generics import convert_vector_to_events
from telemeval.metrics._affiliation_vendor.metrics import pr_from_events
from telemeval.metrics.affiliation import score_affiliation


def _inputs(label_rows: list[list[object]], scores: list[int], with_category: bool = False):
    labels = pd.DataFrame(
        [row[:4] for row in label_rows], columns=["ID", "Channel", "StartTime", "EndTime"]
    )
    metadata = None
    if with_category:
        metadata = pd.DataFrame(
            [[row[0], row[4], "Univariate", "Local", "Subsequence"] for row in label_rows],
            columns=["ID", "Category", "Dimensionality", "Locality", "Length"],
        ).drop_duplicates("ID")
    validated = validate_labels(labels, metadata)
    timestamps = pd.to_datetime(
        [pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=m) for m in range(len(scores))]
    )
    predictions = {"ch1": pd.DataFrame({"Timestamp": timestamps, "Score": scores})}
    return build_metric_inputs(validated, predictions)


def test_perfect_prediction_scores_one() -> None:
    inputs = _inputs(
        [["A-1", "ch1", "2024-01-01T00:02:00", "2024-01-01T00:04:00"]],
        [0, 0, 1, 1, 1, 0, 0],
    )

    scores = score_affiliation(inputs)

    assert scores["affiliation_precision"] == 1.0
    assert scores["affiliation_recall"] == 1.0
    assert scores["affiliation_fbeta"] == 1.0
    assert scores["gt_events_merged"] == 1
    assert scores["predicted_events"] == 1


def test_wrapper_matches_direct_reference_call() -> None:
    gt_vector = [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0]
    pred_vector = [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0]
    inputs = _inputs(
        [
            ["A-1", "ch1", "2024-01-01T00:02:00", "2024-01-01T00:04:00"],
            ["A-2", "ch1", "2024-01-01T00:08:00", "2024-01-01T00:09:00"],
        ],
        pred_vector,
    )

    wrapped = score_affiliation(inputs)
    reference = pr_from_events(
        convert_vector_to_events(pred_vector),
        convert_vector_to_events(gt_vector),
        (0, len(gt_vector)),
    )

    assert wrapped["affiliation_precision"] == pytest.approx(reference["precision"])
    assert wrapped["affiliation_recall"] == pytest.approx(reference["recall"])


def test_off_grid_event_raises_instead_of_vanishing() -> None:
    # The labelled event sits entirely between two grid samples.
    inputs = _inputs(
        [["ghost", "ch1", "2024-01-01T00:02:10", "2024-01-01T00:02:50"]],
        [0, 0, 0, 0],
    )

    with pytest.raises(SchemaError, match="ghost"):
        score_affiliation(inputs)


def test_all_events_excluded_raises() -> None:
    inputs = _inputs(
        [["G-1", "ch1", "2024-01-01T00:01:00", "2024-01-01T00:02:00", "Communication Gap"]],
        [0, 1, 1, 0],
        with_category=True,
    )

    with pytest.raises(SchemaError, match="at least one ground-truth event"):
        score_affiliation(inputs, exclude_categories=("Communication Gap",))


def test_empty_predictions_yield_none_precision_and_zero_fbeta() -> None:
    inputs = _inputs(
        [["A-1", "ch1", "2024-01-01T00:01:00", "2024-01-01T00:02:00"]],
        [0, 0, 0, 0],
    )

    scores = score_affiliation(inputs)

    assert scores["affiliation_precision"] is None
    assert scores["affiliation_fbeta"] == 0.0
    assert scores["predicted_events"] == 0


@settings(max_examples=25, deadline=None)
@given(
    pred=st.lists(st.integers(min_value=0, max_value=1), min_size=8, max_size=40),
    gt_start=st.integers(min_value=0, max_value=5),
    gt_length=st.integers(min_value=1, max_value=4),
)
def test_scores_stay_in_unit_interval(pred: list[int], gt_start: int, gt_length: int) -> None:
    base = pd.Timestamp("2024-01-01")
    start = base + pd.Timedelta(minutes=gt_start)
    end = base + pd.Timedelta(minutes=gt_start + gt_length - 1)
    inputs = _inputs(
        [["A-1", "ch1", str(start), str(end)]],
        pred,
    )

    scores = score_affiliation(inputs)

    for key in ("affiliation_precision", "affiliation_recall", "affiliation_fbeta"):
        value = scores[key]
        if value is not None:
            assert -1e-9 <= value <= 1.0 + 1e-9, f"{key}={value} out of [0,1]"


def test_overlapping_multichannel_rows_merge_into_one_event() -> None:
    # The same event ID spans two channels with overlapping intervals — the
    # merged global timeline must contain a single ground-truth event.
    labels = pd.DataFrame(
        [
            ["A-1", "ch1", "2024-01-01T00:02:00", "2024-01-01T00:04:00"],
            ["A-1", "ch2", "2024-01-01T00:03:00", "2024-01-01T00:05:00"],
        ],
        columns=["ID", "Channel", "StartTime", "EndTime"],
    )
    validated = validate_labels(labels)
    timestamps = pd.to_datetime(
        [pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=m) for m in range(8)]
    )
    predictions = {
        "ch1": pd.DataFrame({"Timestamp": timestamps, "Score": [0, 0, 1, 1, 0, 0, 0, 0]}),
        "ch2": pd.DataFrame({"Timestamp": timestamps, "Score": [0, 0, 0, 0, 1, 1, 0, 0]}),
    }
    inputs = build_metric_inputs(validated, predictions)

    scores = score_affiliation(inputs)

    assert scores["gt_events_merged"] == 1
    assert scores["affiliation_recall"] == 1.0


def _mask_from_inputs(inputs) -> np.ndarray:
    from telemeval.metrics.affiliation import _labels_to_mask

    grid = pd.DatetimeIndex(inputs["global_predictions"]["Timestamp"]).to_numpy(
        "datetime64[ns]"
    )
    mask, off_grid = _labels_to_mask(inputs["global_labels"], grid)
    assert not off_grid
    return mask


def test_label_mask_uses_inclusive_bounds() -> None:
    inputs = _inputs(
        [["A-1", "ch1", "2024-01-01T00:01:00", "2024-01-01T00:03:00"]],
        [0, 0, 0, 0, 0, 0],
    )

    assert _mask_from_inputs(inputs).tolist() == [0, 1, 1, 1, 0, 0]
