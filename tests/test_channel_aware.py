"""Channel/subsystem-aware F-score tests.

The four parity fixtures are ported VERBATIM from the ESA-ADB reference test
suite (tests/metrics/test_metrics.py::TestRankingMetrics.test_from_report),
including their exact expected values. The reference feeds change-point
series; here they are sampled onto a dense hourly grid (equivalent at grid
resolution).
"""

from __future__ import annotations

import pandas as pd
import pytest

from telemeval import build_metric_inputs, evaluate, validate_labels
from telemeval.errors import SchemaError
from telemeval.formats.esa_adb import read_channels
from telemeval.metrics.channel_aware import score_channel_aware

GRID = pd.date_range("2015-01-01", "2015-01-09", freq="1h")


def _mask_from_changes(changes: list[tuple[str, int]]) -> list[int]:
    """Sample a change-point series (value holds until next change) on GRID."""

    points = [(pd.Timestamp(t), v) for t, v in changes]
    mask = []
    for t in GRID:
        value = 0
        for ct, cv in points:
            if ct <= t:
                value = cv
            else:
                break
        mask.append(value)
    return mask


def _inputs(label_rows: list[list[str]], preds: dict[str, list[tuple[str, int]]]):
    labels = validate_labels(
        pd.DataFrame(label_rows, columns=["ID", "Channel", "StartTime", "EndTime"])
    )
    predictions = {
        channel: pd.DataFrame({"Timestamp": GRID, "Score": _mask_from_changes(changes)})
        for channel, changes in preds.items()
    }
    return build_metric_inputs(labels, predictions)


def _assert_scores(result: dict, expected: dict) -> None:
    for key, value in expected.items():
        assert result[key] == pytest.approx(value), f"{key}: {result[key]} != {value}"


# --- reference parity fixtures ---


def test_reference_case_1_everything_correct() -> None:
    inputs = _inputs(
        [
            ["id_0", "ch1", "2015-01-01", "2015-01-04"],
            ["id_0", "ch1", "2015-01-05", "2015-01-08"],
            ["id_0", "ch2", "2015-01-02", "2015-01-04"],
            ["id_0", "ch3", "2015-01-02", "2015-01-03"],
            ["id_0", "ch3", "2015-01-06", "2015-01-07"],
        ],
        {
            "ch1": [("2015-01-01", 1), ("2015-01-04", 0)],
            "ch2": [("2015-01-01", 0), ("2015-01-03", 1), ("2015-01-04", 0)],
            "ch3": [
                ("2015-01-01", 0),
                ("2015-01-02", 1),
                ("2015-01-05", 0),
                ("2015-01-06", 1),
                ("2015-01-08", 0),
            ],
        },
    )

    result = score_channel_aware(
        inputs, beta=0.5, subsystems_mapping={"s1": ["ch1"], "s2": ["ch2", "ch3"]}
    )

    _assert_scores(
        result,
        {
            "channel_precision": 1.0,
            "channel_recall": 1.0,
            "channel_fbeta": 1.0,
            "subsystem_precision": 1.0,
            "subsystem_recall": 1.0,
            "subsystem_fbeta": 1.0,
        },
    )


def test_reference_case_2_missed_channel() -> None:
    inputs = _inputs(
        [
            ["id_0", "ch1", "2015-01-01", "2015-01-04"],
            ["id_0", "ch1", "2015-01-05", "2015-01-08"],
            ["id_0", "ch2", "2015-01-02", "2015-01-04"],
            ["id_0", "ch3", "2015-01-05 12:00", "2015-01-05 14:00"],
        ],
        {
            "ch1": [("2015-01-01", 1), ("2015-01-04", 0)],
            "ch2": [("2015-01-01", 0)],
            "ch3": [
                ("2015-01-01", 0),
                ("2015-01-02", 1),
                ("2015-01-05", 0),
                ("2015-01-06", 1),
                ("2015-01-08", 0),
            ],
        },
    )

    result = score_channel_aware(
        inputs, beta=0.5, subsystems_mapping={"s1": ["ch1"], "s2": ["ch2", "ch3"]}
    )

    _assert_scores(
        result,
        {
            "channel_precision": 1.0,
            "channel_recall": 0.6666666666666666,
            "channel_fbeta": 0.9090909090909091,
            "subsystem_precision": 1.0,
            "subsystem_recall": 1.0,
            "subsystem_fbeta": 1.0,
        },
    )


_CASE_3_LABELS = [
    ["id_0", "ch1", "2015-01-01", "2015-01-06"],
    ["id_0", "ch2", "2015-01-01", "2015-01-03"],
    ["id_1", "ch1", "2015-01-05", "2015-01-09"],
    ["id_1", "ch3", "2015-01-04", "2015-01-09"],
    ["id_1", "ch4", "2015-01-07", "2015-01-09"],
]
_CASE_3_SUBSYSTEMS = {"s1": ["ch1", "ch2"], "s2": ["ch3", "ch4"]}


def test_reference_case_3_cross_event_alarm_is_excused() -> None:
    # ch3 alarms during id_0 (where it is unlabelled) but that same alarm is a
    # true detection of id_1 on ch3 — excused, so precision stays 1.0.
    inputs = _inputs(
        _CASE_3_LABELS,
        {
            "ch1": [("2015-01-01", 1), ("2015-01-05 12:00", 0)],
            "ch2": [("2015-01-01", 0)],
            "ch3": [("2015-01-01", 0), ("2015-01-04", 1), ("2015-01-08", 0)],
            "ch4": [("2015-01-01", 0), ("2015-01-08", 1)],
        },
    )

    result = score_channel_aware(inputs, beta=0.5, subsystems_mapping=_CASE_3_SUBSYSTEMS)

    _assert_scores(
        result,
        {
            "channel_precision": 1.0,
            "channel_recall": 0.75,
            "channel_fbeta": 0.9166666666666667,
            "subsystem_precision": 1.0,
            "subsystem_recall": 1.0,
            "subsystem_fbeta": 1.0,
        },
    )


def test_reference_case_3_variant_always_on_channel_is_a_false_positive() -> None:
    # ch4 alarming the entire time: its detection inside id_0's interval does
    # NOT reach id_1's ch4 label, so it is a genuine false positive for id_0.
    inputs = _inputs(
        _CASE_3_LABELS,
        {
            "ch1": [("2015-01-01", 1), ("2015-01-05 12:00", 0)],
            "ch2": [("2015-01-01", 0)],
            "ch3": [("2015-01-01", 0), ("2015-01-04", 1), ("2015-01-08", 0)],
            "ch4": [("2015-01-01", 1)],
        },
    )

    result = score_channel_aware(inputs, beta=0.5, subsystems_mapping=_CASE_3_SUBSYSTEMS)

    _assert_scores(
        result,
        {
            "channel_precision": 0.75,
            "channel_recall": 0.75,
            "channel_fbeta": 0.75,
            "subsystem_precision": 0.75,
            "subsystem_recall": 1.0,
            "subsystem_fbeta": 0.7777777777777778,
        },
    )


# --- telemeval-specific behavior ---

_SIMPLE_LABELS = [["A-1", "ch1", "2015-01-02", "2015-01-03"]]
_SIMPLE_PREDS = {
    "ch1": [("2015-01-01", 0), ("2015-01-02", 1), ("2015-01-03", 0)],
    "ch2": [("2015-01-01", 0)],
}


def test_subsystem_scores_are_none_without_mapping() -> None:
    inputs = _inputs(_SIMPLE_LABELS, _SIMPLE_PREDS)

    result = score_channel_aware(inputs)

    assert result["channel_precision"] == 1.0
    assert result["subsystem_precision"] is None
    assert result["subsystem_fbeta"] is None


def test_no_events_yield_none_scores() -> None:
    inputs = _inputs(_SIMPLE_LABELS, _SIMPLE_PREDS)
    inputs = {**inputs, "channel_labels": inputs["channel_labels"].iloc[0:0]}

    result = score_channel_aware(inputs)

    assert result["total_events"] == 0
    assert result["channel_precision"] is None


def test_point_anomaly_widened_like_reference() -> None:
    inputs = _inputs(
        [["P-1", "ch1", "2015-01-02", "2015-01-02"]],
        {"ch1": [("2015-01-01", 0), ("2015-01-02", 1), ("2015-01-02 01:00", 0)]},
    )

    result = score_channel_aware(inputs)

    assert result["channel_recall"] == 1.0


def test_exclusions_require_category_metadata() -> None:
    inputs = _inputs(_SIMPLE_LABELS, _SIMPLE_PREDS)

    with pytest.raises(SchemaError, match="no 'Category' metadata"):
        score_channel_aware(inputs, exclude_categories=("Communication Gap",))


def test_runs_through_evaluate_with_metric_options() -> None:
    labels = pd.DataFrame(_SIMPLE_LABELS, columns=["ID", "Channel", "StartTime", "EndTime"])
    predictions = {
        channel: pd.DataFrame({"Timestamp": GRID, "Score": _mask_from_changes(changes)})
        for channel, changes in _SIMPLE_PREDS.items()
    }

    result = evaluate(
        labels,
        predictions,
        metrics=("channel_aware",),
        metric_options={"channel_aware": {"subsystems_mapping": {"s1": ["ch1", "ch2"]}}},
    )

    assert result.metrics["channel_aware"]["subsystem_fbeta"] == 1.0


def test_metric_options_for_unrun_metric_raise() -> None:
    labels = pd.DataFrame(_SIMPLE_LABELS, columns=["ID", "Channel", "StartTime", "EndTime"])
    predictions = {
        channel: pd.DataFrame({"Timestamp": GRID, "Score": _mask_from_changes(changes)})
        for channel, changes in _SIMPLE_PREDS.items()
    }

    with pytest.raises(SchemaError, match="not being run"):
        evaluate(
            labels,
            predictions,
            metrics=("event_wise",),
            metric_options={"channel_aware": {"subsystems_mapping": {}}},
        )


def test_read_channels_builds_subsystem_mapping(tmp_path) -> None:
    path = tmp_path / "channels.csv"
    path.write_text(
        "\n".join(
            [
                "Channel,Subsystem,Physical Unit,Group,Target",
                "channel_41,subsystem_5,physical_unit_4,8,YES",
                "channel_42,subsystem_5,physical_unit_4,8,YES",
                "channel_1,subsystem_1,physical_unit_1,1,NO",
            ]
        ),
        encoding="utf-8",
    )

    channels = read_channels(path)

    assert channels["subsystems_mapping"] == {
        "subsystem_5": ["channel_41", "channel_42"],
        "subsystem_1": ["channel_1"],
    }
    assert channels["target_channels"] == ["channel_41", "channel_42"]
