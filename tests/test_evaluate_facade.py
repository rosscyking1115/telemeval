from __future__ import annotations

import json

import pandas as pd
import pytest

from telemeval import (
    SchemaError,
    WindowLeakageError,
    available_metrics,
    evaluate,
    register_metric,
)


def _labels(rows: list[list[object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["ID", "Channel", "StartTime", "EndTime"])


def _predictions(scores: list[float]) -> dict[str, pd.DataFrame]:
    timestamps = pd.to_datetime(
        [pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=m) for m in range(len(scores))]
    )
    return {"ch1": pd.DataFrame({"Timestamp": timestamps, "Score": scores})}


def test_evaluate_runs_both_default_metrics() -> None:
    result = evaluate(
        _labels([["A-1", "ch1", "2024-01-01T00:02:00", "2024-01-01T00:03:00"]]),
        _predictions([0, 0, 1, 1, 0, 0]),
        dataset="fixture",
    )

    assert set(result.metrics) == {"event_wise", "affiliation"}
    assert result.metrics["event_wise"]["event_wise_fbeta"] == 1.0
    assert result.metrics["affiliation"]["affiliation_fbeta"] == 1.0
    assert result.config["window_guard"] == "strict"


def test_evaluate_leakage_guard_is_on_by_default() -> None:
    labels = _labels(
        [
            ["out-of-window", "ch1", "2023-01-01T00:00:00", "2023-01-01T01:00:00"],
            ["in-window", "ch1", "2024-01-01T00:02:00", "2024-01-01T00:03:00"],
        ]
    )

    with pytest.raises(WindowLeakageError, match="out-of-window"):
        evaluate(labels, _predictions([0, 0, 1, 1, 0, 0]))


def test_evaluate_clip_to_window_records_clipping() -> None:
    labels = _labels(
        [
            ["out-of-window", "ch1", "2023-01-01T00:00:00", "2023-01-01T01:00:00"],
            ["in-window", "ch1", "2024-01-01T00:02:00", "2024-01-01T00:03:00"],
        ]
    )

    result = evaluate(labels, _predictions([0, 0, 1, 1, 0, 0]), clip_to_window=True)

    assert result.config["events_clipped_by_window_guard"] == 1
    assert result.metrics["event_wise"]["total_events"] == 1
    assert result.metrics["event_wise"]["event_wise_recall"] == 1.0


def test_evaluate_with_threshold_binarizes_scores() -> None:
    result = evaluate(
        _labels([["A-1", "ch1", "2024-01-01T00:02:00", "2024-01-01T00:03:00"]]),
        _predictions([0.1, 0.2, 7.5, 9.0, 0.3, 0.2]),
        threshold=5.0,
        metrics=("event_wise",),
    )

    assert result.config["threshold"] == 5.0
    assert result.metrics["event_wise"]["detected_events"] == 1


def test_evaluate_report_roundtrip(tmp_path) -> None:
    result = evaluate(
        _labels([["A-1", "ch1", "2024-01-01T00:02:00", "2024-01-01T00:03:00"]]),
        _predictions([0, 0, 1, 1, 0, 0]),
        dataset="fixture",
    )
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"

    result.save(json_path=str(json_path), markdown_path=str(markdown_path))

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["dataset"] == "fixture"
    assert "event_wise" in loaded["metrics"]
    assert any("Computed metrics" in note for note in loaded["scope_notes"])
    assert markdown_path.read_text(encoding="utf-8").startswith("# telemeval")


def test_registry_rejects_unknown_metric_and_supports_registration() -> None:
    with pytest.raises(SchemaError, match="unknown metric"):
        evaluate(
            _labels([["A-1", "ch1", "2024-01-01T00:02:00", "2024-01-01T00:03:00"]]),
            _predictions([0, 0, 1, 1, 0, 0]),
            metrics=("nope",),
        )

    def constant_metric(metric_inputs, *, beta=0.5, exclude_categories=()):
        return {"answer": 42}

    register_metric("constant_test_metric", constant_metric)
    try:
        assert "constant_test_metric" in available_metrics()
        result = evaluate(
            _labels([["A-1", "ch1", "2024-01-01T00:02:00", "2024-01-01T00:03:00"]]),
            _predictions([0, 0, 1, 1, 0, 0]),
            metrics=("constant_test_metric",),
        )
        assert result.metrics["constant_test_metric"]["answer"] == 42
    finally:
        # Keep the registry clean for other tests.
        from telemeval.registry import _REGISTRY

        _REGISTRY.pop("constant_test_metric", None)


def test_duplicate_registration_requires_overwrite() -> None:
    with pytest.raises(SchemaError, match="already registered"):
        register_metric("event_wise", lambda inputs, **kw: {})


def test_evaluate_preserves_prejoined_metadata_columns() -> None:
    # Labels that already carry Category (e.g. from formats.esa_adb.read_labels)
    # must keep it through evaluate(), so exclude_categories works.
    labels = pd.DataFrame(
        [
            ["A-1", "ch1", "2024-01-01T00:02:00", "2024-01-01T00:03:00", "Anomaly"],
            ["G-1", "ch1", "2024-01-01T00:04:00", "2024-01-01T00:04:00", "Communication Gap"],
        ],
        columns=["ID", "Channel", "StartTime", "EndTime", "Category"],
    )

    result = evaluate(
        labels,
        _predictions([0, 0, 1, 1, 0, 0]),
        metrics=("event_wise",),
        exclude_categories=("Communication Gap",),
    )

    assert result.metrics["event_wise"]["total_events"] == 1
    assert result.metrics["event_wise"]["event_wise_recall"] == 1.0
