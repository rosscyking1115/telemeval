from __future__ import annotations

import json

import pandas as pd

from telemeval import (
    build_metric_inputs,
    build_report,
    render_markdown,
    score_event_wise,
    validate_labels,
    write_report,
)
from telemeval.formats.esa_adb import (
    ESA_ADB_DEFAULT_EXCLUDE_CATEGORIES,
    read_labels,
)


def _evaluated_report() -> dict:
    labels = validate_labels(
        pd.DataFrame(
            [["A-1", "ch1", "2024-01-01T00:01:00", "2024-01-01T00:01:00"]],
            columns=["ID", "Channel", "StartTime", "EndTime"],
        )
    )
    timestamps = pd.to_datetime([f"2024-01-01T00:0{m}:00" for m in range(5)])
    predictions = {"ch1": pd.DataFrame({"Timestamp": timestamps, "Score": [0, 1, 0, 0, 0]})}
    inputs = build_metric_inputs(labels, predictions)
    scores = score_event_wise(inputs)
    return build_report(
        {"event_wise": scores},
        target_channels=inputs["target_channels"],
        dataset="fixture",
        scope_notes=["Event-wise detection only; no timing or proximity metrics."],
        config={"beta": 0.5},
    )


def test_report_serializes_timestamps_and_carries_caveats() -> None:
    report = _evaluated_report()

    assert report["schema_version"] == "telemeval/evaluation-report/v1"
    assert any("Event-wise detection only" in note for note in report["scope_notes"])
    per_event = report["metrics"]["event_wise"]["per_event"]
    assert per_event[0]["start_time"] == "2024-01-01T00:01:00"

    # Deterministic: JSON round-trips identically.
    payload = json.dumps(report, sort_keys=True)
    assert json.dumps(json.loads(payload), sort_keys=True) == payload


def test_report_markdown_renders_metrics_and_detail() -> None:
    report = _evaluated_report()

    markdown = render_markdown(report)

    assert "# telemeval Evaluation Report — fixture" in markdown
    assert "| event_wise_fbeta | 1.000000 |" in markdown
    assert "| A-1 |" in markdown


def test_write_report_emits_json_and_markdown(tmp_path) -> None:
    report = _evaluated_report()
    json_path = tmp_path / "out" / "report.json"
    markdown_path = tmp_path / "out" / "report.md"

    write_report(report, json_path=str(json_path), markdown_path=str(markdown_path))

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["dataset"] == "fixture"
    assert markdown_path.read_text(encoding="utf-8").startswith(
        "# telemeval Evaluation Report"
    )


def test_esa_adb_read_labels_joins_anomaly_types(tmp_path) -> None:
    labels_path = tmp_path / "labels.csv"
    labels_path.write_text(
        "\n".join(
            [
                "ID,Channel,StartTime,EndTime",
                "id_1,channel_41,2004-12-01T20:42:15.429Z,2004-12-08T22:55:45.429Z",
                "id_2,channel_42,2004-12-09T00:00:00.000Z,2004-12-09T01:00:00.000Z",
            ]
        ),
        encoding="utf-8",
    )
    anomaly_types_path = tmp_path / "anomaly_types.csv"
    anomaly_types_path.write_text(
        "\n".join(
            [
                # Real ESA-ADB anomaly_types.csv carries extra columns; the
                # reader must select only the official metadata columns.
                "ID,Class,Subclass,Category,Dimensionality,Locality,Length",
                "id_1,class_6,subclass_1,Rare Event,Multivariate,Global,Subsequence",
                "id_2,class_7,subclass_1,Anomaly,Multivariate,Local,Subsequence",
            ]
        ),
        encoding="utf-8",
    )

    labels = read_labels(labels_path, anomaly_types_path)

    assert labels.columns.tolist() == [
        "ID",
        "Channel",
        "StartTime",
        "EndTime",
        "Category",
        "Dimensionality",
        "Locality",
        "Length",
    ]
    assert labels.loc[labels["ID"] == "id_1", "Category"].item() == "Rare Event"
    assert ESA_ADB_DEFAULT_EXCLUDE_CATEGORIES == ("Communication Gap",)
