"""Guard that the README quick-start example stays runnable and truthful."""

from __future__ import annotations

import pandas as pd

from telemeval import evaluate


def test_readme_quick_start_example(tmp_path) -> None:
    labels = pd.DataFrame(
        {
            "ID": ["anomaly_1"],
            "Channel": ["channel_41"],
            "StartTime": ["2024-01-01T00:02:00"],
            "EndTime": ["2024-01-01T00:03:00"],
        }
    )
    timestamps = pd.date_range("2024-01-01", periods=6, freq="1min")
    predictions = {
        "channel_41": pd.DataFrame({"Timestamp": timestamps, "Score": [0, 0, 1, 1, 0, 0]})
    }

    result = evaluate(labels, predictions, dataset="my-mission")

    assert result.metrics["event_wise"]["event_wise_fbeta"] == 1.0
    assert result.metrics["affiliation"]["affiliation_fbeta"] == 1.0
    result.save(
        json_path=str(tmp_path / "report.json"),
        markdown_path=str(tmp_path / "report.md"),
    )
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.md").exists()
