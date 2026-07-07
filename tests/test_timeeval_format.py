from __future__ import annotations

import pandas as pd
import pytest

from telemeval import evaluate
from telemeval.errors import BinaryDomainError, SchemaError
from telemeval.formats.timeeval import read_dataset


def _write_dataset(tmp_path, rows: list[str], name: str = "data.csv"):
    path = tmp_path / name
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


def test_read_dataset_with_unix_timestamps(tmp_path) -> None:
    path = _write_dataset(
        tmp_path,
        [
            "timestamp,value,is_anomaly",
            "1700000000,0.5,0",
            "1700000060,0.7,1",
            "1700000120,0.9,1",
            "1700000180,0.4,0",
        ],
    )

    dataset = read_dataset(path)

    assert len(dataset["timestamps"]) == 4
    assert dataset["ground_truth_mask"].tolist() == [0, 1, 1, 0]
    labels = dataset["labels"]
    assert labels["ID"].tolist() == ["event_0"]
    assert labels["StartTime"].iloc[0] == pd.Timestamp("2023-11-14T22:14:20")
    assert dataset["values"].columns.tolist() == ["value"]


def test_read_dataset_with_datetime_strings_and_multiple_events(tmp_path) -> None:
    path = _write_dataset(
        tmp_path,
        [
            "timestamp,v1,v2,is_anomaly",
            "2024-01-01T00:00:00,1.0,2.0,1",
            "2024-01-01T00:01:00,1.1,2.1,0",
            "2024-01-01T00:02:00,1.2,2.2,1",
            "2024-01-01T00:03:00,1.3,2.3,1",
        ],
    )

    dataset = read_dataset(path)

    assert dataset["labels"]["ID"].tolist() == ["event_0", "event_1"]
    assert dataset["values"].columns.tolist() == ["v1", "v2"]


def test_read_dataset_feeds_evaluate_end_to_end(tmp_path) -> None:
    path = _write_dataset(
        tmp_path,
        [
            "timestamp,value,is_anomaly",
            "2024-01-01T00:00:00,0.1,0",
            "2024-01-01T00:01:00,5.0,1",
            "2024-01-01T00:02:00,5.1,1",
            "2024-01-01T00:03:00,0.2,0",
        ],
    )
    dataset = read_dataset(path)
    predictions = {
        "series": pd.DataFrame(
            {"Timestamp": dataset["timestamps"], "Score": [0, 1, 1, 0]}
        )
    }

    result = evaluate(dataset["labels"], predictions, dataset=str(path.name))

    assert result.metrics["event_wise"]["event_wise_fbeta"] == 1.0
    assert result.metrics["affiliation"]["affiliation_fbeta"] == 1.0


def test_read_dataset_rejects_missing_columns_and_non_binary_labels(tmp_path) -> None:
    missing = _write_dataset(tmp_path, ["time,value", "1,2"], name="missing.csv")
    with pytest.raises(SchemaError, match="missing required column"):
        read_dataset(missing)

    non_binary = _write_dataset(
        tmp_path,
        ["timestamp,value,is_anomaly", "1700000000,0.5,2"],
        name="nonbinary.csv",
    )
    with pytest.raises(BinaryDomainError, match="must be binary"):
        read_dataset(non_binary)
