# Usage

## The one call

```python
import pandas as pd
from telemeval import evaluate

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
print(result.metrics["event_wise"]["event_wise_fbeta"])
print(result.metrics["affiliation"]["affiliation_fbeta"])
result.save(json_path="report.json", markdown_path="report.md")
```

`evaluate()` validates everything first. Malformed inputs raise typed errors
(`telemeval.ContractError` subclasses) instead of producing a wrong number.

## The leakage guard (on by default)

If any labelled event lies entirely outside the prediction window,
`evaluate()` raises `WindowLeakageError` — such events can never be detected,
and silently counting them corrupts recall (a real bug class: the origin
project measured recall 0.24 instead of the true 0.42 this way). To evaluate
a test split against full-mission labels, opt in explicitly:

```python
result = evaluate(labels, predictions, clip_to_window=True)
print(result.config["events_clipped_by_window_guard"])  # recorded, not silent
```

## Continuous scores

Thresholds are inputs, never fitted here:

```python
result = evaluate(labels, scored_predictions, threshold=5.0)
```

## ESA-ADB files

```python
from telemeval.formats.esa_adb import ESA_ADB_DEFAULT_EXCLUDE_CATEGORIES, read_labels

labels = read_labels("Mission1/labels.csv", "Mission1/anomaly_types.csv")
result = evaluate(
    labels,
    predictions,
    exclude_categories=ESA_ADB_DEFAULT_EXCLUDE_CATEGORIES,
)
```

## TimeEval-format research datasets

```python
from telemeval.formats.timeeval import read_dataset

dataset = read_dataset("kpi/test.csv")          # timestamp, value(s), is_anomaly
detections = my_detector(dataset["values"])     # your detector, any library
predictions = {"series": pd.DataFrame({
    "Timestamp": dataset["timestamps"], "Score": detections,
})}
result = evaluate(dataset["labels"], predictions)
```

## Parquet input

telemeval consumes pandas DataFrames, so any storage pandas reads works
directly — including Parquet extracts from a telemetry store:

```python
frame = pd.read_parquet("mission_predictions.parquet")
predictions = {ch: g[["Timestamp", "Score"]] for ch, g in frame.groupby("Channel")}
```

(Parquet is a convenient interchange format; telemeval does not claim it as a
satellite-ops standard.)

## scikit-learn-style wrappers

```python
from telemeval.sklearn import affiliation_fbeta_score, event_wise_fbeta_score

event_wise_fbeta_score(y_true, y_pred, beta=0.5)
affiliation_fbeta_score(y_true, y_pred)
```

These follow the `score_func(y_true, y_pred)` convention and work with
`sklearn.metrics.make_scorer`.

## Extending: the metric registry

```python
from telemeval import register_metric

def my_metric(metric_inputs, *, beta=0.5, exclude_categories=()):
    ...
    return {"my_score": 0.9}

register_metric("my_metric", my_metric)
result = evaluate(labels, predictions, metrics=("event_wise", "my_metric"))
```
