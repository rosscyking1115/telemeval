# telemeval

[![CI](https://github.com/rosscyking1115/telemeval/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/rosscyking1115/telemeval/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/telemeval)](https://pypi.org/project/telemeval/)
[![Python](https://img.shields.io/pypi/pyversions/telemeval)](https://pypi.org/project/telemeval/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21250548.svg)](https://doi.org/10.5281/zenodo.21250548)

**Leakage-safe, event-wise and affiliation-based evaluation for
spacecraft-telemetry anomaly detection.**

> Status: v0.x — early releases; the API may still change until v1.

Time-series anomaly-detection evaluation is notoriously easy to get wrong:
point-adjusted F1 can rank random predictions above real detectors, and subtle
protocol bugs (like scoring training-window events against test-window
predictions) silently inflate or deflate results. The metrics the literature
recommends instead — corrected **event-wise** F-beta and **affiliation-based**
precision/recall — have not had a maintained, permissively-licensed,
pip-installable home.

telemeval is that home:

- **A validated ingestion contract** that raises typed, actionable errors
  instead of producing a number from a leaky or malformed evaluation —
  including a first-class train/test-window leakage guard.
- **Corrected event-wise precision / recall / F-beta** with unambiguous,
  documented overlap semantics.
- **Affiliation-based precision / recall** — the canonical reference
  implementation (Huet et al., KDD 2022, MIT) vendored, wrapped, tested, and
  maintained here.
- **ADTQC detection-timing quality** (ESA-ADB) — scores *when* each event was
  first caught, not just whether; previously available only inside ESA-ADB's
  research fork.
- **Channel- and subsystem-aware F-beta** (ESA-ADB) — did you flag the right
  *source*? Verified against the reference test suite's exact expected values.
- **Honest reports**: deterministic JSON and Markdown output stamped with
  explicit scope caveats.
- **Telemetry-aware inputs**: channel-keyed predictions (interval labels,
  binary masks, or continuous scores + threshold), an ESA-ADB-format loader,
  and a TimeEval-format reader.
- **scikit-learn-style metric wrappers** so the metrics drop into existing
  scoring code.

Core dependencies: numpy and pandas. Nothing else.

## Quick start

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
print(result.metrics["event_wise"]["event_wise_fbeta"])     # 1.0
print(result.metrics["affiliation"]["affiliation_fbeta"])   # 1.0
result.save(json_path="report.json", markdown_path="report.md")
```

See [docs/usage.md](docs/usage.md) for the leakage guard, continuous scores +
threshold, ESA-ADB and TimeEval-format loaders, parquet input, sklearn-style
wrappers, and the metric registry. See
[docs/related-work.md](docs/related-work.md) for an honest map of prior art
and when to use which tool.

## What telemeval is not

- Not a detector library (see PyOD, aeon, darts).
- Not a benchmark harness or dataset collection (see TimeEval, TSB-AD).
- Not a serving/monitoring/MLOps stack.
- Not affiliated with or endorsed by ESA; it does not redistribute the ESA
  Anomaly Dataset.

Prior art worth knowing: **TSADmetrics** (GPL-3.0, generic time-series AD
metrics), **TSB-AD** (benchmark suite), **Merlion** (point-adjusted F1),
**aeon** (range/VUS metrics). telemeval's niche is the permissive license,
the telemetry-domain contract with leakage guards, and maintained affiliation
metrics; where metrics overlap we aim to reproduce prior-art numbers.

## License

Apache-2.0. Vendored affiliation reference implementation is MIT (retained;
see `NOTICE`). Dataset licenses (e.g. ESA Anomaly Dataset, CC BY 3.0 IGO) are
separate from this code and are never bundled.
