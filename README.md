# telemeval

**Leakage-safe, event-wise and affiliation-based evaluation for
spacecraft-telemetry anomaly detection.**

> Status: pre-release (v0.1 in development). API may change until v1.

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
- **Honest reports**: deterministic JSON and Markdown output stamped with
  explicit scope caveats.
- **Telemetry-aware inputs**: channel-keyed predictions (interval labels,
  binary masks, or continuous scores + threshold), an ESA-ADB-format loader,
  and a TimeEval-format reader.
- **scikit-learn-style metric wrappers** so the metrics drop into existing
  scoring code.

Core dependencies: numpy and pandas. Nothing else.

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
