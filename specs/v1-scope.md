# telemeval v1 — Scope

## Product

Apache-2.0, numpy+pandas core, `pip install telemeval`. Evaluation only.

## v1 ships

- **Ingestion contract** with typed guards: column/format validation,
  cross-channel timestamp alignment, monotonic/duplicate timestamps, binary
  score validation, missing anomaly-type IDs, reversed intervals, and a
  first-class **train/test-window leakage guard**. Channel identity is
  preserved on predictions. Accepted prediction forms: interval labels,
  binary point masks, continuous scores + threshold.
- **Metrics**: corrected event-wise precision/recall/F-beta (ported, tested);
  affiliation-based precision/recall (vendored MIT reference, wrapped).
- **Reports**: deterministic JSON + Markdown with explicit scope caveats.
- **Formats**: ESA-ADB reference loader; TimeEval canonical CSV reader;
  neutral in-memory (pandas) API. Parquet documented via pandas.
- **API**: `evaluate()` facade + `EvaluationResult`; sklearn-convention
  metric wrappers; metric registry.

## Non-goals (v1)

- No detectors, models, thresholding strategies, preprocessing, resampling.
- Not the full ESA-ADB hierarchy: ADTQC, subsystem/channel-aware F0.5, and
  modified affiliation are v1.x (see roadmap.md). The contract carries
  channel identity now so they slot in without breaking the API.
- No serving, drift, dashboards, orchestration, MCP server.
- No dataset redistribution.

## Acceptance criteria

- Clean-venv `pip install telemeval` pulls only numpy + pandas (CI-enforced).
- Event-wise metrics reproduce the aerospace-prognostics ESA-ADB Mission1
  lightweight checkpoint (fixed tau=5 -> P 1.000 / R 0.415 / F0.5 0.780).
- Affiliation wrapper results match the vendored reference; upstream test
  suite ported and green.
- Each contract guard has a test proving it raises its typed error —
  including the leakage guard (regression test from the real bug caught in
  the origin project).
- Reports deterministic; sklearn wrappers usable in a plain sklearn scoring
  flow; CI green on Linux + Windows.
