# Changelog

## 0.3.0 — 2026-07-07

- **New metric: channel-aware / subsystem-aware F-beta** (ESA-ADB affected-
  source diagnosis), implemented from the reference `ChannelAwareFScore` and
  verified against its four test cases ported verbatim (exact expected
  values). Registered as `"channel_aware"`; subsystem mapping via
  `metric_options` and the new `formats.esa_adb.read_channels` reader.
- `evaluate()` gains `metric_options` for per-metric keyword options
  (unknown names rejected).
- Contract: timestamp parsing now falls back to per-element inference for
  mixed-format label files (e.g. dates next to datetimes) before rejecting.
- Real Mission1 readout (per-channel tau=5 baseline): channel F0.5 0.414,
  subsystem F0.5 0.4154 — equal to the 27/65 detection rate, as expected
  for a single-subsystem channel set.


## 0.2.0 — 2026-07-07

- **New metric: ADTQC** (ESA-ADB's Anomaly Detection Timing Quality Curve) —
  scores when each event was first detected; the first packaged
  implementation (previously existed only inside ESA-ADB's research fork).
  Semantics verified against the reference source; latencies computed in
  seconds on the merged global timeline (deviation documented). Registered
  as `"adtqc"`; opt-in via `evaluate(metrics=(..., "adtqc"))`.
  Validated on real ESA-ADB Mission1 telemetry: the robust-threshold
  baseline scores ADTQC total 0.9618 over its 27 detections (median
  latency 0 s) — conservative detectors fire rarely but at onset.


## 0.1.2 — 2026-07-07

- Citation metadata: `.zenodo.json` and version/date in `CITATION.cff` for
  Zenodo DOI archiving. No functional changes.


## 0.1.1 — 2026-07-07

- Suppress the vendored reference code's docstring SyntaxWarnings at import,
  so a fresh `pip install telemeval` imports cleanly. No functional changes.


## 0.1.0 — 2026-07-07

First release.

- Ingestion contract with typed errors: schema/interval/metadata-join/
  alignment/monotonicity/binary-domain guards, and a first-class
  **train/test-window leakage guard** (on by default in `evaluate()`;
  `clip_to_window=True` is explicit and recorded, never silent).
- Corrected **event-wise precision/recall/F-beta** (ESA-ADB-aligned
  semantics; every event weighs equally).
- **Affiliation-based precision/recall** (Huet et al., KDD 2022): the
  canonical MIT reference implementation vendored with attribution, wrapped
  behind the contract, upstream test suite and paper-reproduction fixtures
  kept green.
- `evaluate()` facade + `EvaluationResult`, metric registry, deterministic
  JSON/Markdown reports with scope caveats.
- scikit-learn-convention metric wrappers.
- Format readers: ESA-ADB (`labels.csv` + `anomaly_types.csv`) and TimeEval
  canonical CSV.
- Core dependencies: numpy and pandas only (CI-enforced).

Known scope limits (see `specs/roadmap.md`): ADTQC timing, subsystem-aware
F0.5, and modified affiliation are planned for v0.x follow-ups; this release
covers event-wise detection and affiliation only.
