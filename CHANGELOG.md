# Changelog

## 0.3.2 — 2026-07-08

- Metadata/housekeeping (no functional change): author set to Cheng-Yuan King
  with contact email; added `SECURITY.md` (report to rosscyking@gmail.com);
  CI now tests Python 3.11/3.12/3.13; simplified `CITATION.cff` to the concept
  DOI only (version DOIs are minted per release on Zenodo); added provenance
  notes to the private-data readouts in earlier entries.


## 0.3.1 — 2026-07-07

- **Documentation correction (honesty patch).** An audit against ESA-ADB's
  own test fixtures showed our event-wise metric's earlier "ESA-ADB-aligned
  semantics" wording was too strong: **recall matches ESA-ADB exactly**, but
  precision/F-beta are telemeval's own run-based definition and diverge
  numerically from ESAScores `EW_*` (no TNR duration correction, no
  `alarming_precision`, exclude- vs select-based category handling,
  collapsed vs union event intervals). Docstrings, specs, and README now
  state this precisely, and three divergence tests ported verbatim from
  ESA-ADB's fixtures pin both sets of numbers.
- Roadmap: an ESAScores-parity metric ("esa_scores") is now the next
  milestone, alongside ESA's modified-affiliation wrapper.
- No metric behavior changed in this release.


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
- Real-data readout (measured on the ESA Anomaly Dataset Mission1 lightweight
  subset, which is not redistributed here, so this is not reproducible from
  this repo): per-channel tau=5 baseline scores channel F0.5 0.414, subsystem
  F0.5 0.4154 — equal to the 27/65 detection rate, as expected for a
  single-subsystem channel set.


## 0.2.0 — 2026-07-07

- **New metric: ADTQC** (ESA-ADB's Anomaly Detection Timing Quality Curve) —
  scores when each event was first detected; the first packaged
  implementation (previously existed only inside ESA-ADB's research fork).
  Semantics verified against the reference source; latencies computed in
  seconds on the merged global timeline (deviation documented). Registered
  as `"adtqc"`; opt-in via `evaluate(metrics=(..., "adtqc"))`.
  Validated on real ESA-ADB Mission1 telemetry (the ESA Anomaly Dataset is
  not redistributed here, so this figure is not reproducible from this repo):
  the robust-threshold baseline scores ADTQC total 0.9618 over its 27
  detections (median latency 0 s) — conservative detectors fire rarely but
  at onset.


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
