# Changelog

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
