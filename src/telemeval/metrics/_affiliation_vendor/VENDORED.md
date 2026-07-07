# Vendored: affiliation-metrics-py

- **Upstream:** https://github.com/ahstat/affiliation-metrics-py
- **Upstream state:** `main` branch (frozen upstream since 2022-05-30; the
  repository has had no code changes since its initial publication).
- **License:** MIT, Copyright (c) 2022 Alexis Huet and others — retained
  verbatim in `LICENSE` in this directory; attribution recorded in the
  project-level `NOTICE`.
- **Paper:** Alexis Huet, Jose Manuel Navarro, Dario Rossi. "Local Evaluation
  of Time Series Anomaly Detection Algorithms." KDD 2022.
  https://doi.org/10.1145/3534678.3539339
- **Why vendored:** the reference implementation is not published to PyPI, so
  a versioned dependency is impossible; the ESA-ADB benchmark vendors this
  same code. telemeval maintains it going forward.

## Modifications from upstream

1. Import paths rewritten from `affiliation.` to
   `telemeval.metrics._affiliation_vendor.` (mechanical; required for
   vendoring). Each file carries a header notice.
2. Nothing else. Do not reformat, lint-rewrite, or "modernize" these files.
   Genuine bug fixes require a prominent per-file modification notice and a
   NOTICE entry (Apache-2.0 §4(b)).

Upstream's test suite is ported (imports likewise rewritten) to
`tests/vendor_affiliation/`, including the paper-reproduction data fixtures
(`data/*.gz`, test-only — excluded from the distributed wheel).

Downstream users must not import from `_affiliation_vendor` directly; the
supported surface is `telemeval.metrics.affiliation`.
