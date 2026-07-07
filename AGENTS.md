# telemeval — Agent Guide

## What this is

A leakage-safe, event-wise & affiliation-based evaluation library for
spacecraft-telemetry anomaly detection. Evaluation ONLY.

## Hard boundaries (do not cross without explicit approval)

- Library core depends on numpy + pandas ONLY. Never add torch, sklearn,
  fastapi, streamlit, or any heavy dependency to the runtime core.
- No detectors, models, thresholding strategies, preprocessing, resampling,
  serving, or UI in this repo. Those belong in downstream projects
  (e.g. the aerospace-prognostics reference pipeline).
- Never redistribute datasets. Dataset licenses stay separate; credit ESA-ADB
  in NOTICE and docs.
- Never claim novelty on primitives. The value is packaging + correctness +
  domain fit + maintenance — say so plainly in docs and release notes.

## Vendored code rules

- `src/telemeval/metrics/_affiliation_vendor/` is vendored from
  https://github.com/ahstat/affiliation-metrics-py (MIT). Keep it faithful to
  upstream. Its LICENSE file stays in place; NOTICE carries attribution.
- Do not lint-rewrite, reformat, or "modernize" vendored files. If a genuine
  bug fix is required, add a prominent modification notice in the changed file
  and record it in NOTICE, per Apache-2.0 §4(b).
- Wrap vendored APIs behind telemeval's own modules; downstream users must
  never import from `_affiliation_vendor` directly.

## Correctness rules

- Metrics must be fixture-tested. Affiliation results must match the vendored
  reference (same code, so wrapper tests guard the integration, and the ported
  upstream test suite guards the core).
- The ingestion contract must raise typed, actionable errors — especially for
  train/test-window leakage. A wrong number is worse than a raised error.
- Reports must be deterministic and carry explicit scope caveats. No
  overclaiming, anywhere.
- New metrics go through the metric registry; do not hardcode a metric suite.

## Workflow

- `uv run ruff check .` and `uv run pytest` green before any commit.
- Type-annotate the public API. Keep the sklearn-style wrappers thin.
- Conventional commits. Update specs/ when behavior or scope changes.
