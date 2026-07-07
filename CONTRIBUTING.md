# Contributing to telemeval

Thanks for your interest! Issues and pull requests are welcome.

## Reporting problems

Open an issue with your telemeval version, a minimal input that reproduces
the problem (labels + predictions), and what you expected vs got. Metric
correctness reports are especially valued — if you believe a number is wrong,
a comparison against a reference implementation or a hand-computed example
makes it actionable fast.

## Development setup

```bash
git clone https://github.com/rosscyking1115/telemeval
cd telemeval
uv sync --dev
uv run ruff check .
uv run pytest
```

All contributions must keep `ruff check` and the full test suite green.
CI also enforces the core promise that a clean install pulls only numpy and
pandas — do not add runtime dependencies to the library core.

## Ground rules (from AGENTS.md, they bind humans too)

- **Evaluation only.** No detectors, thresholding strategies, preprocessing,
  serving, or UI in this repo.
- **Vendored code stays faithful.** `src/telemeval/metrics/_affiliation_vendor/`
  is the canonical MIT reference implementation; do not reformat or
  "modernize" it. Genuine bug fixes need a prominent modification notice
  (see `VENDORED.md`).
- **Metrics are fixture-tested.** New metrics implemented from a published
  reference must port the reference's own test cases (or document exactly
  where and why results diverge — see `tests/test_event_wise_divergence.py`
  for the pattern).
- **Honest claims.** Reports carry scope caveats; docs never overstate
  alignment with other tools. A raised error beats a wrong number.
- **No dataset redistribution.** Dataset licenses stay separate from this
  code.

## Adding a metric

Register it through `telemeval.register_metric` (see `docs/usage.md`), accept
`(metric_inputs, *, beta, exclude_categories, **options)`, return a
JSON-serializable mapping, and document its semantics in
`specs/metric-definitions.md`.
