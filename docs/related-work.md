# Related Work — what to use when

telemeval composes standard ideas; none of the primitives are novel here, and
this page exists so you can pick the right tool honestly.

| Tool | What it is | When to prefer it over telemeval |
| --- | --- | --- |
| [TSADmetrics](https://pypi.org/project/tsadmetrics/) | Broad time-series AD metrics collection (incl. affiliation-based F-score, PATE). **GPL-3.0.** | You want many metric families and GPL is acceptable for your project. |
| [TSB-AD](https://github.com/TheDatumOrg/TSB-AD) | NeurIPS 2024 benchmark suite: curated datasets + algorithms + VUS-centric evaluation. | You are benchmarking many detectors across many public datasets. |
| [TimeEval](https://github.com/TimeEval/TimeEval) | Docker-based large-scale benchmark harness (range/VUS metrics; canonical dataset format). | You need distributed, containerized algorithm comparison. telemeval reads its dataset format. |
| [aeon](https://www.aeon-toolkit.org/) | TS ML toolkit; range-based and VUS AD metrics. | You already work in the aeon/sktime ecosystem and range/VUS metrics suffice. |
| [Merlion](https://github.com/salesforce/Merlion) | TS library with point-adjusted F1 evaluation (archived/read-only). | Historical comparison against point-adjusted numbers. |
| [affiliation-metrics-py](https://github.com/ahstat/affiliation-metrics-py) | The canonical affiliation reference implementation (MIT; not on PyPI; frozen since 2022). | You want the raw reference code; telemeval vendors and maintains exactly this code. |
| [ESA-ADB](https://github.com/kplabs-pl/ESA-ADB) | The ESA benchmark: dataset + TimeEval-fork evaluation pipeline with satellite-operator metrics (event-wise, ADTQC, subsystem-aware). | You are reproducing the official ESA-ADB benchmark end to end. telemeval packages the event-wise/affiliation layer as an installable library and reads ESA-ADB label files. |

## telemeval's niche

- **Permissive license** (Apache-2.0; the one existing packaged affiliation
  implementation is GPL-3.0, unusable in Apache/MIT or most industrial code).
- **The ingestion contract**: typed errors for malformed inputs and a
  train/test-window **leakage guard on by default** — the piece none of the
  above provide.
- **Telemetry-domain support**: channel-keyed predictions, ESA-ADB label
  files, category exclusions as explicit configuration.
- **Maintained affiliation metrics**: the canonical MIT reference code,
  vendored with attribution (the same approach ESA-ADB takes), with its test
  suite and the KDD-2022 paper-reproduction fixtures kept green.

## Metric background (why not point-adjusted F1)

Point-adjusted evaluation can rank random predictions above real detectors
(Kim et al. 2022, arXiv:2109.05257 follow-ups; "Rethink benchmarking",
arXiv:2507.15584). ESA-ADB (arXiv:2406.17826) adopted corrected event-wise
and affiliation scoring for exactly this reason, and a 2025 formal study of
37 metrics (arXiv:2510.17562) finds no metric satisfies all desirable
properties — so telemeval treats the metric suite as extensible (registry)
rather than settled.
