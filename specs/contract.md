# telemeval — Ingestion Contract

The contract's job: **refuse to produce a metric from a leaky or malformed
evaluation.** Every guard raises a typed error with an actionable message.

## Inputs

### Labels (ground truth)

Interval events, one row per (event, channel):

| Column | Type | Notes |
| --- | --- | --- |
| `ID` | str | event identifier; one event may span channels |
| `Channel` | str | channel the interval applies to |
| `StartTime` | timestamp | inclusive |
| `EndTime` | timestamp | inclusive; must be >= StartTime |

Optional event metadata joined by `ID` (ESA-ADB `anomaly_types.csv` shape):
`Category`, `Dimensionality`, `Locality`, `Length`. Category filtering
(e.g. excluding communication gaps) is explicit configuration, never silent.

### Predictions

Channel-keyed mapping `{channel: frame}`. Accepted forms:

1. **Binary mask**: `Timestamp, Score` with Score in {0, 1}.
2. **Continuous scores + threshold**: `Timestamp, Score` float plus a
   threshold supplied in configuration; telemeval binarizes and records the
   threshold in the report (thresholds are inputs, never fitted here).
3. **Interval predictions**: `StartTime, EndTime` rows per channel,
   converted to the internal event form.

Channel identity is preserved end-to-end (required by v1.x ADTQC and
subsystem-aware scoring).

## Guards (typed errors)

| Guard | Error condition |
| --- | --- |
| Schema | missing/invalid columns in labels, types, or predictions |
| Interval sanity | `StartTime > EndTime` |
| Type join | label `ID` missing from anomaly-type metadata; duplicate IDs |
| Alignment | prediction timestamps differ across channels |
| Monotonicity | non-increasing or duplicate timestamps in a channel |
| Binary domain | scores outside {0,1} when a binary mask is declared |
| **Window leakage** | labelled events lying entirely outside the prediction window (they can never be detected; counting them corrupts recall). The caller must either pass window-consistent labels or explicitly opt into `clip_to_window` — silent filtering is not performed. |

Rationale for the leakage guard: this exact bug occurred in the origin
project (training-half events counted as missed against test-half
predictions, deflating recall 0.42 -> 0.24) and was caught only by manual
audit. The contract makes that class of error impossible to miss.
