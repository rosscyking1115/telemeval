# telemeval — Metric Definitions

## Corrected event-wise precision / recall / F-beta (v1)

Semantics (ported from the origin project's tested scorer, aligned with
ESA-ADB's corrected event-wise scoring):

- A labelled **event is detected** when any positive prediction sample falls
  inside its inclusive `[StartTime, EndTime]` interval (on the global
  max-over-channels series).
- A **predicted positive run** (consecutive positive samples) is a **true
  alarm** if any of its samples falls inside any event, else a false alarm.
- `recall = detected_events / total_events`;
  `precision = true_alarm_runs / total_runs`;
  `F-beta` with beta=0.5 by default (precision-weighted, matching ESA-ADB's
  false-alarm sensitivity). Empty-denominator conventions are documented in
  code and tested.
- All events weigh equally regardless of length ("corrected" semantics).

Reference: Kotowski et al., ESA-ADB, arXiv:2406.17826.

## Affiliation-based precision / recall (v1)

The canonical definition from Huet, Navarro, Rossi (KDD 2022,
doi:10.1145/3534678.3539339): each ground-truth event owns an affiliation
zone; predicted intervals are affiliated to their closest event; precision/
recall are averaged per-event from distance-based probabilities against a
uniform-random baseline predictor.

Implementation: the authors' reference code, vendored verbatim (MIT) at
`metrics/_affiliation_vendor/`, wrapped behind telemeval's contract. We do
not re-derive the math; we maintain the canonical code. Upstream's test
suite is ported and kept green.

## ADTQC — detection-timing quality (v0.2)

Semantics follow ESA-ADB's `latency_metrics.py` (verified against the
reference source):

- Latency ``x`` = start of the earliest predicted run **overlapping** the
  event, minus event start (negative = early detection).
- Curve ``f(x, a, b)`` with exponent ``e``: 1 when ``x == 0`` and ``a`` or
  ``b`` is zero (point anomaly caught at onset); 0 when ``x <= -a`` or
  ``x >= b``; ``((x+a)/a)^e`` on the early side; ``1/(1+(x/(b-x))^e)`` after
  onset.
- Allowance ``a = min(length, gap to previous event's START)``; the first
  event's allowance is its own length.
- Aggregates (``nb_before``, ``nb_after``, ``after_rate``, ``adtqc_total``)
  are computed **over detected events only** — undetected events are the
  recall metric's business. No detections at all → ``None`` aggregates.

Deviation, recorded honestly: telemeval computes latencies in seconds on the
merged global timeline (consistent with its other metrics) rather than on
ESA-ADB's per-channel resampled index. Opt-in via
``evaluate(metrics=("event_wise", "affiliation", "adtqc"))``.

## v0.x (specified, not shipped)

- **Subsystem-aware F0.5**: adds a channel->subsystem mapping input
  (ESA-ADB `channels.csv` shape); will also make ADTQC channel-aware.
- Watched, not promised: PATE, LARM/ALARM.
