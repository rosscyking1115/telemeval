# telemeval — Metric Definitions

## Event-wise precision / recall / F-beta (v1, telemeval semantics)

- A labelled **event is detected** when any positive prediction sample falls
  inside its inclusive `[StartTime, EndTime]` interval (on the global
  max-over-channels series). Multi-row events are collapsed to one
  `[min start, max end]` interval.
- A **predicted positive run** (consecutive positive samples) is a **true
  alarm** if any of its samples falls inside any scored event, else a false
  alarm.
- `recall = detected_events / total_events`;
  `precision = true_alarm_runs / total_runs`;
  `F-beta` with beta=0.5 by default. Empty-denominator conventions are
  documented in code and tested.

**Relationship to ESA-ADB's ESAScores `EW_*` (audited 2026-07-07 on their own
fixtures; see `tests/test_event_wise_divergence.py`):** recall matches
ESA-ADB exactly (every event weighs equally). Precision and F-beta are
telemeval's own run-based definition and **diverge numerically** from
`EW_precision`/`EW_F`: telemeval has no TNR duration correction, no
`alarming_precision`, exclude-based (not select-based) category handling, and
collapsed (not union) event intervals. Do not quote these as ESA-ADB `EW_*`
numbers. An ESAScores-parity metric is planned (roadmap.md).

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

## Channel-aware / subsystem-aware F-beta (v0.3)

Affected-source diagnosis, implemented from and fixture-verified against
ESA-ADB's `ranking_metrics.py::ChannelAwareFScore` (four reference test cases
ported verbatim with exact expected values):

- Per event, each predicted channel is TP (labelled + any detection overlaps
  the event's full interval), FN (labelled, none), or FP (unlabelled +
  detection) — where an FP is excused if its detection, restricted to this
  event's interval, overlaps another event's labels on the same channel.
- Precision/recall/F-beta per event, averaged over events. Point labels are
  widened by 1 ms exactly as the reference does.
- Subsystem variant groups channels via a mapping (ESA-ADB `channels.csv`;
  `telemeval.formats.esa_adb.read_channels`), with the reference's
  zero-out-and-recheck excusal.
- Registered as `"channel_aware"`; the mapping is passed via
  `evaluate(metric_options={"channel_aware": {"subsystems_mapping": ...}})`.

## v0.x (specified, not shipped)

- Channel-aware ADTQC variant; modified affiliation-based F0.5.
- Watched, not promised: PATE, LARM/ALARM.
