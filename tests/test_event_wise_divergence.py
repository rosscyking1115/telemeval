"""Documented divergence between telemeval's event-wise metric and ESA-ADB's
ESAScores EW_*.

The input fixtures below are ported VERBATIM from ESA-ADB's own test suite
(tests/metrics/test_metrics.py::TestOtherMetrics.test_time_aware), and each
test pins BOTH numbers: what telemeval returns, and what ESAScores returns for
the same inputs (their published expected values). They agree on recall and
deliberately diverge on precision/F-beta:

- telemeval precision is **run-based** (fraction of predicted positive runs
  that hit a scored event) and treats alarms inside *excluded* events as
  false alarms;
- ESAScores precision is event-based multiplied by a **TNR duration
  correction** (false-positive seconds in nominal time), uses select-labels
  semantics where alarms inside non-selected anomalies are shielded by the
  full ground truth, and additionally reports **alarming_precision**, which
  telemeval does not compute.

If a change ever makes these numbers match, that is not a bug fix — it is a
semantics change, and these tests exist to make it loud.
"""

from __future__ import annotations

import pandas as pd
import pytest

from telemeval import build_metric_inputs, score_event_wise, validate_labels

FULL_RANGE = (pd.Timestamp("2015-01-01"), pd.Timestamp("2015-01-15"))
GRID = pd.date_range(*FULL_RANGE, freq="1h")

# ESA-ADB fixture labels (their frame is channel-agnostic; telemeval requires
# a channel, so a single "global" channel carries them).
LABELS = pd.DataFrame(
    [
        ["id_0", "global", "2015-01-01", "2015-01-02", "Rare Event"],
        ["id_1", "global", "2015-01-04", "2015-01-05", "Anomaly"],
        ["id_1", "global", "2015-01-06", "2015-01-07", "Anomaly"],
        ["id_2", "global", "2015-01-09", "2015-01-10", "Communication Gap"],
        ["id_3", "global", "2015-01-12", "2015-01-14", "Rare Event"],
    ],
    columns=["ID", "Channel", "StartTime", "EndTime", "Category"],
)

# telemeval's exclude_categories=("Communication Gap",) corresponds to
# ESAScores' select_labels={"Category": ["Rare Event", "Anomaly"]}.
EXCLUDE = ("Communication Gap",)


def _score(changes: list[tuple[str, int]]) -> dict:
    points = [(pd.Timestamp(t), v) for t, v in changes]
    mask = []
    for t in GRID:
        value = 0
        for ct, cv in points:
            if ct <= t:
                value = cv
            else:
                break
        mask.append(value)
    inputs = build_metric_inputs(
        validate_labels(LABELS),
        {"global": pd.DataFrame({"Timestamp": GRID, "Score": mask})},
    )
    return score_event_wise(inputs, beta=0.5, exclude_categories=EXCLUDE)


def test_fixture_a_alarm_inside_excluded_event() -> None:
    # Positives over [01-01,01-02), [01-04,01-05), [01-09,01-10).
    # ESAScores (select Rare Event+Anomaly): EW_precision=1.0,
    # EW_recall=0.6667, EW_F_0.50=0.9091, alarming_precision=1.0 — the 01-09
    # alarm lands inside the de-selected Communication Gap and is SHIELDED.
    scores = _score(
        [
            ("2015-01-01", 1),
            ("2015-01-02", 0),
            ("2015-01-04", 1),
            ("2015-01-05", 0),
            ("2015-01-09", 1),
            ("2015-01-10", 0),
        ]
    )

    # Recall agrees with ESAScores.
    assert scores["event_wise_recall"] == pytest.approx(2 / 3)
    # telemeval treats the alarm inside the EXCLUDED event as a false alarm.
    assert scores["false_alarms"] == 1
    assert scores["event_wise_precision"] == pytest.approx(2 / 3)
    assert scores["event_wise_precision"] != pytest.approx(1.0)  # ESA's value
    assert scores["event_wise_fbeta"] != pytest.approx(0.9090909090909091)


def test_fixture_c_no_tnr_duration_correction() -> None:
    # Adds a genuinely nominal alarm at [01-11,01-12).
    # ESAScores: EW_precision=0.5833 (= 2/3 event precision x 0.875 TNR),
    # EW_recall=0.6667, EW_F_0.50=0.5983.
    scores = _score(
        [
            ("2015-01-01", 1),
            ("2015-01-02", 0),
            ("2015-01-04", 1),
            ("2015-01-05", 0),
            ("2015-01-06", 1),
            ("2015-01-07", 0),
            ("2015-01-09", 1),
            ("2015-01-10", 0),
            ("2015-01-11", 1),
            ("2015-01-12", 0),
        ]
    )

    assert scores["event_wise_recall"] == pytest.approx(2 / 3)  # agrees
    # telemeval: 3 of 5 runs are true alarms; no TNR term exists.
    assert scores["event_wise_precision"] == pytest.approx(0.6)
    assert scores["event_wise_precision"] != pytest.approx(0.5833333333333333)
    assert scores["event_wise_fbeta"] != pytest.approx(0.5982905982905982)


def test_fixture_d_no_alarming_precision_and_id_collapse() -> None:
    # Two separate alarms both inside id_1's two intervals.
    # ESAScores: EW_precision=0.75 (= 1.0 x 0.75 TNR: the alarms spill into
    # the nominal gap between id_1's intervals), EW_recall=0.3333,
    # EW_F_0.50=0.60, alarming_precision=0.5 (double detection penalized).
    scores = _score(
        [
            ("2015-01-01", 0),
            ("2015-01-04", 1),
            ("2015-01-06 12:00", 0),
            ("2015-01-06 14:00", 1),
            ("2015-01-08", 0),
        ]
    )

    assert scores["event_wise_recall"] == pytest.approx(1 / 3)  # agrees
    # telemeval collapses id_1's rows to one [01-04, 01-07] interval and has
    # no duration term: both runs are true alarms -> precision 1.0.
    assert scores["event_wise_precision"] == pytest.approx(1.0)
    assert scores["event_wise_precision"] != pytest.approx(0.75)  # ESA's value
    assert scores["event_wise_fbeta"] != pytest.approx(0.6000000000000001)
    # And telemeval intentionally emits no alarming_precision at all.
    assert "alarming_precision" not in scores
