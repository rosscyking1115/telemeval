from __future__ import annotations

import math

import numpy as np
import pytest

from telemeval.errors import SchemaError
from telemeval.sklearn import (
    affiliation_fbeta_score,
    affiliation_precision_score,
    affiliation_recall_score,
    event_wise_fbeta_score,
    event_wise_precision_score,
    event_wise_recall_score,
)


def test_perfect_prediction_scores_one_everywhere() -> None:
    y_true = [0, 0, 1, 1, 0, 0, 1, 0]
    y_pred = [0, 0, 1, 1, 0, 0, 1, 0]

    assert event_wise_precision_score(y_true, y_pred) == 1.0
    assert event_wise_recall_score(y_true, y_pred) == 1.0
    assert event_wise_fbeta_score(y_true, y_pred) == 1.0
    assert affiliation_precision_score(y_true, y_pred) == 1.0
    assert affiliation_recall_score(y_true, y_pred) == 1.0
    assert affiliation_fbeta_score(y_true, y_pred) == 1.0


def test_missed_event_and_false_alarm_are_penalised() -> None:
    y_true = [0, 1, 1, 0, 0, 0, 0, 0]
    y_pred = [0, 0, 0, 0, 0, 1, 0, 0]

    assert event_wise_recall_score(y_true, y_pred) == 0.0
    assert event_wise_precision_score(y_true, y_pred) == 0.0
    assert event_wise_fbeta_score(y_true, y_pred) == 0.0


def test_empty_prediction_gives_nan_affiliation_precision() -> None:
    y_true = [0, 1, 1, 0]
    y_pred = [0, 0, 0, 0]

    assert math.isnan(affiliation_precision_score(y_true, y_pred))
    assert affiliation_fbeta_score(y_true, y_pred) == 0.0


def test_wrappers_accept_numpy_arrays() -> None:
    y_true = np.array([0, 1, 1, 0, 0])
    y_pred = np.array([0, 1, 0, 0, 0])

    assert event_wise_recall_score(y_true, y_pred) == 1.0


def test_wrappers_work_with_sklearn_make_scorer_convention() -> None:
    # make_scorer-style call: score_func(y_true, y_pred, **kwargs)
    score = event_wise_fbeta_score([0, 1, 1, 0], [0, 1, 1, 0], beta=1.0)
    assert score == 1.0


def test_input_validation() -> None:
    with pytest.raises(SchemaError, match="same length"):
        event_wise_fbeta_score([0, 1], [0, 1, 0])
    with pytest.raises(SchemaError, match="must be binary"):
        event_wise_fbeta_score([0, 2], [0, 1])
    with pytest.raises(SchemaError, match="no anomalous samples"):
        event_wise_fbeta_score([0, 0], [0, 1])
    with pytest.raises(SchemaError, match="must not be empty"):
        event_wise_fbeta_score([], [])
