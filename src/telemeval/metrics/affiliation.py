"""Affiliation-based precision / recall (Huet et al., KDD 2022).

This is the supported wrapper around the vendored canonical reference
implementation (see ``_affiliation_vendor/VENDORED.md``). Ground-truth
intervals and channel predictions are converted to the reference code's
index-space event convention using the reference's own binary-vector
converter, so the semantics cannot drift from upstream.

Affiliation here is computed on the *merged* global event timeline (all
labelled intervals collapsed onto the prediction grid), which matches how the
reference implementation consumes a single binary ground-truth vector.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from telemeval.errors import SchemaError

# The vendored reference code (kept faithful to upstream) contains LaTeX-ish
# escape sequences in docstrings that emit SyntaxWarnings when first compiled.
# Suppress them here, at the only supported import site, instead of editing
# the vendored files.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", SyntaxWarning)
    from telemeval.metrics._affiliation_vendor.generics import convert_vector_to_events
    from telemeval.metrics._affiliation_vendor.metrics import pr_from_events

__all__ = ["score_affiliation"]


def score_affiliation(
    metric_inputs: Mapping[str, Any],
    *,
    beta: float = 0.5,
    exclude_categories: Sequence[str] = (),
) -> dict[str, Any]:
    """Score affiliation-based precision/recall from validated metric inputs.

    ``metric_inputs`` is the mapping returned by
    :func:`telemeval.contract.build_metric_inputs`. ``beta`` controls the
    reported F-beta (0.5 by default, matching ESA-ADB's precision weighting).

    Raises :class:`~telemeval.errors.SchemaError` when the evaluation would be
    silently wrong: no ground-truth events left after category exclusion, or
    labelled events that cover no prediction samples (such events would
    vanish from a merged timeline and inflate recall).
    """

    if beta <= 0:
        raise SchemaError("beta must be positive")

    global_labels = metric_inputs["global_labels"]
    global_predictions = metric_inputs["global_predictions"]
    excluded = set(exclude_categories)
    if excluded and "Category" not in global_labels.columns:
        raise SchemaError(
            "exclude_categories was requested but labels carry no 'Category' "
            "metadata column; join event metadata first"
        )
    if excluded:
        global_labels = global_labels[~global_labels["Category"].isin(excluded)]

    grid = pd.DatetimeIndex(global_predictions["Timestamp"]).to_numpy("datetime64[ns]")
    gt_mask, off_grid_ids = _labels_to_mask(global_labels, grid)
    if off_grid_ids:
        raise SchemaError(
            "labelled event(s) cover no prediction samples and would silently "
            f"vanish from the affiliation timeline: {off_grid_ids[:10]}"
            f"{'...' if len(off_grid_ids) > 10 else ''}. Align the prediction "
            "grid with the labels or remove these events explicitly."
        )

    events_gt = convert_vector_to_events(gt_mask.tolist())
    if not events_gt:
        raise SchemaError(
            "affiliation requires at least one ground-truth event on the "
            "prediction grid (after any category exclusions)"
        )
    events_pred = convert_vector_to_events(global_predictions["Score"].tolist())
    trange = (0, len(grid))

    result = pr_from_events(events_pred, events_gt, trange)

    precision = _nan_to_none(result["precision"])
    recall = _nan_to_none(result["recall"])
    fbeta = _fbeta(precision, recall, beta)

    return {
        "beta": beta,
        "excluded_categories": sorted(excluded),
        "gt_events_merged": len(events_gt),
        "predicted_events": len(events_pred),
        "affiliation_precision": precision,
        "affiliation_recall": recall,
        "affiliation_fbeta": fbeta,
        "individual_precision_probabilities": [
            _nan_to_none(value) for value in result["individual_precision_probabilities"]
        ],
        "individual_recall_probabilities": [
            _nan_to_none(value) for value in result["individual_recall_probabilities"]
        ],
    }


def _labels_to_mask(
    global_labels: pd.DataFrame,
    grid: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Mark grid samples inside any labelled interval; report off-grid events."""

    mask = np.zeros(len(grid), dtype="uint8")
    off_grid: list[str] = []
    for event_id, group in global_labels.groupby("ID", sort=False):
        covered = False
        for start, end in zip(
            pd.to_datetime(group["StartTime"]).to_numpy("datetime64[ns]"),
            pd.to_datetime(group["EndTime"]).to_numpy("datetime64[ns]"),
            strict=True,
        ):
            lo = int(np.searchsorted(grid, start, side="left"))
            hi = int(np.searchsorted(grid, end, side="right"))
            if hi > lo:
                mask[lo:hi] = 1
                covered = True
        if not covered:
            off_grid.append(str(event_id))
    return mask, sorted(off_grid)


def _nan_to_none(value: float) -> float | None:
    if isinstance(value, float) and math.isnan(value):
        return None
    return float(value)


def _fbeta(precision: float | None, recall: float | None, beta: float) -> float:
    if precision is None or recall is None:
        return 0.0
    beta_sq = beta * beta
    denominator = beta_sq * precision + recall
    if denominator <= 0.0:
        return 0.0
    return (1.0 + beta_sq) * precision * recall / denominator
