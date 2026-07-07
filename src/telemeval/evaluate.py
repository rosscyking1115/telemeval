"""The telemeval facade: one call from labels + predictions to guarded metrics.

``evaluate()`` runs the full contract — including the train/test-window
leakage guard, ON by default — then every requested metric, and returns an
:class:`EvaluationResult` that renders honest reports.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from telemeval.contract import (
    assert_window_consistent,
    build_metric_inputs,
    validate_labels,
)
from telemeval.registry import get_metric
from telemeval.report import build_report, render_markdown, write_report

DEFAULT_METRICS = ("event_wise", "affiliation")

__all__ = ["DEFAULT_METRICS", "EvaluationResult", "evaluate"]


@dataclass
class EvaluationResult:
    """Metrics plus the configuration that produced them."""

    metrics: dict[str, dict[str, Any]]
    target_channels: list[str]
    config: dict[str, Any] = field(default_factory=dict)
    dataset: str | None = None

    def to_report(self, *, scope_notes: list[str] | None = None) -> dict[str, Any]:
        notes = list(scope_notes or [])
        computed = ", ".join(sorted(self.metrics))
        notes.append(
            f"Computed metrics: {computed}. Timing (ADTQC), subsystem-aware, "
            "and proximity metrics beyond affiliation are not included."
        )
        return build_report(
            self.metrics,
            target_channels=self.target_channels,
            dataset=self.dataset,
            scope_notes=notes,
            config=self.config,
        )

    def to_markdown(self, *, scope_notes: list[str] | None = None) -> str:
        return render_markdown(self.to_report(scope_notes=scope_notes))

    def save(
        self,
        *,
        json_path: str | None = None,
        markdown_path: str | None = None,
        scope_notes: list[str] | None = None,
    ) -> None:
        write_report(
            self.to_report(scope_notes=scope_notes),
            json_path=json_path,
            markdown_path=markdown_path,
        )


def evaluate(
    labels: pd.DataFrame,
    predictions_by_channel: Mapping[str, pd.DataFrame],
    *,
    metadata: pd.DataFrame | None = None,
    threshold: float | None = None,
    metrics: Sequence[str] = DEFAULT_METRICS,
    beta: float = 0.5,
    exclude_categories: Sequence[str] = (),
    clip_to_window: bool = False,
    dataset: str | None = None,
) -> EvaluationResult:
    """Validate inputs, guard the evaluation window, and score all metrics.

    Parameters mirror the contract: ``labels`` are interval events
    (``ID, Channel, StartTime, EndTime``), optionally joined with event
    ``metadata`` by ID; ``predictions_by_channel`` maps channels to
    ``Timestamp, Score`` frames (binary, or continuous with ``threshold``).

    The window-leakage guard is **on by default**: events lying entirely
    outside the prediction window raise
    :class:`~telemeval.errors.WindowLeakageError` unless ``clip_to_window=True``
    (in which case the clipping is recorded in the result config).
    """

    validated_labels = validate_labels(labels, metadata)
    inputs = build_metric_inputs(
        validated_labels, predictions_by_channel, threshold=threshold
    )

    grid = inputs["global_predictions"]["Timestamp"]
    window_start, window_end = grid.iloc[0], grid.iloc[-1]
    n_before = inputs["channel_labels"]["ID"].nunique()
    guarded = assert_window_consistent(
        inputs["channel_labels"], window_start, window_end, clip_to_window=clip_to_window
    )
    n_after = guarded["ID"].nunique()
    if n_after != n_before:
        # Re-derive metric inputs from the explicitly clipped labels.
        inputs = build_metric_inputs(
            guarded, predictions_by_channel, threshold=threshold
        )

    results: dict[str, dict[str, Any]] = {}
    for name in metrics:
        metric_fn = get_metric(name)
        results[name] = dict(
            metric_fn(inputs, beta=beta, exclude_categories=exclude_categories)
        )

    config: dict[str, Any] = {
        "beta": beta,
        "exclude_categories": sorted(exclude_categories),
        "window_start": pd.Timestamp(window_start).isoformat(),
        "window_end": pd.Timestamp(window_end).isoformat(),
        "window_guard": "clip_to_window" if clip_to_window else "strict",
        "events_clipped_by_window_guard": int(n_before - n_after),
    }
    if threshold is not None:
        config["threshold"] = float(threshold)

    return EvaluationResult(
        metrics=results,
        target_channels=inputs["target_channels"],
        config=config,
        dataset=dataset,
    )
