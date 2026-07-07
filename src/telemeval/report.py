"""Deterministic evaluation reports with explicit scope caveats.

Reports never overclaim: every artifact records which metrics were computed,
which events were excluded, and what the numbers do NOT cover.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from telemeval._io import prepare_output_path, write_json_payload

REPORT_SCHEMA = "telemeval/evaluation-report/v1"

_DEFAULT_CAVEAT = (
    "Metrics cover only what is listed under 'metrics'; nothing beyond the "
    "supplied labels, predictions, and configuration is claimed."
)

__all__ = ["REPORT_SCHEMA", "build_report", "render_markdown", "write_report"]


def build_report(
    metrics: Mapping[str, Mapping[str, Any]],
    *,
    target_channels: list[str],
    dataset: str | None = None,
    scope_notes: list[str] | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a JSON-serializable evaluation report.

    ``metrics`` maps metric names (e.g. ``"event_wise"``, ``"affiliation"``)
    to their score mappings. Timestamps are ISO-formatted so the payload is
    deterministic and diff-friendly.
    """

    return {
        "schema_version": REPORT_SCHEMA,
        "dataset": dataset,
        "target_channels": [str(channel) for channel in target_channels],
        "config": dict(config) if config is not None else {},
        "scope_notes": [_DEFAULT_CAVEAT, *(scope_notes or [])],
        "metrics": {name: _serialize(result) for name, result in metrics.items()},
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a report as reviewer-facing Markdown."""

    dataset = report.get("dataset") or "unnamed dataset"
    lines = [
        f"# telemeval Evaluation Report — {dataset}",
        "",
    ]
    for note in report.get("scope_notes", []):
        lines.append(f"> {note}")
    lines.extend(
        [
            "",
            f"- Target channels: {len(report.get('target_channels', []))}.",
        ]
    )
    config = report.get("config") or {}
    for key in sorted(config):
        lines.append(f"- {key}: {config[key]}.")
    lines.append("")

    metrics = report.get("metrics", {})
    for name in sorted(metrics):
        result = metrics[name]
        lines.extend([f"## {name}", "", "| Metric | Value |", "| --- | ---: |"])
        for key in sorted(result):
            value = result[key]
            if isinstance(value, bool | int | str):
                lines.append(f"| {key} | {value} |")
            elif isinstance(value, float):
                lines.append(f"| {key} | {value:.6f} |")
            # nested structures (per_event, curves) are JSON-only detail
        lines.append("")

        per_event = result.get("per_event")
        if isinstance(per_event, list) and per_event:
            lines.extend(
                [
                    "### Per-event detail",
                    "",
                    "| Event ID | Category | Start | End | Detected |",
                    "| --- | --- | --- | --- | --- |",
                ]
            )
            for row in per_event:
                lines.append(
                    f"| {row.get('id')} | {row.get('category')} | {row.get('start_time')} | "
                    f"{row.get('end_time')} | {'yes' if row.get('detected') else 'no'} |"
                )
            lines.append("")

    return "\n".join(lines)


def write_report(
    report: Mapping[str, Any],
    *,
    json_path: str | None = None,
    markdown_path: str | None = None,
) -> None:
    """Write a report as JSON and/or Markdown artifacts."""

    if json_path is not None:
        write_json_payload(dict(report), json_path)
    if markdown_path is not None:
        output_path = prepare_output_path(markdown_path)
        output_path.write_text(render_markdown(report) + "\n", encoding="utf-8")


def _serialize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item") and not isinstance(value, str):  # numpy scalars
        try:
            return value.item()
        except (AttributeError, ValueError):
            return value
    return value
