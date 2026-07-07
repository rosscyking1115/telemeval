"""ESA-ADB reference format reader.

Reads the ESA Anomaly Dataset benchmark label files (``labels.csv`` +
``anomaly_types.csv``) into the telemeval label contract. telemeval does not
redistribute the dataset itself (CC BY 3.0 IGO; Zenodo 10.5281/zenodo.15237121).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from telemeval.contract import validate_labels
from telemeval.errors import SchemaError

# The event-metadata columns joined from anomaly_types.csv by the official
# ESA-ADB evaluator.
ESA_ADB_ANOMALY_TYPE_COLUMNS = ("Category", "Dimensionality", "Locality", "Length")

# The default benchmark tables evaluate all events except communication gaps.
ESA_ADB_DEFAULT_EXCLUDE_CATEGORIES = ("Communication Gap",)

__all__ = [
    "ESA_ADB_ANOMALY_TYPE_COLUMNS",
    "ESA_ADB_DEFAULT_EXCLUDE_CATEGORIES",
    "read_channels",
    "read_labels",
]


def read_channels(channels_csv: str | Path) -> dict[str, object]:
    """Read ESA-ADB ``channels.csv`` into a subsystem mapping and target list.

    Returns ``{"subsystems_mapping": {subsystem: [channels...]},
    "target_channels": [...]}`` — the mapping feeds
    ``evaluate(metric_options={"channel_aware": {"subsystems_mapping": ...}})``;
    ESA-ADB evaluates only channels marked ``Target == "YES"``.
    """

    frame = pd.read_csv(channels_csv)
    for column in ("Channel", "Subsystem"):
        if column not in frame.columns:
            raise SchemaError(f"channels.csv is missing required column {column!r}")

    mapping = {
        str(subsystem): [str(channel) for channel in group]
        for subsystem, group in frame.groupby("Subsystem")["Channel"]
    }
    target_channels = (
        [str(c) for c in frame.loc[frame["Target"] == "YES", "Channel"]]
        if "Target" in frame.columns
        else [str(c) for c in frame["Channel"]]
    )
    return {"subsystems_mapping": mapping, "target_channels": target_channels}


def read_labels(
    labels_csv: str | Path,
    anomaly_types_csv: str | Path,
) -> pd.DataFrame:
    """Read ESA-ADB ``labels.csv`` joined with ``anomaly_types.csv`` metadata.

    Returns validated labels in the telemeval contract shape (``ID, Channel,
    StartTime, EndTime`` plus the four official anomaly-type columns).
    """

    labels = pd.read_csv(labels_csv)
    anomaly_types = pd.read_csv(anomaly_types_csv)
    return validate_labels(
        labels,
        anomaly_types,
        metadata_columns=list(ESA_ADB_ANOMALY_TYPE_COLUMNS),
    )
