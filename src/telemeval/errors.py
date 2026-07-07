"""Typed contract errors.

Every ingestion guard raises one of these instead of letting a malformed or
leaky evaluation silently produce a number. Catch :class:`ContractError` to
handle any guard failure generically.
"""

from __future__ import annotations

__all__ = [
    "AlignmentError",
    "BinaryDomainError",
    "ContractError",
    "IntervalError",
    "MonotonicityError",
    "SchemaError",
    "TypeJoinError",
    "WindowLeakageError",
]


class ContractError(ValueError):
    """Base class for all telemeval ingestion-contract violations."""


class SchemaError(ContractError):
    """Required columns are missing or have invalid values."""


class IntervalError(ContractError):
    """A labelled interval is malformed (e.g. StartTime after EndTime)."""


class TypeJoinError(ContractError):
    """Event metadata cannot be joined (missing or duplicate IDs)."""


class AlignmentError(ContractError):
    """Prediction timestamps differ across channels."""


class MonotonicityError(ContractError):
    """Prediction timestamps are non-increasing or duplicated."""


class BinaryDomainError(ContractError):
    """Scores declared binary contain values outside {0, 1}."""


class WindowLeakageError(ContractError):
    """Labelled events lie entirely outside the prediction window.

    Such events can never be detected by the supplied predictions, so counting
    them corrupts recall (the train/test-window leakage class of bug). Pass
    window-consistent labels, or opt in explicitly with ``clip_to_window=True``.
    """
