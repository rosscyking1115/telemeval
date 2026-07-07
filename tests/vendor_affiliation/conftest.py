"""The vendored upstream tests read their paper-reproduction fixtures via
cwd-relative ``data/*.gz`` globs; run them from this directory."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _run_from_vendor_test_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(Path(__file__).parent)
    # Guard: the fixtures must exist or the data tests silently pass on
    # empty globs.
    assert os.path.isdir("data"), "vendored affiliation test data missing"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    # Upstream's `metrics.test_events` is an input *validator* whose name
    # matches pytest's test pattern; it is imported into the vendored
    # test_metrics.py and must not be collected as a test.
    items[:] = [
        item
        for item in items
        if not (item.name == "test_events" and "vendor_affiliation" in str(item.fspath))
    ]
