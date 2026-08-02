"""Tests for the read-only synthetic dataset diagnostics endpoint."""

import asyncio
from pathlib import Path

from app.api.pipeline import _CASE_TO_FILE, _DATA_DIR, debug_datasets


def test_debug_datasets_reports_resolved_dataset_paths() -> None:
    """The endpoint reports each configured dataset without invoking ingestion."""
    payload = asyncio.run(debug_datasets())

    assert payload["data_dir"] == str(_DATA_DIR)
    assert payload["data_dir_exists"] is True
    assert Path(payload["pipeline_file"]).name == "pipeline.py"
    assert set(payload["datasets"]) == set(_CASE_TO_FILE)
    assert all(item["exists"] is True for item in payload["datasets"].values())
    assert payload["data_dir_files"] == sorted(_CASE_TO_FILE.values())
