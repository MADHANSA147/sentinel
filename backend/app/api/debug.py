"""Read-only runtime diagnostics for synthetic dataset discovery."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter

from app.api import pipeline

router = APIRouter(tags=["debug"])


@router.get("/debug/datasets")
async def debug_datasets() -> dict[str, Any]:
    """Return the resolved synthetic-dataset paths without running ingestion."""
    datasets = {
        case_id: pipeline._DATA_DIR / filename
        for case_id, filename in pipeline._CASE_TO_FILE.items()
    }
    return {
        "data_dir": str(pipeline._DATA_DIR),
        "data_dir_exists": pipeline._DATA_DIR.exists(),
        "datasets": {
            case_id: {"path": str(path), "exists": path.exists()}
            for case_id, path in datasets.items()
        },
        "current_working_directory": str(Path.cwd()),
        "pipeline_file": pipeline.__file__,
        "data_dir_files": (
            sorted(path.name for path in pipeline._DATA_DIR.iterdir() if path.is_file())
            if pipeline._DATA_DIR.exists()
            else []
        ),
    }
