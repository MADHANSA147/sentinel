"""
SENTINEL — POST /api/v1/ingest
SHA-256 hashes the raw file before parsing, validates every record
independently, quarantines bad ones, then loads the clean graph into Neo4j.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.models.raw_message import RawMessage, IngestResponse
from app.services.graph_db import batch_load_graph

router = APIRouter(prefix="/api/v1", tags=["ingestion"])

# Map synthetic filenames to their demo case_id partition
_FILENAME_TO_CASE: dict[str, str] = {
    "whatsapp_conversation_40_messages.json": "case-dataset1",
    "whatsapp_synthetic_35_messages.json":    "case-dataset2",
    "corrupted_whatsapp_30_messages.json":    "case-dataset3",
    "network_test_45_messages.json":          "case-dataset4",
}


def _resolve_case_id(filename: str) -> str:
    """Return the canonical case_id for this file, falling back to the stem."""
    return _FILENAME_TO_CASE.get(filename, filename.split(".")[0])


def _record_message_id(record: Any, index: int) -> str:
    """Return a stable quarantine id without assuming the record is a dict."""
    if isinstance(record, dict) and record.get("message_id"):
        return str(record["message_id"])
    return f"<missing-message-id-{index + 1}>"


def _build_edges(messages: list[RawMessage]) -> list[dict[str, Any]]:
    """Convert validated messages into edge dicts for Neo4j batch load."""
    edges = []
    for msg in messages:
        if not msg.flags.is_quarantined and msg.sender_id and msg.receiver_id:
            edges.append(
                {
                    "sender_id": msg.sender_id,
                    "receiver_id": msg.receiver_id,
                    "message_id": msg.message_id,
                    "platform": msg.platform,
                    "timestamp_iso": (
                        msg.timestamp.isoformat() if msg.timestamp else None
                    ),
                }
            )
    return edges


@router.post("/ingest", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)) -> IngestResponse:
    """
    Accept a JSON file upload, SHA-256 hash it, parse each record independently.
    Returns counts of total / validated / quarantined records and the file hash.
    """
    raw_bytes = await file.read()

    # ── Chain-of-custody: hash before any parsing ──────────────────────────
    sha256_hash = hashlib.sha256(raw_bytes).hexdigest()

    # ── Parse outer JSON ───────────────────────────────────────────────────
    try:
        raw_records: list[dict] = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc

    if not isinstance(raw_records, list):
        raise HTTPException(status_code=400, detail="Expected a JSON array at top level.")

    # ── Validate each record independently ────────────────────────────────
    validated: list[RawMessage] = []
    quarantined_ids: list[str] = []

    for index, record in enumerate(raw_records):
        try:
            msg = RawMessage.model_validate(record)
            validated.append(msg)
            if msg.flags.is_quarantined:
                quarantined_ids.append(msg.message_id)
        except Exception:
            # Even if Pydantic itself raises (e.g. missing message_id),
            # capture the raw record id if possible and quarantine.
            raw_id = _record_message_id(record, index)
            quarantined_ids.append(raw_id)

    # ── Load graph ─────────────────────────────────────────────────────────
    case_id = _resolve_case_id(file.filename or "")
    try:
        edges = _build_edges(validated)
        if edges:
            batch_load_graph(edges, case_id=case_id)
    except Exception as exc:
        # Graph load failure must not lose the ingestion result — log and continue
        print(f"[WARN] Neo4j load failed: {exc}")

    return IngestResponse(
        filename=file.filename or "upload",
        sha256_hash=sha256_hash,
        total=len(raw_records),
        validated=len(validated),
        quarantined=len(quarantined_ids),
        quarantined_ids=quarantined_ids,
    )
