"""SENTINEL — pipeline trigger endpoint."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from app.agents.orchestrator import run_pipeline
from app.services.ledger import clear_ledger, log_step
from app.models.raw_message import RawMessage
from app.services.graph_db import batch_load_graph
from app.agents.module_05_words import embed_messages

router = APIRouter(prefix="/api/v1", tags=["pipeline"])
logger = logging.getLogger(__name__)

# In-memory case state (keyed by case_id)
_case_states: dict[str, dict] = {}

# Map case_id → synthetic data filename
_CASE_TO_FILE: dict[str, str] = {
    "case-dataset1": "whatsapp_conversation_40_messages.json",
    "case-dataset2": "whatsapp_synthetic_35_messages.json",
    "case-dataset3": "corrupted_whatsapp_30_messages.json",
    "case-dataset4": "network_test_45_messages.json",
}

# Locate the data/synthetic directory relative to this file
_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "synthetic"


def _record_value(record: Any, key: str, default: Any = None) -> Any:
    """Read a key from a raw JSON object without assuming it is a dict."""
    return record.get(key, default) if isinstance(record, dict) else default


def _record_message_id(record: Any, index: int) -> str:
    """Return a stable placeholder id for hard-invalid records."""
    raw_id = _record_value(record, "message_id")
    return str(raw_id) if raw_id else f"<missing-message-id-{index + 1}>"


def _message_state(msg: RawMessage, case_id: str) -> dict[str, Any]:
    """Serialize a validated RawMessage into the pipeline state shape."""
    return {
        "case_id": case_id,
        "message_id": msg.message_id,
        "timestamp": msg.timestamp,
        "sender_id": msg.sender_id,
        "receiver_id": msg.receiver_id,
        "content": msg.content,
        "platform": msg.platform,
        "is_quarantined": msg.flags.is_quarantined,
        "flags": msg.flags.model_dump(),
    }


def _quarantined_record_state(
    record: Any,
    *,
    case_id: str,
    message_id: str,
    error: Exception,
) -> dict[str, Any]:
    """Serialize a record that failed validation without dropping it."""
    content = _record_value(record, "content")
    sender_id = _record_value(record, "sender_id")
    timestamp = _record_value(record, "timestamp")
    return {
        "case_id": case_id,
        "message_id": message_id,
        "timestamp": None,
        "sender_id": sender_id,
        "receiver_id": _record_value(record, "receiver_id"),
        "content": content,
        "platform": _record_value(record, "platform", "unknown") or "unknown",
        "is_quarantined": True,
        "flags": {
            "timestamp_missing": timestamp is None,
            "sender_missing": sender_id is None,
            "content_missing": content is None,
            "content_suspect": False,
            "validation_error": str(error),
        },
    }


def _auto_ingest(case_id: str) -> dict[str, Any]:
    """
    Load the synthetic dataset for a given case_id:
    1. Read JSON file, SHA-256 hash it
    2. Validate each record via RawMessage
    3. Load edges into Neo4j with case_id partition
    4. Embed messages into ChromaDB for Module 05
    5. Return initial state with raw_messages + ingestion_hash
    """
    filename = _CASE_TO_FILE.get(case_id)
    if not filename:
        return {}

    filepath = _DATA_DIR / filename
    if not filepath.exists():
        print(f"[WARN] Dataset file not found: {filepath}")
        return {}

    raw_bytes = filepath.read_bytes()
    sha256_hash = hashlib.sha256(raw_bytes).hexdigest()

    raw_records: list[Any] = json.loads(raw_bytes)

    # Validate each record independently
    validated: list[RawMessage] = []
    quarantined_ids: list[str] = []
    raw_messages: list[dict] = []

    for index, record in enumerate(raw_records):
        try:
            msg = RawMessage.model_validate(record)
            validated.append(msg)
            if msg.flags.is_quarantined:
                quarantined_ids.append(msg.message_id)
            raw_messages.append(_message_state(msg, case_id))
        except Exception as exc:
            raw_id = _record_message_id(record, index)
            quarantined_ids.append(raw_id)
            raw_messages.append(
                _quarantined_record_state(
                    record,
                    case_id=case_id,
                    message_id=raw_id,
                    error=exc,
                )
            )

    # Build edges and load into Neo4j
    edges = []
    for msg in validated:
        if not msg.flags.is_quarantined and msg.sender_id and msg.receiver_id:
            edges.append({
                "sender_id": msg.sender_id,
                "receiver_id": msg.receiver_id,
                "message_id": msg.message_id,
                "platform": msg.platform,
                "timestamp_iso": (
                    msg.timestamp.isoformat() if msg.timestamp else None
                ),
            })

    if edges:
        try:
            batch_load_graph(edges, case_id=case_id)
        except Exception as exc:
            print(f"[WARN] Neo4j load failed during auto-ingest: {exc}")

    # Embed messages into ChromaDB for semantic search
    embed_messages(raw_messages, case_id=case_id)

    log_step(
        case_id=case_id,
        module_name="AUTO_INGEST",
        input_summary=f"Auto-ingested {filename} ({len(raw_records)} records)",
        output_summary=(
            f"Validated: {len(validated)}, "
            f"Quarantined: {len(quarantined_ids)}"
        ),
        raw_file_hash=sha256_hash,
    )

    return {
        "raw_messages": raw_messages,
        "ingestion_hash": sha256_hash,
        "quarantined_ids": quarantined_ids,
    }


@router.post("/run/{case_id}")
async def run_case(case_id: str) -> dict:
    """Trigger the full agent pipeline for a given case_id."""
    logger.info("Pipeline request started for case_id=%s", case_id)
    try:
        return await asyncio.to_thread(_run_case_sync, case_id)
    except Exception:
        logger.exception("Pipeline request failed for case_id=%s", case_id)
        raise


def _run_case_sync(case_id: str) -> dict:
    """Execute blocking pipeline work off FastAPI's event-loop thread."""
    # Clear stale state to ensure fresh results per dataset
    _case_states.pop(case_id, None)
    clear_ledger(case_id)

    # Auto-ingest the corresponding dataset file
    ingest_state = _auto_ingest(case_id)

    # Build initial state with case_id and ingested data
    initial_state = {"case_id": case_id, **ingest_state}

    state = run_pipeline(initial_state)
    _case_states[case_id] = state

    response = {
        "case_id": case_id,
        "status": "completed",
        "persons_analysed": len(state.get("risk_scores", {})),
        "gap_alerts": len(
            [a for a in state.get("gap_alerts", []) if a.get("flagged_as_anomaly")]
        ),
        "theories": state.get("theories", []),
        "simulation": state.get("simulation", {}),
    }
    logger.info("Pipeline request completed for case_id=%s", case_id)
    return response


def get_case_state(case_id: str) -> dict:
    return _case_states.get(case_id, {})
