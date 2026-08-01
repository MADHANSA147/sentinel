"""
Module 08 - Timeline Engine (LangGraph node).
Merges every message's UTC timestamp into one strictly ordered chronological
sequence per Person pair, then stores it in the graph state for Module 04/09.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.services.graph_db import run_query_for_case


def _to_utc_datetime(value: Any) -> datetime | None:
    """Parse Neo4j/string/datetime timestamp values and normalize to UTC."""
    if value is None:
        return None
    try:
        if isinstance(value, str):
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
                timezone.utc
            )
        if hasattr(value, "to_native"):
            return value.to_native().astimezone(timezone.utc)
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None
    return None


def _append_timeline_message(
    timelines: dict[tuple[str, str], list[dict]],
    *,
    sender: str | None,
    receiver: str | None,
    message_id: str | None,
    timestamp: Any,
) -> None:
    """Append one timestamped message to the unordered person-pair timeline."""
    if not sender or not receiver or not message_id:
        return

    ts = _to_utc_datetime(timestamp)
    if ts is None:
        return

    key = tuple(sorted((sender, receiver)))
    timelines.setdefault(key, []).append(
        {
            "message_id": message_id,
            "timestamp": ts,
            "sender_id": sender,
            "receiver_id": receiver,
        }
    )


def _serialise_timelines(
    timelines: dict[tuple[str, str], list[dict]],
) -> dict[str, list[dict]]:
    """Sort each pair timeline and return JSON-safe string keys."""
    for key in timelines:
        timelines[key].sort(key=lambda x: x["timestamp"])
    return {f"{k[0]}->{k[1]}": v for k, v in timelines.items()}


def timeline_engine(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: builds a sorted timeline per unordered Person pair.

    State outputs:
        timelines: dict mapping "person_a->person_b" to sorted message dicts.
    """
    case_id: str = state.get("case_id", "default")
    raw_messages: list[dict[str, Any]] = state.get("raw_messages", [])
    timelines: dict[tuple[str, str], list[dict]] = {}

    if raw_messages:
        for msg in raw_messages:
            if msg.get("is_quarantined"):
                continue
            _append_timeline_message(
                timelines,
                sender=msg.get("sender_id"),
                receiver=msg.get("receiver_id"),
                message_id=msg.get("message_id"),
                timestamp=msg.get("timestamp"),
            )
        state["timelines"] = _serialise_timelines(timelines)
        return state

    try:
        records = run_query_for_case(
            case_id,
            """
            MATCH (s:Person {case_id: $case_id})-[rel:MESSAGED]->(r:Person {case_id: $case_id})
            WHERE rel.timestamps IS NOT NULL OR rel.timestamp IS NOT NULL
            RETURN s.id AS sender,
                   r.id AS receiver,
                   coalesce(
                       rel.timestamped_message_ids,
                       rel.message_ids,
                       [rel.message_id]
                   ) AS message_ids,
                   coalesce(rel.timestamps, [rel.timestamp]) AS timestamps
            """,
        )
    except Exception as exc:
        print(f"[WARN] Timeline Engine graph query failed: {exc}")
        records = []

    for rec in records:
        sender = rec.get("sender")
        receiver = rec.get("receiver")
        message_ids = rec.get("message_ids") or []
        timestamps = rec.get("timestamps") or []
        for index, message_id in enumerate(message_ids):
            timestamp = timestamps[index] if index < len(timestamps) else None
            _append_timeline_message(
                timelines,
                sender=sender,
                receiver=receiver,
                message_id=message_id,
                timestamp=timestamp,
            )

    state["timelines"] = _serialise_timelines(timelines)
    return state


def compute_gap_seconds(tl: list[dict]) -> list[dict]:
    """
    Return gap records between consecutive messages.

    Each record has before, after, and gap_seconds fields.
    """
    gaps: list[dict[str, Any]] = []
    for i in range(1, len(tl)):
        delta = (tl[i]["timestamp"] - tl[i - 1]["timestamp"]).total_seconds()
        gaps.append(
            {
                "before": tl[i - 1]["message_id"],
                "after": tl[i]["message_id"],
                "gap_seconds": delta,
            }
        )
    return gaps
