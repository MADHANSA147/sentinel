"""
Module 08 — Timeline Engine (LangGraph node).
Merges every message's UTC timestamp into one strictly-ordered chronological
sequence per Person pair, then stores it in the graph state for Module 04/09.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.services.graph_db import run_query


def timeline_engine(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: builds a sorted timeline per (sender, receiver) pair.

    State outputs:
        timelines: dict mapping (sender_id, receiver_id) -> sorted list of
                   {'message_id': str, 'timestamp': datetime} dicts.
    """
    # Fetch all MESSAGED edges that carry a timestamp from Neo4j
    records = run_query(
        """
        MATCH (s:Person)-[rel:MESSAGED]->(r:Person)
        WHERE rel.timestamp IS NOT NULL
        RETURN s.id AS sender, r.id AS receiver,
               rel.message_id AS message_id, rel.timestamp AS ts
        ORDER BY rel.timestamp ASC
        """
    )

    timelines: dict[tuple[str, str], list[dict]] = {}

    for rec in records:
        key = (rec["sender"], rec["receiver"])
        ts_raw = rec["ts"]
        # ts may come back as a neo4j DateTime or a string — normalise to datetime
        if isinstance(ts_raw, str):
            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00")).astimezone(
                timezone.utc
            )
        elif hasattr(ts_raw, "to_native"):  # neo4j DateTime
            ts = ts_raw.to_native().astimezone(timezone.utc)
        else:
            ts = ts_raw  # already a datetime

        timelines.setdefault(key, []).append(
            {"message_id": rec["message_id"], "timestamp": ts}
        )

    # Ensure each list is sorted (should already be from ORDER BY, but belt-and-braces)
    for key in timelines:
        timelines[key].sort(key=lambda x: x["timestamp"])

    # Serialise keys for JSON-safe state storage
    state["timelines"] = {
        f"{k[0]}->{k[1]}": v for k, v in timelines.items()
    }
    return state


def compute_gap_seconds(tl: list[dict]) -> list[dict]:
    """
    Helper: returns a list of gap records between consecutive messages.
    Each record: {'before': message_id, 'after': message_id, 'gap_seconds': float}
    """
    gaps = []
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
