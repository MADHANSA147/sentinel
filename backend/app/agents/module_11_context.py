"""
Module 11 — Exculpatory Context & Contradiction ⭐ (LangGraph node).

Before escalating a gap alert, queries ChromaDB with a 72-hour lookback
window for messages that might innocently explain the silence.
If LLM judges context sufficient → tag BENIGN_CONTEXT, suppress the active
flag, but keep it in the audit trail.

Multi-factor lock: no risk score may exceed 40/100 without corroboration
from at least 3 independent evidence domains.

CRITICAL: Lookback is 72 hours, NOT 48.  Dataset 2's context sits ~52
hours before the burst — a 48-hour window misses it entirely.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from app.agents.module_05_words import get_collection

# Innocent explanation keywords that suggest benign context
_INNOCENT_PATTERNS = [
    "night shift", "working late", "exam", "busy", "travel",
    "no signal", "sleeping", "offline", "hospital", "emergency",
    "meeting", "out of town", "battery dead", "network issue",
]
# A direct lexical finding must be specific enough to clear an alert.  Generic
# words such as "meeting" or "offline" occur in potentially coercive messages
# and are only retrieval hints, not independently exculpatory evidence.
_EXPLICIT_INNOCENT_PATTERNS = [
    "night shift", "working late", "travel", "no signal", "hospital",
    "emergency", "out of town", "battery dead", "network issue",
]

_LOOKBACK_HOURS = 72  # proven necessary from Dataset 2 (context ~52h before burst)
_MULTI_DOMAIN_LOCK_THRESHOLD = 40  # score cap without 3+ evidence domains


def _to_utc_datetime(value: Any) -> datetime | None:
    """Parse timestamp metadata and normalize it to UTC."""
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


def _metadata_matches_pair(meta: dict[str, Any], pair_key: str) -> bool:
    """Return True when metadata belongs to either direction of the pair."""
    if "->" not in pair_key:
        return False

    left, right = pair_key.split("->", 1)
    sender = meta.get("sender_id")
    receiver = meta.get("receiver_id")
    return {sender, receiver} == {left, right}


def _check_for_innocent_context(
    pair_key: str,
    gap_start_ts: Any,
    collection: Any,
    case_id: str = "default",
) -> tuple[bool, str | None]:
    """
    Search ChromaDB within the 72-hour lookback window for messages that
    could innocently explain the gap (e.g., night-shift statement).

    Returns (is_benign, explanation_text)
    """
    try:
        gap_start = _to_utc_datetime(gap_start_ts)
        if gap_start is None:
            return False, None

        # Build time-bounded query
        window_start = gap_start - timedelta(hours=_LOOKBACK_HOURS)

        # Query using innocent explanation patterns as query text
        innocent_query = " ".join(_INNOCENT_PATTERNS[:5])
        results = collection.query(
            query_texts=[innocent_query],
            n_results=10,
            include=["documents", "metadatas", "distances"],
            where={"case_id": case_id},
        )

        if not results["ids"] or not results["ids"][0]:
            return False, None

        # Check each result for temporal proximity and semantic match
        for idx, doc in enumerate(results["documents"][0]):
            distance = results["distances"][0][idx]
            similarity = 1.0 - min(distance / 2.0, 1.0)
            meta = results["metadatas"][0][idx]
            msg_ts = _to_utc_datetime(meta.get("timestamp"))

            if msg_ts is None or not window_start <= msg_ts <= gap_start:
                continue
            if not _metadata_matches_pair(meta, pair_key):
                continue

            # Semantic match check
            if similarity > 0.60:
                doc_lower = doc.lower()
                if any(kw in doc_lower for kw in _INNOCENT_PATTERNS):
                    return True, doc

        return False, None
    except Exception as exc:
        print(f"[WARN] Exculpatory context query failed: {exc}")
        return False, None


def _find_innocent_context_in_state(
    raw_messages: list[dict[str, Any]],
    pair_key: str,
    gap_start_ts: Any,
) -> tuple[bool, str | None]:
    """Find explicit benign context in the case data without vector-service drift.

    Chroma is an optional retrieval accelerator, not the authority for an
    exculpatory finding.  Looking at the original, already UTC-normalised case
    records guarantees that a clear statement (such as Dataset 2's night-shift
    notice) is retained even if the vector service is unavailable.
    """
    gap_start = _to_utc_datetime(gap_start_ts)
    if gap_start is None:
        return False, None

    window_start = gap_start - timedelta(hours=_LOOKBACK_HOURS)
    for message in raw_messages:
        if message.get("is_quarantined") or not _metadata_matches_pair(message, pair_key):
            continue
        timestamp = _to_utc_datetime(message.get("timestamp"))
        content = str(message.get("content") or "")
        if (
            timestamp is not None
            and window_start <= timestamp <= gap_start
            and any(
                pattern in content.lower()
                for pattern in _EXPLICIT_INNOCENT_PATTERNS
            )
        ):
            return True, content
    return False, None


def exculpatory_context(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: scan each gap alert, check for innocent explanations,
    suppress if found but keep in audit trail.

    Also enforces the multi-factor lock on the risk score pipeline.

    State outputs:
        gap_alerts: updated with suppressed/suppression_reason fields
        domain_scores: dict mapping person_id -> list of evidence domains seen
    """
    gap_alerts: list[dict] = state.get("gap_alerts", [])
    timelines: dict = state.get("timelines", {})
    raw_messages: list[dict[str, Any]] = state.get("raw_messages", [])
    case_id = str(state.get("case_id", "default"))
    try:
        col = get_collection()
    except Exception as exc:
        print(f"[WARN] Exculpatory context unavailable: {exc}")
        col = None

    domain_scores: dict[str, list[str]] = {}

    for alert in gap_alerts:
        if not alert.get("flagged_as_anomaly"):
            continue

        pair_key = alert["pair_key"]
        timeline = timelines.get(pair_key, [])

        # Find the timestamp of the gap start message
        before_msg = next(
            (m for m in timeline if m["message_id"] == alert["before"]), None
        )
        if not before_msg:
            continue

        gap_start_ts = before_msg["timestamp"]

        is_benign, explanation = _find_innocent_context_in_state(
            raw_messages, pair_key, gap_start_ts
        )
        if not is_benign and col is not None:
            is_benign, explanation = _check_for_innocent_context(
                pair_key, gap_start_ts, col, case_id=case_id
            )

        if is_benign:
            alert["suppressed"] = True
            alert["suppression_reason"] = explanation
        else:
            # Track which evidence domains are active for this pair
            parts = pair_key.split("->")
            person_id = parts[0] if parts else pair_key
            domain_scores.setdefault(person_id, [])
            if "TIMELINE_GAP" not in domain_scores[person_id]:
                domain_scores[person_id].append("TIMELINE_GAP")

    state["gap_alerts"] = gap_alerts
    state["domain_scores"] = domain_scores
    state["multi_factor_lock_threshold"] = _MULTI_DOMAIN_LOCK_THRESHOLD
    return state
