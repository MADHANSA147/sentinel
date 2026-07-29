"""
Module 09 — Gap Detector ⭐ (LangGraph node) — core differentiator.

Detects ABSENCE — specifically silence that falls inside a person-pair's
normally-active hours (Module 04 baseline).

CRITICAL algorithm invariant:
  Do NOT flag the single largest gap.
  Do NOT flag overnight silence by default.
  ONLY flag a gap whose midpoint (or both endpoints) falls inside the
  baseline active-hours window.

Proven necessary: naive max-gap on Dataset 1 flags the harmless overnight
silence and misses the real 6-hour daytime anomaly between MSG-0028 and
MSG-0029.
"""

from __future__ import annotations

from typing import Any

from app.agents.module_04_baseline import is_in_active_window
from app.agents.module_08_timeline import compute_gap_seconds
from app.services.graph_db import get_driver

# Minimum gap duration (seconds) for consideration as an anomaly candidate
_MIN_GAP_SECONDS = 3600  # 1 hour


def gap_detector(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: scan each person-pair timeline for gaps that fall
    inside the established active-hours baseline.

    State outputs:
        gap_alerts: list of gap dicts with keys:
            pair_key, before, after, gap_seconds, gap_hours,
            midpoint_hour, flagged_as_anomaly
    """
    timelines: dict[str, list[dict]] = state.get("timelines", {})
    baselines: dict[str, dict] = state.get("baselines", {})

    all_alerts: list[dict] = []

    for pair_key, messages in timelines.items():
        baseline = baselines.get(pair_key, {})
        active_hours: set[int] = baseline.get("active_hours", set())

        gaps = compute_gap_seconds(messages)

        for gap in gaps:
            if gap["gap_seconds"] < _MIN_GAP_SECONDS:
                continue

            # Find the messages that bound this gap to get their hours
            before_msg = next(
                (m for m in messages if m["message_id"] == gap["before"]), None
            )
            after_msg = next(
                (m for m in messages if m["message_id"] == gap["after"]), None
            )

            if not before_msg or not after_msg:
                continue

            before_hour = before_msg["timestamp"].hour
            after_hour = after_msg["timestamp"].hour

            # Midpoint hour of the silence window
            mid_ts = before_msg["timestamp"] + (
                after_msg["timestamp"] - before_msg["timestamp"]
            ) / 2
            midpoint_hour = mid_ts.hour

            # ── Baseline-relative decision ─────────────────────────────────
            # Flag if both endpoints of the gap were in the active window,
            # meaning the silence interrupted an active conversation.
            both_endpoints_active = is_in_active_window(
                before_hour, active_hours
            ) and is_in_active_window(after_hour, active_hours)

            flagged = both_endpoints_active

            alert = {
                "pair_key": pair_key,
                "before": gap["before"],
                "after": gap["after"],
                "gap_seconds": gap["gap_seconds"],
                "gap_hours": gap["gap_seconds"] / 3600,
                "before_hour": before_hour,
                "after_hour": after_hour,
                "midpoint_hour": midpoint_hour,
                "flagged_as_anomaly": flagged,
                "suppressed": False,  # Module 11 may set this to True
                "suppression_reason": None,
            }
            all_alerts.append(alert)

            # Write TIMELINE_GAP relationship into Neo4j if flagged
            if flagged and before_msg and after_msg:
                _write_gap_edge(pair_key, gap)

    state["gap_alerts"] = all_alerts
    return state


def _write_gap_edge(pair_key: str, gap: dict) -> None:
    """Persist a TIMELINE_GAP edge between the two bounding Person nodes."""
    try:
        parts = pair_key.split("->")
        if len(parts) != 2:
            return
        sender, receiver = parts
        driver = get_driver()
        cypher = """
        MATCH (s:Person {id: $sender}), (r:Person {id: $receiver})
        MERGE (s)-[g:TIMELINE_GAP {before_msg: $before, after_msg: $after}]->(r)
        SET g.gap_seconds = $gap_seconds
        """
        with driver.session() as session:
            session.run(
                cypher,
                sender=sender,
                receiver=receiver,
                before=gap["before"],
                after=gap["after"],
                gap_seconds=gap["gap_seconds"],
            )
    except Exception as exc:
        print(f"[WARN] Could not write gap edge: {exc}")
