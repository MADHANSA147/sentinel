"""
Module 09 — Gap Detector (INTENTIONALLY BROKEN for Codex self-correction demo).

THIS VERSION uses naive max-gap logic — it flags the single largest gap
regardless of active hours. This is the exact bug the blueprint warns about.

The tests in test_gap_detector.py will FAIL because:
  - Dataset 1 overnight gap (09:00→15:00, 6h) is the same size as MSG-0028→MSG-0029
  - But the overnight gap doesn't fall inside active hours
  - This naive implementation cannot tell the difference

Codex's job: diagnose the failure, replace this with baseline-relative detection.
"""

from __future__ import annotations

from typing import Any

from app.agents.module_08_timeline import compute_gap_seconds
from app.services.graph_db import get_driver

# BUG: hardcoded minimum — no baseline comparison at all
_MIN_GAP_SECONDS = 3600


def gap_detector(state: dict[str, Any]) -> dict[str, Any]:
    """
    BROKEN: Flags the largest gap(s) regardless of whether they fall inside
    normally-active hours. This is exactly the 'max-gap' anti-pattern.

    Will fail test_overnight_gap_is_not_primary_anomaly because:
    - The overnight silence in Dataset 1 is ~21,600 seconds
    - The MSG-0028->MSG-0029 gap is also 21,600 seconds
    - This implementation cannot distinguish between them
    - It flags BOTH as anomalies, so the overnight gap becomes the "primary"
      (first in the list, since it comes chronologically before the real anomaly)
    """
    timelines: dict[str, list[dict]] = state.get("timelines", {})

    # BUG: ignoring baselines entirely
    all_alerts: list[dict] = []

    for pair_key, messages in timelines.items():
        gaps = compute_gap_seconds(messages)

        if not gaps:
            continue

        # BUG: find the single largest gap and flag it, regardless of time-of-day
        max_gap = max(gaps, key=lambda g: g["gap_seconds"])

        for gap in gaps:
            flagged = gap["gap_seconds"] >= max_gap["gap_seconds"] * 0.9  # flags all "large" gaps

            before_msg = next((m for m in messages if m["message_id"] == gap["before"]), None)
            after_msg = next((m for m in messages if m["message_id"] == gap["after"]), None)

            alert = {
                "pair_key": pair_key,
                "before": gap["before"],
                "after": gap["after"],
                "gap_seconds": gap["gap_seconds"],
                "gap_hours": gap["gap_seconds"] / 3600,
                "before_hour": before_msg["timestamp"].hour if before_msg else 0,
                "after_hour": after_msg["timestamp"].hour if after_msg else 0,
                "midpoint_hour": 12,  # BUG: hardcoded, not computed
                "flagged_as_anomaly": flagged,
                "suppressed": False,
                "suppression_reason": None,
            }
            all_alerts.append(alert)

    state["gap_alerts"] = all_alerts
    return state


def _write_gap_edge(pair_key: str, gap: dict) -> None:
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
            session.run(cypher, sender=sender, receiver=receiver,
                        before=gap["before"], after=gap["after"],
                        gap_seconds=gap["gap_seconds"])
    except Exception as exc:
        print(f"[WARN] Could not write gap edge: {exc}")
