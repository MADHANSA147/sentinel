"""
Module 09 — Gap Detector ⭐ (LangGraph node) — core differentiator.

Detects ABSENCE — specifically silence that falls inside a person-pair's
normally-active hours (Module 04 baseline).

CRITICAL algorithm invariant:
  Do NOT flag the single largest gap.
  Do NOT flag overnight silence by default.
  ONLY flag same-day silence that interrupts a baseline-active pair and is
  long relative to that pair's observed communication cadence.

Proven necessary: naive max-gap on Dataset 1 flags the harmless overnight
silence and misses the real 6-hour daytime anomaly between MSG-0028 and
MSG-0029.
"""

from __future__ import annotations

from statistics import median
from typing import Any

from app.agents.module_04_baseline import is_in_active_window
from app.agents.module_08_timeline import compute_gap_seconds
from app.services.graph_db import run_write

# Minimum gap duration (seconds) for consideration as an anomaly candidate
_MIN_GAP_SECONDS = 3 * 3600  # 3 hours; Dataset 4 has routine 1-2h pauses.
_BASELINE_GAP_MULTIPLIER = 6.0
_OFF_HOURS_BURST_MIN_MESSAGES = 5
_OFF_HOURS_BURST_WINDOW_SECONDS = 45 * 60


def _same_utc_date(before_msg: dict[str, Any], after_msg: dict[str, Any]) -> bool:
    """Return True when both messages occurred on the same UTC calendar date."""
    before_ts = before_msg.get("timestamp")
    after_ts = after_msg.get("timestamp")
    if not before_ts or not after_ts:
        return False
    return before_ts.date() == after_ts.date()


def _both_endpoints_active(
    before_msg: dict[str, Any],
    after_msg: dict[str, Any],
    active_hours: set[int],
) -> bool:
    """Return True when the gap starts and ends inside the pair's active hours."""
    before_ts = before_msg.get("timestamp")
    after_ts = after_msg.get("timestamp")
    if not before_ts or not after_ts:
        return False
    return is_in_active_window(before_ts.hour, active_hours) and is_in_active_window(
        after_ts.hour, active_hours
    )


def _routine_gap_seconds(
    gaps: list[dict[str, Any]],
    message_lookup: dict[str, dict[str, Any]],
    active_hours: set[int],
    excluded_gap: dict[str, Any],
) -> list[float]:
    """Return same-day active-window gaps, excluding the candidate being scored."""
    routine_gaps: list[float] = []
    for gap in gaps:
        same_boundary = (
            gap["before"] == excluded_gap["before"]
            and gap["after"] == excluded_gap["after"]
        )
        if same_boundary:
            continue

        before_msg = message_lookup.get(gap["before"])
        after_msg = message_lookup.get(gap["after"])
        if not before_msg or not after_msg:
            continue
        if not _same_utc_date(before_msg, after_msg):
            continue
        if not _both_endpoints_active(before_msg, after_msg, active_hours):
            continue

        routine_gaps.append(float(gap["gap_seconds"]))

    return routine_gaps


def _anomaly_threshold_seconds(routine_gaps: list[float]) -> float:
    """Baseline-relative gap threshold with a practical floor for demo data."""
    if not routine_gaps:
        return float(_MIN_GAP_SECONDS)
    return max(
        float(_MIN_GAP_SECONDS),
        median(routine_gaps) * _BASELINE_GAP_MULTIPLIER,
    )


def _is_unanswered_resume_burst(
    messages: list[dict[str, Any]], candidate_index: int
) -> bool:
    """Identify an unusual one-sided burst immediately after a long silence.

    This is intentionally narrow: an ordinary next-day conversation (Dataset 4)
    does not qualify.  It captures the Dataset 2 false positive, where one
    participant sends a dense sequence at an unusual time after a prolonged
    interruption.  The candidate remains baseline-relative because it is only
    evaluated after a gap has exceeded the pair's cadence threshold.
    """
    resumed = messages[candidate_index]
    sender = resumed.get("sender_id")
    start = resumed.get("timestamp")
    if not sender or start is None:
        return False

    burst_count = 0
    for msg in messages[candidate_index:]:
        timestamp = msg.get("timestamp")
        if (
            timestamp is None
            or (timestamp - start).total_seconds() > _OFF_HOURS_BURST_WINDOW_SECONDS
            or msg.get("sender_id") != sender
        ):
            break
        burst_count += 1

    return burst_count >= _OFF_HOURS_BURST_MIN_MESSAGES


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
    case_id: str = state.get("case_id", "default")

    all_alerts: list[dict] = []

    for pair_key, messages in timelines.items():
        baseline = baselines.get(pair_key, {})
        active_hours: set[int] = baseline.get("active_hours", set())

        gaps = compute_gap_seconds(messages)
        message_lookup = {
            m["message_id"]: m
            for m in messages
            if m.get("message_id") and m.get("timestamp")
        }

        for gap_index, gap in enumerate(gaps):
            if gap["gap_seconds"] < _MIN_GAP_SECONDS:
                continue

            # Find the messages that bound this gap to get their hours
            before_msg = message_lookup.get(gap["before"])
            after_msg = message_lookup.get(gap["after"])

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
            # Same-day gaps avoid ordinary overnight / day-to-day silence.
            # The cadence threshold prevents Dataset 4's routine 1-2h pauses
            # from being treated like engineered communication gaps.
            same_day = _same_utc_date(before_msg, after_msg)
            both_endpoints_active = _both_endpoints_active(
                before_msg, after_msg, active_hours
            )
            routine_gaps = _routine_gap_seconds(
                gaps, message_lookup, active_hours, gap
            )
            threshold_seconds = _anomaly_threshold_seconds(routine_gaps)

            standard_gap = (
                same_day
                and both_endpoints_active
                and gap["gap_seconds"] >= threshold_seconds
            )
            resume_burst = (
                not same_day
                and gap["gap_seconds"] >= threshold_seconds
                and _is_unanswered_resume_burst(messages, gap_index + 1)
            )
            flagged = standard_gap or resume_burst

            alert = {
                "pair_key": pair_key,
                "before": gap["before"],
                "after": gap["after"],
                "gap_seconds": gap["gap_seconds"],
                "gap_hours": gap["gap_seconds"] / 3600,
                "before_hour": before_hour,
                "after_hour": after_hour,
                "midpoint_hour": midpoint_hour,
                "same_utc_date": same_day,
                "baseline_gap_seconds": median(routine_gaps) if routine_gaps else None,
                "anomaly_threshold_seconds": threshold_seconds,
                "alert_type": "OFF_HOURS_BURST" if resume_burst else "TIMELINE_GAP",
                "flagged_as_anomaly": flagged,
                "suppressed": False,  # Module 11 may set this to True
                "suppression_reason": None,
            }
            all_alerts.append(alert)

            # Write TIMELINE_GAP relationship into Neo4j if flagged
            if flagged and before_msg and after_msg:
                _write_gap_edge(case_id, pair_key, gap)

    state["gap_alerts"] = all_alerts
    return state


def _write_gap_edge(case_id: str, pair_key: str, gap: dict[str, Any]) -> None:
    """Persist a TIMELINE_GAP edge between the two bounding Person nodes."""
    try:
        parts = pair_key.split("->")
        if len(parts) != 2:
            return
        sender, receiver = parts
        cypher = """
        MATCH (s:Person {id: $sender, case_id: $case_id}),
              (r:Person {id: $receiver, case_id: $case_id})
        MERGE (s)-[g:TIMELINE_GAP {before_msg: $before, after_msg: $after}]->(r)
        SET g.gap_seconds = $gap_seconds,
            g.case_id = $case_id
        """
        run_write(
            cypher,
            {
                "case_id": case_id,
                "sender": sender,
                "receiver": receiver,
                "before": gap["before"],
                "after": gap["after"],
                "gap_seconds": gap["gap_seconds"],
            },
        )
    except Exception as exc:
        print(f"[WARN] Could not write gap edge: {exc}")
