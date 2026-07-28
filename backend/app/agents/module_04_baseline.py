"""
Module 04 — Behavioral Print (LangGraph node).
Builds a per-person-pair baseline of normally-active hours from Module 08's
timeline, then marks which hours are "active" for Module 09 to use.

Key invariant: do NOT flag silence outside normal active hours (e.g. overnight)
as anomalous — only flag silence that falls inside the baseline active window.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


# Minimum fraction of messages within a given hour for it to count as "active"
_ACTIVE_HOUR_THRESHOLD_FRACTION = 0.05  # hour must contain ≥5% of messages


def behavioral_print(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: for each (sender→receiver) timeline in state['timelines'],
    compute the set of normally-active UTC hours (0-23) and store as baseline.

    State outputs:
        baselines: dict mapping pair_key -> {'active_hours': set[int], ...}
    """
    timelines: dict[str, list[dict]] = state.get("timelines", {})
    baselines: dict[str, dict] = {}

    for pair_key, messages in timelines.items():
        if not messages:
            continue

        # Count how many messages fall in each UTC hour
        hour_counter: Counter = Counter(
            msg["timestamp"].hour for msg in messages
        )
        total = sum(hour_counter.values())

        # An hour is "active" if it carries at least the threshold fraction
        active_hours = {
            hour
            for hour, count in hour_counter.items()
            if count / total >= _ACTIVE_HOUR_THRESHOLD_FRACTION
        }

        baselines[pair_key] = {
            "active_hours": active_hours,
            "hour_distribution": dict(hour_counter),
            "total_messages": total,
            "message_span_seconds": (
                (messages[-1]["timestamp"] - messages[0]["timestamp"]).total_seconds()
                if len(messages) > 1
                else 0
            ),
        }

    state["baselines"] = baselines
    return state


def is_in_active_window(hour: int, active_hours: set[int]) -> bool:
    """Return True if the given UTC hour falls inside the established baseline."""
    return hour in active_hours
