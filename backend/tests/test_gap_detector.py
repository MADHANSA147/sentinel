"""
Tests for Module 09 — Gap Detector.
Run with: pytest tests/test_gap_detector.py -v

Critical assertions:
  1. The 6-hour gap between MSG-0028 and MSG-0029 MUST be flagged as anomaly
  2. Overnight silence (09:00 → 15:00, crossing no active window) must NOT
     be flagged as the primary anomaly
"""

from __future__ import annotations

import json
import pathlib
from datetime import datetime, timezone

import pytest

from app.models.raw_message import RawMessage
from app.agents.module_08_timeline import compute_gap_seconds
from app.agents.module_04_baseline import behavioral_print, is_in_active_window
from app.agents.module_09_gap import gap_detector

DATASET_1 = (
    pathlib.Path(__file__).parent.parent.parent
    / "data" / "synthetic" / "whatsapp_conversation_40_messages.json"
)


def _load_and_validate(filepath: pathlib.Path) -> list[RawMessage]:
    """Load JSON dataset and validate each record through RawMessage."""
    raw = json.loads(filepath.read_text(encoding="utf-8"))
    validated = []
    for record in raw:
        try:
            validated.append(RawMessage.model_validate(record))
        except Exception:
            pass
    return validated


def _messages_to_timeline_entry(messages: list[RawMessage], sender: str, receiver: str) -> list[dict]:
    """Build a timeline list for a given sender→receiver pair."""
    return sorted(
        [
            {"message_id": m.message_id, "timestamp": m.timestamp}
            for m in messages
            if m.timestamp and m.sender_id == sender and m.receiver_id == receiver
        ],
        key=lambda x: x["timestamp"],
    )


def _build_combined_timeline(messages: list[RawMessage]) -> list[dict]:
    """All messages for U-101↔U-102 combined and sorted by timestamp."""
    return sorted(
        [
            {"message_id": m.message_id, "timestamp": m.timestamp}
            for m in messages
            if m.timestamp and m.sender_id in ("U-101", "U-102") and m.receiver_id in ("U-101", "U-102")
        ],
        key=lambda x: x["timestamp"],
    )


class TestDataset1GapDetector:
    """Tests against Dataset 1 — whatsapp_conversation_40_messages.json"""

    def setup_method(self):
        self.messages = _load_and_validate(DATASET_1)

    def test_dataset_loads_40_messages(self):
        assert len(self.messages) == 40, f"Expected 40 messages, got {len(self.messages)}"

    def test_no_quarantined_messages(self):
        """Dataset 1 is clean — zero messages should be quarantined."""
        quarantined = [m for m in self.messages if m.flags.is_quarantined]
        assert len(quarantined) == 0, f"Expected 0 quarantined, got {len(quarantined)}"

    def test_timestamps_are_utc(self):
        """All timestamps must be UTC-aware."""
        for msg in self.messages:
            if msg.timestamp:
                assert msg.timestamp.tzinfo is not None, f"Timestamp not tz-aware: {msg.message_id}"
                assert msg.timestamp.utcoffset().total_seconds() == 0, (
                    f"Timestamp not UTC: {msg.message_id}"
                )

    def test_timeline_strictly_increasing(self):
        """The combined timeline for U-101↔U-102 must be strictly increasing."""
        timeline = _build_combined_timeline(self.messages)
        assert len(timeline) > 0, "No timeline entries found"
        for i in range(1, len(timeline)):
            assert timeline[i]["timestamp"] > timeline[i - 1]["timestamp"], (
                f"Timeline not strictly increasing at {timeline[i]['message_id']}"
            )

    def test_msg_0028_to_0029_gap_is_exactly_6_hours(self):
        """
        The gap between MSG-0028 and MSG-0029 must be exactly 6 hours (21600 seconds).
        This is the engineered anomaly proving Module 09 can find it.
        """
        timeline = _build_combined_timeline(self.messages)
        gaps = compute_gap_seconds(timeline)

        target_gap = next(
            (g for g in gaps if g["before"] == "MSG-0028" and g["after"] == "MSG-0029"),
            None,
        )
        assert target_gap is not None, "Gap between MSG-0028 and MSG-0029 not found"
        assert target_gap["gap_seconds"] == 21600.0, (
            f"Expected 21600s gap, got {target_gap['gap_seconds']}s"
        )

    def test_gap_detector_flags_msg_0028_to_0029(self):
        """
        Module 09 MUST flag the MSG-0028→MSG-0029 gap as an anomaly.
        It falls inside the active hours window (15:20→21:20).
        """
        timeline = _build_combined_timeline(self.messages)
        timelines_state = {"U-101->U-102": timeline}

        # Build baseline from the full conversation
        baseline_state = behavioral_print({"timelines": timelines_state})

        # Run gap detection
        gap_state = gap_detector({**baseline_state, "timelines": timelines_state})
        flagged = [
            a for a in gap_state["gap_alerts"]
            if a["before"] == "MSG-0028" and a["after"] == "MSG-0029"
            and a["flagged_as_anomaly"]
        ]
        assert len(flagged) == 1, (
            "Gap between MSG-0028 and MSG-0029 was NOT flagged as anomaly — "
            "check baseline active-hours calculation"
        )

    def test_overnight_gap_is_not_primary_anomaly(self):
        """
        The overnight span must NOT be flagged as the primary (highest-score) anomaly.
        Module 09 is baseline-relative — overnight silence ≠ anomaly.
        """
        timeline = _build_combined_timeline(self.messages)
        timelines_state = {"U-101->U-102": timeline}
        baseline_state = behavioral_print({"timelines": timelines_state})
        gap_state = gap_detector({**baseline_state, "timelines": timelines_state})

        flagged = sorted(
            [a for a in gap_state["gap_alerts"] if a["flagged_as_anomaly"]],
            key=lambda x: x["gap_seconds"],
            reverse=True,
        )

        assert len(flagged) > 0, "No gaps flagged at all — check gap_detector logic"

        # The largest flagged gap must be MSG-0028→MSG-0029, not an overnight span
        top_gap = flagged[0]
        assert top_gap["before"] == "MSG-0028" and top_gap["after"] == "MSG-0029", (
            f"Wrong primary anomaly: {top_gap['before']}→{top_gap['after']} "
            f"({top_gap['gap_hours']:.1f}h). "
            "Gap Detector appears to be using max-gap logic rather than "
            "baseline-relative anomaly detection."
        )


class TestDataset3Ingestion:
    """Tests for Module 01 ingestion against Dataset 3 (corrupted)."""

    DATASET_3 = (
        pathlib.Path(__file__).parent.parent.parent
        / "data" / "synthetic" / "corrupted_whatsapp_30_messages.json"
    )

    def setup_method(self):
        raw = json.loads(self.DATASET_3.read_text(encoding="utf-8"))
        self.messages = []
        for record in raw:
            try:
                self.messages.append(RawMessage.model_validate(record))
            except Exception:
                pass

    def test_dataset3_validates_all_30(self):
        """All 30 records must be parsed (never dropped)."""
        assert len(self.messages) == 30, f"Expected 30, got {len(self.messages)}"

    def test_dataset3_quarantines_exactly_9(self):
        """Exactly 9 records (30%) must be quarantined."""
        quarantined = [m for m in self.messages if m.flags.is_quarantined]
        assert len(quarantined) == 9, (
            f"Expected 9 quarantined, got {len(quarantined)}. "
            f"IDs: {[m.message_id for m in quarantined]}"
        )
