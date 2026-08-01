"""
Tests for Module 09 — Gap Detector.
Run with: pytest tests/test_gap_detector.py -v

Critical assertions:
  1. The 6-hour gap between MSG-0028 and MSG-0029 MUST be flagged as anomaly
  2. Overnight silence (09:00 → 15:00, crossing no active window) must NOT
     be flagged as the primary anomaly
"""

from __future__ import annotations

import asyncio
import json
import pathlib
from collections import Counter
from datetime import datetime, timezone

import networkx as nx
import pytest

from app.models.raw_message import RawMessage
from app.agents.module_08_timeline import compute_gap_seconds
from app.agents.module_04_baseline import behavioral_print, is_in_active_window
from app.agents.module_09_gap import gap_detector
from app.agents.module_11_context import exculpatory_context
from app.agents import module_07_roles
from app.agents.module_07_roles import role_discovery
from app.api import dashboard as dashboard_api
from app.services.graph_db import _aggregate_message_edges

DATASET_1 = (
    pathlib.Path(__file__).parent.parent.parent
    / "data" / "synthetic" / "whatsapp_conversation_40_messages.json"
)
DATASET_4 = (
    pathlib.Path(__file__).parent.parent.parent
    / "data" / "synthetic" / "network_test_45_messages.json"
)
DATASET_2 = (
    pathlib.Path(__file__).parent.parent.parent
    / "data" / "synthetic" / "whatsapp_synthetic_35_messages.json"
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


def _build_pair_timelines(messages: list[RawMessage]) -> dict[str, list[dict]]:
    """Build unordered per-pair timelines matching Module 08 behavior."""
    timelines: dict[str, list[dict]] = {}
    for msg in messages:
        if not msg.timestamp or not msg.sender_id or not msg.receiver_id:
            continue
        source, target = sorted((msg.sender_id, msg.receiver_id))
        pair_key = f"{source}->{target}"
        timelines.setdefault(pair_key, []).append(
            {
                "message_id": msg.message_id,
                "timestamp": msg.timestamp,
                "sender_id": msg.sender_id,
                "receiver_id": msg.receiver_id,
            }
        )

    for timeline in timelines.values():
        timeline.sort(key=lambda x: x["timestamp"])

    return timelines


def _dataset4_centrality(
    messages: list[RawMessage],
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute Module 06-style centrality scores directly from Dataset 4."""
    graph = nx.DiGraph()
    for msg in messages:
        if not msg.sender_id or not msg.receiver_id:
            continue
        if graph.has_edge(msg.sender_id, msg.receiver_id):
            graph[msg.sender_id][msg.receiver_id]["weight"] += 1
        else:
            graph.add_edge(msg.sender_id, msg.receiver_id, weight=1)

    pagerank = nx.pagerank(graph, alpha=0.85, weight="weight")
    betweenness = nx.betweenness_centrality(
        graph.to_undirected(), normalized=True, weight="weight"
    )
    return pagerank, betweenness


def _raw_message_dict(message: RawMessage) -> dict:
    """Return the pipeline-state shape used by dashboard tests."""
    return {
        "message_id": message.message_id,
        "timestamp": message.timestamp,
        "sender_id": message.sender_id,
        "receiver_id": message.receiver_id,
        "content": message.content,
        "platform": message.platform,
        "is_quarantined": message.flags.is_quarantined,
        "flags": message.flags.model_dump(),
    }


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


class TestDataset4NetworkRegression:
    """Regression tests for network_test_45_messages.json."""

    def setup_method(self):
        self.messages = _load_and_validate(DATASET_4)

    def test_dataset4_gap_detector_flags_zero_anomalies(self):
        """Dataset 4 has no engineered communication gaps."""
        timelines_state = _build_pair_timelines(self.messages)
        baseline_state = behavioral_print({"timelines": timelines_state})

        gap_state = gap_detector(
            {
                **baseline_state,
                "timelines": timelines_state,
                "case_id": "case-dataset4",
            }
        )

        flagged = [
            alert
            for alert in gap_state["gap_alerts"]
            if alert["flagged_as_anomaly"]
        ]

        assert flagged == []

    def test_dataset4_roles_separate_hubs_from_bridge(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """U-301/U-305 are PageRank hubs; U-304 is the bridge role."""
        pagerank, betweenness = _dataset4_centrality(self.messages)
        sent_counts: Counter[str] = Counter(
            msg.sender_id for msg in self.messages if msg.sender_id
        )
        recv_counts: Counter[str] = Counter(
            msg.receiver_id for msg in self.messages if msg.receiver_id
        )

        def fake_run_query_for_case(
            case_id: str,
            cypher: str,
            params: dict | None = None,
        ) -> list[dict]:
            assert case_id == "case-dataset4"
            if "AS sent" in cypher:
                return [{"id": pid, "sent": count} for pid, count in sent_counts.items()]
            return [
                {"id": pid, "received": count}
                for pid, count in recv_counts.items()
            ]

        monkeypatch.setattr(
            module_07_roles,
            "run_query_for_case",
            fake_run_query_for_case,
        )
        monkeypatch.setattr(
            module_07_roles,
            "write_person_properties",
            lambda updates, case_id="default": None,
        )

        role_state = role_discovery(
            {
                "case_id": "case-dataset4",
                "pagerank": pagerank,
                "betweenness": betweenness,
            }
        )

        roles = role_state["roles"]
        assert roles["U-301"]["tag"] == "Orchestrator"
        assert roles["U-305"]["tag"] == "Orchestrator"
        assert roles["U-304"]["tag"] == "Bridge"
        assert {
            person_id
            for person_id, role_info in roles.items()
            if role_info["tag"] == "Bridge"
        } == {"U-304"}

    def test_dataset4_dashboard_has_no_gap_badge_edges_and_aggregates_pairs(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Dashboard map data for Dataset 4 has weighted pair edges and no gap badges."""
        pagerank, betweenness = _dataset4_centrality(self.messages)
        unique_pairs = {
            tuple(sorted((msg.sender_id, msg.receiver_id)))
            for msg in self.messages
            if msg.sender_id and msg.receiver_id
        }
        state = {
            "risk_scores": {
                person_id: {
                    "person_id": person_id,
                    "score": 0,
                    "role_tag": "Peripheral",
                    "pagerank": pagerank.get(person_id, 0.0),
                    "betweenness": betweenness.get(person_id, 0.0),
                }
                for person_id in set(pagerank) | set(betweenness)
            },
            "gap_alerts": [],
            "centrality_summary": [],
            "roles": {},
            "theories": [],
            "coercive_matches": [],
            "raw_messages": [_raw_message_dict(msg) for msg in self.messages],
        }

        monkeypatch.setattr(
            dashboard_api,
            "get_case_state",
            lambda case_id: state if case_id == "case-dataset4" else {},
        )

        payload = asyncio.run(dashboard_api.get_dashboard("case-dataset4"))
        comm_edges = payload["graph"]["comm_edges"]

        assert len(comm_edges) == len(unique_pairs)
        assert sum(edge["message_count"] for edge in comm_edges) == len(self.messages)
        assert payload["graph"]["gap_edges"] == []
        assert all(edge["has_gap_alert"] is False for edge in comm_edges)


class TestDataset2ExculpatoryContext:
    """Dataset 2 must retain, then visibly clear, its false-positive burst."""

    def test_night_shift_context_suppresses_but_retains_alert(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        messages = _load_and_validate(DATASET_2)
        raw_messages = [_raw_message_dict(message) for message in messages]
        timelines = _build_pair_timelines(messages)
        baseline_state = behavioral_print({"timelines": timelines})
        gap_state = gap_detector({**baseline_state, "timelines": timelines})

        # The alert is first produced from the pair's cadence, then Module 11
        # applies the explicit in-case exculpatory statement.
        assert any(alert["alert_type"] == "OFF_HOURS_BURST" for alert in gap_state["gap_alerts"])
        monkeypatch.setattr(
            "app.agents.module_11_context.get_collection",
            lambda: (_ for _ in ()).throw(RuntimeError("offline")),
        )
        state = exculpatory_context({
            **gap_state,
            "timelines": timelines,
            "raw_messages": raw_messages,
            "case_id": "case-dataset2",
        })
        suppressed = [a for a in state["gap_alerts"] if a.get("suppressed")]
        assert len(suppressed) == 1
        assert "night shift" in suppressed[0]["suppression_reason"].lower()


class TestGraphAggregation:
    """Regression coverage for one weighted relationship per directed pair."""

    def test_batch_graph_edges_are_aggregated_by_sender_receiver(self):
        edges = [
            {
                "sender_id": "U-1",
                "receiver_id": "U-2",
                "message_id": "M-2",
                "timestamp_iso": "2026-07-29T10:00:00+00:00",
                "platform": "whatsapp",
            },
            {
                "sender_id": "U-1",
                "receiver_id": "U-2",
                "message_id": "M-1",
                "timestamp_iso": "2026-07-29T09:00:00+00:00",
                "platform": "whatsapp",
            },
            {
                "sender_id": "U-2",
                "receiver_id": "U-1",
                "message_id": "M-3",
                "timestamp_iso": "2026-07-29T11:00:00+00:00",
                "platform": "whatsapp",
            },
        ]

        aggregated = sorted(
            _aggregate_message_edges(edges),
            key=lambda edge: (edge["sender_id"], edge["receiver_id"]),
        )

        assert len(aggregated) == 2
        assert aggregated[0]["sender_id"] == "U-1"
        assert aggregated[0]["receiver_id"] == "U-2"
        assert aggregated[0]["message_count"] == 2
        assert aggregated[0]["message_ids"] == ["M-1", "M-2"]
        assert aggregated[1]["sender_id"] == "U-2"
        assert aggregated[1]["receiver_id"] == "U-1"
        assert aggregated[1]["message_count"] == 1


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
        """Exactly 30% of Dataset 3 records must be quarantined."""
        quarantined = [m for m in self.messages if m.flags.is_quarantined]
        assert len(quarantined) == 9, (
            f"Expected 9 quarantined, got {len(quarantined)}. "
            f"IDs: {[m.message_id for m in quarantined]}"
        )
