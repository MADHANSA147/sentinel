"""Regression tests for case-scoped pipeline and vector-search behavior."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from app.agents import module_05_words, module_11_context
from app.api import pipeline as pipeline_api
from app.services import graph_db
from app.services.graph_db import _aggregate_message_edges


class FakeCollection:
    """Small ChromaDB stand-in that records query kwargs."""

    def __init__(self, results: dict[str, Any]) -> None:
        self.results = results
        self.calls: list[dict[str, Any]] = []

    def query(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return self.results


def test_word_patterns_filters_queries_by_case_id(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ENABLE_SEMANTIC_EMBEDDINGS", "true")
    fake = FakeCollection(
        {
            "ids": [["case-a:MSG-1"]],
            "documents": [["don't tell anyone about this"]],
            "metadatas": [[{"message_id": "MSG-1", "sender_id": "U-1"}]],
            "distances": [[0.1]],
        }
    )
    monkeypatch.setattr(module_05_words, "get_collection", lambda: fake)

    state = module_05_words.word_patterns({"case_id": "case-a"})

    assert fake.calls
    assert all(call["where"] == {"case_id": "case-a"} for call in fake.calls)
    assert state["coercive_matches"]
    assert {match["message_id"] for match in state["coercive_matches"]} == {"MSG-1"}


def test_word_patterns_uses_lightweight_fallback_when_embeddings_disabled(
    monkeypatch,
) -> None:
    """Render's default path must not initialise SentenceTransformer per request."""
    monkeypatch.delenv("ENABLE_SEMANTIC_EMBEDDINGS", raising=False)
    monkeypatch.setattr(
        module_05_words,
        "_get_chroma_collection",
        lambda: pytest.fail("embedding backend must not be constructed"),
    )

    state = module_05_words.word_patterns(
        {
            "case_id": "case-a",
            "raw_messages": [
                {
                    "message_id": "MSG-1",
                    "sender_id": "U-1",
                    "content": "Don't tell anyone about this.",
                    "is_quarantined": False,
                }
            ],
        }
    )

    assert state["coercive_matches"] == [
        {
            "query": "explicit coercive-language rule",
            "message_id": "MSG-1",
            "content": "Don't tell anyone about this.",
            "sender_id": "U-1",
            "similarity": 1.0,
        }
    ]


def test_exculpatory_context_requires_case_pair_and_lookback_window() -> None:
    gap_start = datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc)
    valid_ts = (gap_start - timedelta(hours=2)).isoformat()
    old_ts = (gap_start - timedelta(hours=73)).isoformat()

    fake = FakeCollection(
        {
            "ids": [["old", "wrong-pair", "valid"]],
            "documents": [
                [
                    "night shift caused delay",
                    "night shift caused delay",
                    "night shift caused delay",
                ]
            ],
            "metadatas": [
                [
                    {
                        "timestamp": old_ts,
                        "sender_id": "U-1",
                        "receiver_id": "U-2",
                    },
                    {
                        "timestamp": valid_ts,
                        "sender_id": "U-1",
                        "receiver_id": "U-9",
                    },
                    {
                        "timestamp": valid_ts,
                        "sender_id": "U-2",
                        "receiver_id": "U-1",
                    },
                ]
            ],
            "distances": [[0.1, 0.1, 0.1]],
        }
    )

    is_benign, explanation = module_11_context._check_for_innocent_context(
        "U-1->U-2",
        gap_start,
        fake,
        case_id="case-a",
    )

    assert fake.calls[0]["where"] == {"case_id": "case-a"}
    assert is_benign is True
    assert explanation == "night shift caused delay"


def test_exculpatory_context_rejects_out_of_window_context() -> None:
    gap_start = datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc)
    old_ts = (gap_start - timedelta(hours=73)).isoformat()
    fake = FakeCollection(
        {
            "ids": [["old"]],
            "documents": [["night shift caused delay"]],
            "metadatas": [
                [
                    {
                        "timestamp": old_ts,
                        "sender_id": "U-1",
                        "receiver_id": "U-2",
                    }
                ]
            ],
            "distances": [[0.1]],
        }
    )

    is_benign, explanation = module_11_context._check_for_innocent_context(
        "U-1->U-2",
        gap_start,
        fake,
        case_id="case-a",
    )

    assert is_benign is False
    assert explanation is None


def test_auto_ingest_keeps_hard_invalid_records(
    tmp_path,
    monkeypatch,
) -> None:
    dataset = [
        {
            "timestamp": "2026-07-29T09:00:00Z",
            "sender_id": "U-1",
            "receiver_id": "U-2",
            "content": "Missing message id.",
            "platform": "whatsapp",
        },
        "not an object",
        {
            "message_id": "MSG-1",
            "timestamp": "2026-07-29T09:05:00Z",
            "sender_id": "U-1",
            "receiver_id": "U-2",
            "content": "Valid record.",
            "platform": "whatsapp",
        },
    ]
    dataset_path = tmp_path / "invalid_records.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    loaded_edges: list[dict[str, Any]] = []
    embedded_messages: list[dict[str, Any]] = []
    monkeypatch.setattr(
        pipeline_api,
        "_CASE_TO_FILE",
        {"case-invalid": dataset_path.name},
    )
    monkeypatch.setattr(pipeline_api, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(
        pipeline_api,
        "batch_load_graph",
        lambda edges, case_id="default": loaded_edges.extend(edges),
    )
    monkeypatch.setattr(
        pipeline_api,
        "embed_messages",
        lambda messages, case_id="default": embedded_messages.extend(messages),
    )

    state = pipeline_api._auto_ingest("case-invalid")

    assert len(state["raw_messages"]) == 3
    assert state["quarantined_ids"] == [
        "<missing-message-id-1>",
        "<missing-message-id-2>",
    ]
    assert state["raw_messages"][0]["is_quarantined"] is True
    assert state["raw_messages"][1]["is_quarantined"] is True
    assert loaded_edges == [
        {
            "sender_id": "U-1",
            "receiver_id": "U-2",
            "message_id": "MSG-1",
            "platform": "whatsapp",
            "timestamp_iso": "2026-07-29T09:05:00+00:00",
        }
    ]
    assert len(embedded_messages) == 3
    assert all(msg["case_id"] == "case-invalid" for msg in embedded_messages)


def test_aggregate_message_edges_keeps_counts_without_null_timestamp_lists() -> None:
    aggregated = _aggregate_message_edges(
        [
            {
                "sender_id": "U-1",
                "receiver_id": "U-2",
                "message_id": "MISSING-TS",
                "timestamp_iso": None,
                "platform": "whatsapp",
            },
            {
                "sender_id": "U-1",
                "receiver_id": "U-2",
                "message_id": "MSG-1",
                "timestamp_iso": "2026-07-29T09:00:00+00:00",
                "platform": "whatsapp",
            },
        ]
    )

    assert len(aggregated) == 1
    assert aggregated[0]["message_count"] == 2
    assert aggregated[0]["message_ids"] == ["MISSING-TS", "MSG-1"]
    assert aggregated[0]["timestamped_message_ids"] == ["MSG-1"]
    assert aggregated[0]["timestamps"] == ["2026-07-29T09:00:00+00:00"]


def test_neo4j_driver_uses_bounded_connection_timeouts(monkeypatch) -> None:
    """A remote graph outage must not consume Render's request budget."""
    calls: list[dict[str, Any]] = []

    def fake_driver(uri: str, **kwargs: Any) -> object:
        assert uri == "neo4j+s://example.databases.neo4j.io"
        calls.append(kwargs)
        return object()

    monkeypatch.setenv("NEO4J_URI", "neo4j+s://example.databases.neo4j.io")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "test-password")
    monkeypatch.setattr(graph_db.GraphDatabase, "driver", fake_driver)
    monkeypatch.setattr(graph_db, "_driver", None)
    monkeypatch.setattr(graph_db, "_unavailable_until", 0.0)

    graph_db.get_driver()

    assert calls == [
        {
            "auth": ("neo4j", "test-password"),
            "connection_timeout": 3.0,
            "connection_acquisition_timeout": 3.0,
            "max_transaction_retry_time": 0.0,
        }
    ]
