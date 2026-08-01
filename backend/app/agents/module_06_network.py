"""
Module 06 — Network Mapping (LangGraph node).
Uses networkx (not Neo4j GDS — free tier likely lacks it) to compute
PageRank and Betweenness Centrality over the Person-to-Person MESSAGED graph.

Why both metrics:
  - PageRank: catches structurally central nodes regardless of raw message count
  - Betweenness: catches brokers bridging otherwise-disconnected clusters
    (these can have LOW total messages but HIGH structural importance)

Expected outcome against network_test_45_messages.json:
  U-301 and U-305 highest PageRank  |  U-304 highest Betweenness
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import networkx as nx

from app.services.graph_db import run_query_for_case, write_person_properties


def _edge_records_from_raw_messages(
    raw_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate directed sender->receiver counts from raw pipeline state."""
    counts: Counter[tuple[str, str]] = Counter()
    for msg in raw_messages:
        if msg.get("is_quarantined"):
            continue
        sender = msg.get("sender_id")
        receiver = msg.get("receiver_id")
        if sender and receiver:
            counts[(str(sender), str(receiver))] += 1

    return [
        {"sender": sender, "receiver": receiver, "weight": weight}
        for (sender, receiver), weight in counts.items()
    ]


def network_mapping(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: build directed graph → compute metrics → write to Neo4j.

    State outputs:
        pagerank: dict[person_id -> float]
        betweenness: dict[person_id -> float]
        centrality_summary: list of dicts for easy display
    """
    case_id: str = state.get("case_id", "default")
    raw_messages: list[dict[str, Any]] = state.get("raw_messages", [])

    if raw_messages:
        records = _edge_records_from_raw_messages(raw_messages)
    else:
        try:
            # ── Fetch MESSAGED edges scoped to this case ─────────────────────
            records = run_query_for_case(
                case_id,
                """
                MATCH (s:Person {case_id: $case_id})-[r:MESSAGED]->(t:Person {case_id: $case_id})
                RETURN s.id AS sender,
                       t.id AS receiver,
                       coalesce(r.message_count, 1) AS weight
                """,
            )
        except Exception as exc:
            print(f"[WARN] Network Mapping graph query failed: {exc}")
            records = []

    if not records:
        state["pagerank"] = {}
        state["betweenness"] = {}
        state["centrality_summary"] = []
        return state

    # ── Build directed graph ──────────────────────────────────────────────
    G = nx.DiGraph()
    for rec in records:
        if rec["sender"] and rec["receiver"]:
            weight = int(rec.get("weight") or 1)
            if G.has_edge(rec["sender"], rec["receiver"]):
                G[rec["sender"]][rec["receiver"]]["weight"] += weight
            else:
                G.add_edge(rec["sender"], rec["receiver"], weight=weight)

    # ── Compute metrics ───────────────────────────────────────────────────
    if G.number_of_nodes() == 0:
        state["pagerank"] = {}
        state["betweenness"] = {}
        state["centrality_summary"] = []
        return state

    pagerank: dict[str, float] = nx.pagerank(G, alpha=0.85, weight="weight")
    # Betweenness on undirected projection catches bridge role better
    G_undirected = G.to_undirected()
    betweenness: dict[str, float] = nx.betweenness_centrality(
        G_undirected, normalized=True, weight="weight"
    )

    # ── Write scores back to Neo4j Person nodes ───────────────────────────────
    updates = [
        {
            "id": node,
            "pagerank": round(pagerank.get(node, 0.0), 6),
            "betweenness": round(betweenness.get(node, 0.0), 6),
        }
        for node in set(list(pagerank) + list(betweenness))
    ]
    try:
        write_person_properties(updates, case_id=case_id)
    except Exception as exc:
        print(f"[WARN] Network Mapping graph write failed: {exc}")

    # ── Build summary for state ───────────────────────────────────────────
    summary = sorted(
        [
            {
                "person_id": node,
                "pagerank": round(pagerank.get(node, 0.0), 6),
                "betweenness": round(betweenness.get(node, 0.0), 6),
            }
            for node in G.nodes()
        ],
        key=lambda x: x["pagerank"],
        reverse=True,
    )

    state["pagerank"] = pagerank
    state["betweenness"] = betweenness
    state["centrality_summary"] = summary
    return state
