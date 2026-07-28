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

from typing import Any

import networkx as nx

from app.services.graph_db import run_query, write_person_properties


def network_mapping(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: build directed graph → compute metrics → write to Neo4j.

    State outputs:
        pagerank: dict[person_id -> float]
        betweenness: dict[person_id -> float]
        centrality_summary: list of dicts for easy display
    """
    # ── Fetch all MESSAGED edges from Neo4j ────────────────────────────────
    records = run_query(
        """
        MATCH (s:Person)-[r:MESSAGED]->(t:Person)
        RETURN s.id AS sender, t.id AS receiver
        """
    )

    if not records:
        state["pagerank"] = {}
        state["betweenness"] = {}
        state["centrality_summary"] = []
        return state

    # ── Build directed graph ──────────────────────────────────────────────
    G = nx.DiGraph()
    for rec in records:
        if rec["sender"] and rec["receiver"]:
            # Each duplicate edge (multiple messages same pair) adds weight
            if G.has_edge(rec["sender"], rec["receiver"]):
                G[rec["sender"]][rec["receiver"]]["weight"] += 1
            else:
                G.add_edge(rec["sender"], rec["receiver"], weight=1)

    # ── Compute metrics ───────────────────────────────────────────────────
    pagerank: dict[str, float] = nx.pagerank(G, alpha=0.85, weight="weight")
    # Betweenness on undirected projection catches bridge role better
    G_undirected = G.to_undirected()
    betweenness: dict[str, float] = nx.betweenness_centrality(
        G_undirected, normalized=True, weight="weight"
    )

    # ── Write scores back to Neo4j Person nodes ───────────────────────────
    updates = [
        {
            "id": node,
            "pagerank": round(pagerank.get(node, 0.0), 6),
            "betweenness": round(betweenness.get(node, 0.0), 6),
        }
        for node in set(list(pagerank) + list(betweenness))
    ]
    write_person_properties(updates)

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
