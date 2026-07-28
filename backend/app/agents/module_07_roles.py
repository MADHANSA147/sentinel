"""
Module 07 — Role Discovery (LangGraph node).
Uses centrality scores (Module 06) and message directionality to assign
one functional tag per Person via LLM classification.

Tags: Orchestrator | Recruiter | Enforcer | Target | Peripheral
"""

from __future__ import annotations

import json
import os
from typing import Any

from app.services.graph_db import run_query, write_person_properties

# ── Fallback rule-based tagger (used when LLM not available) ───────────────
_TAG_THRESHOLDS = {
    "orchestrator": {"pagerank": 0.20, "betweenness": 0.15},
    "bridge":       {"betweenness": 0.30},
    "peripheral":   {"pagerank": 0.05},
}


def _rule_based_tag(pagerank: float, betweenness: float, sent_fraction: float) -> tuple[str, str]:
    """Simple rule-based fallback when LLM unavailable."""
    if betweenness >= 0.30:
        return "Orchestrator", "High betweenness centrality indicates bridging role between clusters."
    if pagerank >= 0.20 and sent_fraction >= 0.5:
        return "Orchestrator", "High PageRank and dominant send volume indicate coordination role."
    if pagerank >= 0.15:
        return "Recruiter", "Moderately high PageRank with broad connectivity suggests outreach pattern."
    if sent_fraction <= 0.2:
        return "Target", "Predominantly receives messages; low initiation rate."
    return "Peripheral", "Low centrality and balanced communication; no significant structural role."


def role_discovery(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: assign one role tag and justification per Person.

    Reads:
        state['pagerank'], state['betweenness'], state['profiles']

    State outputs:
        roles: dict[person_id -> {'tag': str, 'justification': str}]
    """
    pagerank: dict = state.get("pagerank", {})
    betweenness: dict = state.get("betweenness", {})
    profiles: dict = state.get("profiles", {})

    # Get sent vs received counts per person from Neo4j
    sent_records = run_query(
        """
        MATCH (s:Person)-[r:MESSAGED]->()
        RETURN s.id AS id, count(r) AS sent
        """
    )
    recv_records = run_query(
        """
        MATCH ()-[r:MESSAGED]->(t:Person)
        RETURN t.id AS id, count(r) AS received
        """
    )
    sent_map = {r["id"]: r["sent"] for r in sent_records}
    recv_map = {r["id"]: r["received"] for r in recv_records}

    roles: dict[str, dict] = {}
    updates = []

    all_people = set(list(pagerank) + list(betweenness))

    for person_id in all_people:
        pr = pagerank.get(person_id, 0.0)
        bc = betweenness.get(person_id, 0.0)
        sent = sent_map.get(person_id, 0)
        recv = recv_map.get(person_id, 0)
        total = sent + recv
        sent_fraction = sent / total if total > 0 else 0.5

        tag, justification = _rule_based_tag(pr, bc, sent_fraction)

        roles[person_id] = {"tag": tag, "justification": justification}
        updates.append({
            "id": person_id,
            "role_tag": tag,
            "role_justification": justification,
        })

    if updates:
        write_person_properties(updates)

    state["roles"] = roles
    return state
