"""
Module 07 — Role Discovery (LangGraph node).
Uses centrality scores (Module 06) and message directionality to assign
one functional tag per Person.

Tags: Orchestrator | Bridge | Recruiter | Enforcer | Target | Peripheral
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from app.services.graph_db import run_query_for_case, write_person_properties

# ── Fallback rule-based tagger (used when LLM not available) ───────────────
_TAG_THRESHOLDS = {
    "hub_quantile": 0.70,
    "bridge_quantile": 0.70,
    "orchestrator": {"pagerank": 0.15},
    "bridge":       {"betweenness": 0.30},
    "peripheral":   {"pagerank": 0.05},
}


def _quantile_threshold(scores: dict[str, float], quantile: float) -> float:
    """Return the score at a quantile-like rank for small demo graphs."""
    values = sorted(scores.values())
    if not values:
        return 0.0
    index = min(len(values) - 1, int(len(values) * quantile))
    return float(values[index])


def _rule_based_tag(
    pagerank: float,
    betweenness: float,
    sent_fraction: float,
    *,
    is_pagerank_hub: bool,
    is_bridge_candidate: bool,
) -> tuple[str, str]:
    """Simple rule-based fallback when LLM unavailable."""
    if is_bridge_candidate:
        return "Bridge", "High betweenness with lower PageRank indicates a broker between clusters."
    if is_pagerank_hub and sent_fraction >= 0.35:
        return "Orchestrator", "Top PageRank and active send volume indicate a cluster hub."
    if pagerank >= 0.15:
        return "Recruiter", "Moderately high PageRank with broad connectivity suggests outreach pattern."
    if sent_fraction <= 0.2:
        return "Target", "Predominantly receives messages; low initiation rate."
    return "Peripheral", "Low centrality and balanced communication; no significant structural role."


def _direction_counts_from_raw_messages(
    raw_messages: list[dict[str, Any]],
) -> tuple[dict[str, int], dict[str, int]]:
    """Return sent and received message counts from raw pipeline state."""
    sent: Counter[str] = Counter()
    received: Counter[str] = Counter()
    for msg in raw_messages:
        if msg.get("is_quarantined"):
            continue
        sender = msg.get("sender_id")
        receiver = msg.get("receiver_id")
        if sender:
            sent[str(sender)] += 1
        if receiver:
            received[str(receiver)] += 1
    return dict(sent), dict(received)


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
    case_id: str = state.get("case_id", "default")
    raw_messages: list[dict[str, Any]] = state.get("raw_messages", [])

    if raw_messages:
        sent_map, recv_map = _direction_counts_from_raw_messages(raw_messages)
    else:
        try:
            # Get sent vs received counts per person from Neo4j
            sent_records = run_query_for_case(
                case_id,
                """
                MATCH (s:Person {case_id: $case_id})-[r:MESSAGED]->()
                RETURN s.id AS id, sum(coalesce(r.message_count, 1)) AS sent
                """,
            )
            recv_records = run_query_for_case(
                case_id,
                """
                MATCH ()-[r:MESSAGED]->(t:Person {case_id: $case_id})
                RETURN t.id AS id, sum(coalesce(r.message_count, 1)) AS received
                """,
            )
            sent_map = {r["id"]: r["sent"] for r in sent_records}
            recv_map = {r["id"]: r["received"] for r in recv_records}
        except Exception as exc:
            print(f"[WARN] Role Discovery graph query failed: {exc}")
            sent_map = {}
            recv_map = {}

    roles: dict[str, dict] = {}
    updates = []

    all_people = set(list(pagerank) + list(betweenness) + list(profiles))
    pagerank_hub_threshold = _quantile_threshold(
        pagerank, _TAG_THRESHOLDS["hub_quantile"]
    )
    betweenness_bridge_threshold = _quantile_threshold(
        betweenness, _TAG_THRESHOLDS["bridge_quantile"]
    )

    for person_id in all_people:
        pr = pagerank.get(person_id, 0.0)
        bc = betweenness.get(person_id, 0.0)
        sent = sent_map.get(person_id, 0)
        recv = recv_map.get(person_id, 0)
        total = sent + recv
        sent_fraction = sent / total if total > 0 else 0.5

        is_pagerank_hub = (
            pr >= pagerank_hub_threshold
            and pr >= _TAG_THRESHOLDS["orchestrator"]["pagerank"]
        )
        is_bridge_candidate = (
            bc >= betweenness_bridge_threshold
            and bc >= _TAG_THRESHOLDS["bridge"]["betweenness"]
            and not is_pagerank_hub
        )

        tag, justification = _rule_based_tag(
            pr,
            bc,
            sent_fraction,
            is_pagerank_hub=is_pagerank_hub,
            is_bridge_candidate=is_bridge_candidate,
        )

        roles[person_id] = {"tag": tag, "justification": justification}
        updates.append({
            "id": person_id,
            "role_tag": tag,
            "role_justification": justification,
        })

    if updates:
        try:
            write_person_properties(updates, case_id=case_id)
        except Exception as exc:
            print(f"[WARN] Role Discovery graph write failed: {exc}")

    state["roles"] = roles
    return state
