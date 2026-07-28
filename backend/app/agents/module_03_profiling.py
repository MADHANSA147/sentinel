"""
Module 03 — Subject Profiling (LangGraph node).
Aggregates per-Person: devices/platforms seen, alias list (from Module 02),
and centrality scores (from Module 06) into a single profile object
written as Neo4j node properties.
"""

from __future__ import annotations

from typing import Any

from app.services.graph_db import run_query, write_person_properties


def subject_profiling(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: aggregate profile properties onto each Person node.

    Reads:
        state['canonical_map']   (from Module 02)
        state['pagerank']        (from Module 06)
        state['betweenness']     (from Module 06)

    State outputs:
        profiles: dict[person_id -> profile dict]
    """
    canonical_map: dict = state.get("canonical_map", {})
    pagerank: dict = state.get("pagerank", {})
    betweenness: dict = state.get("betweenness", {})

    # Collect platform/device info from MESSAGED edges
    edge_records = run_query(
        """
        MATCH (s:Person)-[r:MESSAGED]->(t:Person)
        RETURN s.id AS person_id, collect(DISTINCT r.platform) AS platforms,
               count(r) AS sent_count
        UNION
        MATCH (s:Person)-[r:MESSAGED]->(t:Person)
        RETURN t.id AS person_id, collect(DISTINCT r.platform) AS platforms,
               0 AS sent_count
        """
    )

    # Aggregate per person
    profile_data: dict[str, dict] = {}
    for rec in edge_records:
        pid = rec["person_id"]
        if pid not in profile_data:
            profile_data[pid] = {"platforms": set(), "sent_count": 0}
        profile_data[pid]["platforms"].update(rec.get("platforms") or [])
        profile_data[pid]["sent_count"] += rec.get("sent_count", 0)

    profiles: dict[str, dict] = {}
    updates = []

    for pid, data in profile_data.items():
        profile = {
            "id": pid,
            "canonical_id": canonical_map.get(pid, pid),
            "platforms": list(data["platforms"]),
            "sent_count": data["sent_count"],
            "pagerank": round(pagerank.get(pid, 0.0), 6),
            "betweenness": round(betweenness.get(pid, 0.0), 6),
        }
        profiles[pid] = profile
        updates.append({
            "id": pid,
            "platforms": list(data["platforms"]),
            "sent_count": data["sent_count"],
        })

    if updates:
        write_person_properties(updates)

    state["profiles"] = profiles
    return state
