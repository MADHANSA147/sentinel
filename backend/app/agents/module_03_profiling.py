"""
Module 03 — Subject Profiling (LangGraph node).
Aggregates per-Person: devices/platforms seen, alias list (from Module 02),
and centrality scores (from Module 06) into a single profile object
written as Neo4j node properties.
"""

from __future__ import annotations

from typing import Any

from app.services.graph_db import run_query_for_case, write_person_properties


def _profile_data_from_raw_messages(
    raw_messages: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Aggregate platform and sent-count features from raw pipeline state."""
    profile_data: dict[str, dict[str, Any]] = {}
    for msg in raw_messages:
        sender = msg.get("sender_id")
        receiver = msg.get("receiver_id")
        platform = msg.get("platform")

        for person_id in (sender, receiver):
            if not person_id:
                continue
            profile = profile_data.setdefault(
                str(person_id),
                {"platforms": set(), "sent_count": 0},
            )
            if platform:
                profile["platforms"].add(str(platform))

        if sender:
            profile_data[str(sender)]["sent_count"] += 1

    return profile_data


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
    case_id: str = state.get("case_id", "default")
    raw_messages: list[dict[str, Any]] = state.get("raw_messages", [])

    if raw_messages:
        profile_data = _profile_data_from_raw_messages(raw_messages)
    else:
        try:
            # Collect platform/device info from MESSAGED edges
            edge_records = run_query_for_case(
                case_id,
                """
                MATCH (s:Person {case_id: $case_id})-[r:MESSAGED]->(t:Person {case_id: $case_id})
                RETURN s.id AS person_id, collect(DISTINCT r.platform) AS platforms,
                       collect(r.platforms) AS platform_lists,
                       sum(coalesce(r.message_count, 1)) AS sent_count
                UNION
                MATCH (s:Person {case_id: $case_id})-[r:MESSAGED]->(t:Person {case_id: $case_id})
                RETURN t.id AS person_id, collect(DISTINCT r.platform) AS platforms,
                       collect(r.platforms) AS platform_lists,
                       0 AS sent_count
                """,
            )
        except Exception as exc:
            print(f"[WARN] Subject Profiling graph query failed: {exc}")
            edge_records = []

        # Aggregate per person
        profile_data: dict[str, dict] = {}
        for rec in edge_records:
            pid = rec["person_id"]
            if pid not in profile_data:
                profile_data[pid] = {"platforms": set(), "sent_count": 0}
            profile_data[pid]["platforms"].update(
                p for p in rec.get("platforms", []) if p
            )
            for platform_list in rec.get("platform_lists", []) or []:
                if not platform_list:
                    continue
                profile_data[pid]["platforms"].update(p for p in platform_list if p)
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
        try:
            write_person_properties(updates, case_id=case_id)
        except Exception as exc:
            print(f"[WARN] Subject Profiling graph write failed: {exc}")

    state["profiles"] = profiles
    return state
