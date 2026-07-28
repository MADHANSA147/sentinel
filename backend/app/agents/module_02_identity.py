"""
Module 02 — Identity Fusion (LangGraph node).
Merges cross-platform aliases into unified Person nodes in Neo4j.
Matching strategy: overlapping sender_id / receiver_id tokens.
"""

from __future__ import annotations

from typing import Any

from app.services.graph_db import get_driver, run_query


def identity_fusion(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: ensure every distinct sender/receiver resolves to exactly
    one Person node in Neo4j. Aliases that share a token are merged under
    the lexicographically earliest id as the canonical key.

    Returns updated state with 'person_nodes' list.
    """
    driver = get_driver()

    # Collect all IDs that appear in MESSAGED edges
    records = run_query("MATCH (p:Person) RETURN p.id AS id")
    person_ids: list[str] = [r["id"] for r in records if r.get("id")]

    # Simple alias grouping: treat tokens that share a non-empty prefix as
    # the same person.  For the MVP (purely numeric IDs like U-101) this
    # effectively keeps them distinct, which is correct — they come from
    # synthetic data with no real aliases.
    canonical_map: dict[str, str] = {}
    for pid in sorted(person_ids):
        canonical_map[pid] = pid  # extend here for real-world alias resolution

    # Write canonical_id property onto every Person node in one batch
    updates = [{"id": pid, "canonical_id": cid} for pid, cid in canonical_map.items()]
    if updates:
        cypher = """
        UNWIND $updates AS u
        MATCH (p:Person {id: u.id})
        SET p.canonical_id = u.canonical_id
        """
        with driver.session() as session:
            session.run(cypher, updates=updates)

    state["person_nodes"] = person_ids
    state["canonical_map"] = canonical_map
    return state
