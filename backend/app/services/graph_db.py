"""
SENTINEL — Neo4j driver connection module.
Reads credentials from environment variables: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD.
"""

from __future__ import annotations

import os
from typing import Any

from neo4j import GraphDatabase, Driver

_driver: Driver | None = None


def get_driver() -> Driver:
    """Return a singleton Neo4j driver; create one if not yet initialised."""
    global _driver
    if _driver is None:
        uri = os.environ["NEO4J_URI"]
        user = os.environ["NEO4J_USER"]
        password = os.environ["NEO4J_PASSWORD"]
        _driver = GraphDatabase.driver(uri, auth=(user, password))
    return _driver


def close_driver() -> None:
    """Explicitly close the driver (call on app shutdown)."""
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


def batch_load_graph(edges: list[dict[str, Any]]) -> None:
    """
    Load Person nodes and MESSAGED edges into Neo4j via a single UNWIND batch.

    Args:
        edges: List of dicts with keys sender_id, receiver_id, platform,
               message_id, and timestamp_iso.
    """
    driver = get_driver()

    cypher = """
    UNWIND $edges AS edge
    MERGE (s:Person {id: edge.sender_id})
    MERGE (r:Person {id: edge.receiver_id})
    MERGE (s)-[rel:MESSAGED {message_id: edge.message_id}]->(r)
    SET rel.platform   = edge.platform,
        rel.timestamp  = edge.timestamp_iso
    """

    with driver.session() as session:
        session.run(cypher, edges=edges)


def write_person_properties(updates: list[dict[str, Any]]) -> None:
    """
    Bulk-update properties on existing Person nodes.

    Args:
        updates: List of dicts with at least {'id': <person_id>, ...extra props}.
    """
    driver = get_driver()

    cypher = """
    UNWIND $updates AS u
    MATCH (p:Person {id: u.id})
    SET p += u
    """
    with driver.session() as session:
        session.run(cypher, updates=updates)


def run_query(cypher: str, params: dict[str, Any] | None = None) -> list[dict]:
    """
    General-purpose read helper returning a list of record dicts.
    """
    driver = get_driver()
    params = params or {}
    with driver.session() as session:
        result = session.run(cypher, **params)
        return [dict(r) for r in result]
