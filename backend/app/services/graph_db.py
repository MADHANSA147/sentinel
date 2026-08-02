"""
SENTINEL — Neo4j driver connection module.
Reads credentials from environment variables: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD.
"""

from __future__ import annotations

import os
import time
from typing import Any

from neo4j import GraphDatabase, Driver

_driver: Driver | None = None
_CONNECTION_TIMEOUT_SECONDS = 3.0
_QUERY_TIMEOUT_SECONDS = 5.0
_RETRY_COOLDOWN_SECONDS = 30.0
_unavailable_until = 0.0


def _mark_unavailable() -> None:
    """Avoid repeating a failed remote graph connection during one pipeline run."""
    global _driver, _unavailable_until
    _unavailable_until = time.monotonic() + _RETRY_COOLDOWN_SECONDS
    if _driver is not None:
        try:
            _driver.close()
        except Exception:
            pass
    _driver = None


def _mark_available() -> None:
    """Clear the connection circuit breaker after a successful graph operation."""
    global _unavailable_until
    _unavailable_until = 0.0


def _aggregate_message_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group message records into one weighted directed relationship per pair."""
    grouped: dict[tuple[str, str], dict[str, Any]] = {}

    for edge in edges:
        sender = edge.get("sender_id")
        receiver = edge.get("receiver_id")
        if not sender or not receiver:
            continue

        key = (str(sender), str(receiver))
        group = grouped.setdefault(
            key,
            {
                "sender_id": str(sender),
                "receiver_id": str(receiver),
                "message_ids": [],
                "timestamps": [],
                "platforms": set(),
            },
        )
        group["message_ids"].append(str(edge.get("message_id", "")))
        group["timestamps"].append(edge.get("timestamp_iso"))
        if edge.get("platform"):
            group["platforms"].add(str(edge["platform"]))

    aggregated: list[dict[str, Any]] = []
    for group in grouped.values():
        rows = sorted(
            zip(group["message_ids"], group["timestamps"]),
            key=lambda row: row[1] or "",
        )
        message_ids = [row[0] for row in rows]
        timestamped_rows = [row for row in rows if row[1]]
        timestamped_message_ids = [row[0] for row in timestamped_rows]
        timestamps = [row[1] for row in timestamped_rows]
        platforms = sorted(group["platforms"])

        aggregated.append(
            {
                "sender_id": group["sender_id"],
                "receiver_id": group["receiver_id"],
                "message_ids": message_ids,
                "timestamped_message_ids": timestamped_message_ids,
                "timestamps": timestamps,
                "platforms": platforms,
                "platform": platforms[0] if platforms else "unknown",
                "message_count": len(message_ids),
                "first_timestamp": timestamps[0] if timestamps else None,
                "last_timestamp": timestamps[-1] if timestamps else None,
            }
        )

    return aggregated


def get_driver() -> Driver:
    """Return a singleton Neo4j driver; create one if not yet initialised."""
    global _driver
    if time.monotonic() < _unavailable_until:
        raise RuntimeError("Neo4j is temporarily unavailable; retry cooldown is active.")
    if _driver is None:
        try:
            uri = os.environ["NEO4J_URI"]
            user = os.environ["NEO4J_USER"]
            password = os.environ["NEO4J_PASSWORD"]
            _driver = GraphDatabase.driver(
                uri,
                auth=(user, password),
                connection_timeout=_CONNECTION_TIMEOUT_SECONDS,
                connection_acquisition_timeout=_CONNECTION_TIMEOUT_SECONDS,
                max_transaction_retry_time=0.0,
            )
        except Exception:
            _mark_unavailable()
            raise
    return _driver


def close_driver() -> None:
    """Explicitly close the driver (call on app shutdown)."""
    global _driver, _unavailable_until
    if _driver is not None:
        _driver.close()
        _driver = None
    _unavailable_until = 0.0


def batch_load_graph(edges: list[dict[str, Any]], case_id: str = "default") -> None:
    """
    Load Person nodes and MESSAGED edges into Neo4j via a single UNWIND batch.
    Each node and edge is tagged with case_id so multiple datasets can coexist.

    Args:
        edges:   List of dicts with keys sender_id, receiver_id, platform,
                 message_id, and timestamp_iso.
        case_id: Partition key — one per demo dataset (e.g. 'case-dataset1').
    """
    driver = get_driver()

    tagged = [{**e, "case_id": case_id} for e in _aggregate_message_edges(edges)]

    cypher = """
    UNWIND $edges AS edge
    MERGE (s:Person {id: edge.sender_id, case_id: edge.case_id})
    MERGE (r:Person {id: edge.receiver_id, case_id: edge.case_id})
    MERGE (s)-[rel:MESSAGED]->(r)
    SET rel.platform        = edge.platform,
        rel.platforms       = edge.platforms,
        rel.message_count   = edge.message_count,
        rel.message_ids     = edge.message_ids,
        rel.timestamped_message_ids = edge.timestamped_message_ids,
        rel.timestamps      = edge.timestamps,
        rel.first_timestamp = edge.first_timestamp,
        rel.last_timestamp  = edge.last_timestamp,
        rel.case_id         = edge.case_id
    REMOVE rel.message_id, rel.timestamp
    """

    try:
        with driver.session() as session:
            session.run(
                "MATCH (p:Person {case_id: $case_id}) DETACH DELETE p",
                case_id=case_id,
                timeout=_QUERY_TIMEOUT_SECONDS,
            ).consume()
            session.run(cypher, edges=tagged, timeout=_QUERY_TIMEOUT_SECONDS).consume()
    except Exception:
        _mark_unavailable()
        raise
    _mark_available()


def write_person_properties(updates: list[dict[str, Any]], case_id: str = "default") -> None:
    """
    Bulk-update properties on existing Person nodes, scoped to a case_id.

    Args:
        updates: List of dicts with at least {'id': <person_id>, ...extra props}.
        case_id: Only update Person nodes belonging to this case partition.
    """
    driver = get_driver()

    cypher = """
    UNWIND $updates AS u
    MATCH (p:Person {id: u.id, case_id: $case_id})
    SET p += u
    """
    run_write(cypher, {"updates": updates, "case_id": case_id})


def run_write(cypher: str, params: dict[str, Any] | None = None) -> None:
    """Execute a bounded Cypher write and open the failure circuit on error."""
    driver = get_driver()
    try:
        with driver.session() as session:
            session.run(cypher, **(params or {}), timeout=_QUERY_TIMEOUT_SECONDS).consume()
    except Exception:
        _mark_unavailable()
        raise
    _mark_available()


def run_query(cypher: str, params: dict[str, Any] | None = None) -> list[dict]:
    """
    General-purpose read helper returning a list of record dicts.
    """
    driver = get_driver()
    params = params or {}
    try:
        with driver.session() as session:
            result = session.run(cypher, **params, timeout=_QUERY_TIMEOUT_SECONDS)
            records = [dict(r) for r in result]
    except Exception:
        _mark_unavailable()
        raise
    _mark_available()
    return records


def run_query_for_case(case_id: str, cypher: str, params: dict[str, Any] | None = None) -> list[dict]:
    """
    Convenience read helper that automatically adds case_id to params.
    Use $case_id inside your Cypher to filter by partition.
    """
    driver = get_driver()
    params = {**(params or {}), "case_id": case_id}
    try:
        with driver.session() as session:
            result = session.run(cypher, **params, timeout=_QUERY_TIMEOUT_SECONDS)
            records = [dict(r) for r in result]
    except Exception:
        _mark_unavailable()
        raise
    _mark_available()
    return records
