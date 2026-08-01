"""SENTINEL — dashboard data endpoints."""

from __future__ import annotations

from collections import Counter

from fastapi import APIRouter, HTTPException

from app.agents.module_12_risk import _compute_score
from app.api.pipeline import get_case_state

router = APIRouter(prefix="/api/v1", tags=["dashboard"])


@router.get("/dashboard/{case_id}")
async def get_dashboard(case_id: str) -> dict:
    """Return all data needed to render the full dashboard."""
    state = get_case_state(case_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found. Run the pipeline first.")

    risk_scores = state.get("risk_scores", {})
    gap_alerts = state.get("gap_alerts", [])
    centrality = state.get("centrality_summary", [])
    roles = state.get("roles", {})
    theories = state.get("theories", [])
    simulation = state.get("simulation", {})
    coercive = state.get("coercive_matches", [])
    raw_messages = state.get("raw_messages", [])
    quarantined_count = sum(1 for message in raw_messages if message.get("is_quarantined"))
    total_records = len(raw_messages)

    # Priority Board — sorted by risk score desc
    priority_board = sorted(risk_scores.values(), key=lambda x: x["score"], reverse=True)

    # ── Graph nodes for D3 ─────────────────────────────────────────────────
    nodes = [
        {
            "id": pid,
            "score": data["score"],
            "role": data["role_tag"],
            "pagerank": data["pagerank"],
            "betweenness": data["betweenness"],
        }
        for pid, data in risk_scores.items()
    ]

    # ── Active gap-alert pairs (same source for map badges and stat strip) ──
    active_gap_pairs = {}
    for alert in gap_alerts:
        pair_key = alert.get("pair_key", "")
        if (
            not alert.get("flagged_as_anomaly")
            or alert.get("suppressed")
            or "->" not in pair_key
        ):
            continue

        source, target = pair_key.split("->", 1)
        pair = tuple(sorted((source, target)))
        gap_info = active_gap_pairs.setdefault(
            pair,
            {"gap_alert_count": 0, "gap_hours": 0.0},
        )
        gap_info["gap_alert_count"] += 1
        gap_info["gap_hours"] = max(gap_info["gap_hours"], alert.get("gap_hours", 0.0))

    # ── Aggregated communication edges (one per unordered pair, weighted) ──
    pair_counts: Counter = Counter()
    for msg in raw_messages:
        if msg.get("is_quarantined"):
            continue
        s = msg.get("sender_id")
        r = msg.get("receiver_id")
        if s and r:
            pair_counts[tuple(sorted((s, r)))] += 1

    comm_edges = [
        {
            "source": pair[0],
            "target": pair[1],
            "type": "MESSAGED",
            "weight": count,
            "message_count": count,
            "has_gap_alert": pair in active_gap_pairs,
            "gap_alert_count": active_gap_pairs.get(pair, {}).get("gap_alert_count", 0),
            "gap_hours": active_gap_pairs.get(pair, {}).get("gap_hours"),
        }
        for pair, count in pair_counts.items()
    ]

    # ── Gap-alert edges (aggregated to the same pair semantics as comm_edges) ─
    gap_edges = [
        {
            "source": pair[0],
            "target": pair[1],
            "type": "TIMELINE_GAP",
            "gap_alert_count": info["gap_alert_count"],
            "gap_hours": info["gap_hours"],
        }
        for pair, info in active_gap_pairs.items()
    ]

    # Alert list ranked by score-delta — includes suppressed alerts so frontend
    # can show them as "cleared" cards instead of silently hiding them
    alert_list = []
    for alert in gap_alerts:
        if not alert.get("flagged_as_anomaly"):
            continue

        person_id = alert.get("pair_key", "").split("->", 1)[0]
        person = risk_scores.get(person_id, {})
        current_score = float(person.get("score", 0))
        resolved_indicators = [
            indicator
            for indicator in person.get("indicators", [])
            if indicator != "TIMELINE_GAP"
        ]
        if_resolved_score, _ = _compute_score(
            resolved_indicators,
            state.get("session_weights", {}).get(person_id),
        )
        resolved_domains = {item.split("_")[0] for item in resolved_indicators}
        resolved_domains.update(
            domain
            for domain in state.get("domain_scores", {}).get(person_id, [])
            if domain != "TIMELINE_GAP"
        )
        if (
            if_resolved_score > state.get("multi_factor_lock_threshold", 40)
            and len(resolved_domains) < 3
        ):
            if_resolved_score = state.get("multi_factor_lock_threshold", 40)

        alert_list.append(
            {
                **alert,
                "current_score": current_score,
                "if_resolved_score": if_resolved_score,
                "score_delta": round(current_score - if_resolved_score, 1),
            }
        )
    # Active alerts first (sorted by score-delta desc), then suppressed alerts at the end
    alert_list.sort(key=lambda x: (x.get("suppressed", False), -x.get("score_delta", 0)))

    return {
        "case_id": case_id,
        "priority_board": priority_board,
        "graph": {
            "nodes": nodes,
            "comm_edges": comm_edges,
            "gap_edges": gap_edges,
            "gap_badge_count": len(gap_edges),
        },
        "alert_list": alert_list,
        "theories": theories,
        "simulation": simulation,
        "coercive_matches_count": len(coercive),
        "suppressed_alerts_count": len([a for a in gap_alerts if a.get("suppressed")]),
        "data_quality": {
            "total_records": total_records,
            "quarantined_records": quarantined_count,
            "quarantine_rate": round(
                (quarantined_count / total_records) * 100, 1
            ) if total_records else 0.0,
        },
    }
