"""SENTINEL — dashboard data endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

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
    coercive = state.get("coercive_matches", [])

    # Priority Board — sorted by risk score desc
    priority_board = sorted(risk_scores.values(), key=lambda x: x["score"], reverse=True)

    # Graph nodes and edges for D3
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

    # Gap-alert edges (always render regardless of top-50 cutoff)
    gap_edges = [
        {
            "source": a["pair_key"].split("->")[0],
            "target": a["pair_key"].split("->")[1] if "->" in a["pair_key"] else "",
            "type": "TIMELINE_GAP",
            "gap_hours": a["gap_hours"],
            "suppressed": a.get("suppressed", False),
        }
        for a in gap_alerts
        if a.get("flagged_as_anomaly") and "->" in a.get("pair_key", "")
    ]

    # Alert list ranked by raw score (score-delta would need baseline — simplified here)
    alert_list = [
        {
            **a,
            "score_delta": risk_scores.get(
                a["pair_key"].split("->")[0] if "->" in a["pair_key"] else "",
                {}
            ).get("score", 0),
        }
        for a in gap_alerts
        if a.get("flagged_as_anomaly") and not a.get("suppressed")
    ]
    alert_list.sort(key=lambda x: x.get("score_delta", 0), reverse=True)

    return {
        "case_id": case_id,
        "priority_board": priority_board,
        "graph": {"nodes": nodes, "gap_edges": gap_edges},
        "alert_list": alert_list,
        "theories": theories,
        "coercive_matches_count": len(coercive),
        "suppressed_alerts_count": len([a for a in gap_alerts if a.get("suppressed")]),
    }
