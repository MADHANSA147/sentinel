"""SENTINEL — pipeline trigger endpoint."""

from __future__ import annotations

from fastapi import APIRouter
from app.agents.orchestrator import run_pipeline
from app.services.ledger import log_step, get_ledger

router = APIRouter(prefix="/api/v1", tags=["pipeline"])

# In-memory case state (keyed by case_id)
_case_states: dict[str, dict] = {}


@router.post("/run/{case_id}")
async def run_case(case_id: str) -> dict:
    """Trigger the full agent pipeline for a given case_id."""
    existing_state = _case_states.get(case_id, {})
    state = run_pipeline({"case_id": case_id, **existing_state})
    _case_states[case_id] = state

    log_step(
        case_id=case_id,
        module_name="ORCHESTRATOR",
        input_summary=f"Case {case_id} pipeline triggered",
        output_summary=f"Pipeline completed. Persons: {len(state.get('risk_scores', {}))}",
    )

    return {
        "case_id": case_id,
        "status": "completed",
        "persons_analysed": len(state.get("risk_scores", {})),
        "gap_alerts": len([a for a in state.get("gap_alerts", []) if a.get("flagged_as_anomaly")]),
        "theories": state.get("theories", []),
    }


def get_case_state(case_id: str) -> dict:
    return _case_states.get(case_id, {})
