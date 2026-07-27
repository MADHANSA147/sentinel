"""SENTINEL — HITL (Human-In-The-Loop) endpoint wired to Module 13."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.api.pipeline import get_case_state, _case_states
from app.agents.module_13_feedback import reject_flag, approve_flag, get_appeal_log

router = APIRouter(prefix="/api/v1", tags=["hitl"])


class FlagDecision(BaseModel):
    person_id: str
    indicator: str
    action: str  # "approve" | "reject"


@router.post("/hitl/{case_id}")
async def submit_decision(case_id: str, decision: FlagDecision) -> dict:
    """
    Submit an investigator decision (approve/reject) for a flagged indicator.
    Rejection triggers immediate score recompute via Module 13.
    """
    state = get_case_state(case_id)
    if not state:
        raise HTTPException(status_code=404, detail="Case not found.")

    if decision.action == "reject":
        updated_state = reject_flag(state, decision.person_id, decision.indicator)
    elif decision.action == "approve":
        updated_state = approve_flag(state, decision.person_id, decision.indicator)
    else:
        raise HTTPException(status_code=400, detail="action must be 'approve' or 'reject'")

    _case_states[case_id] = updated_state

    new_score = updated_state.get("risk_scores", {}).get(decision.person_id, {}).get("score")

    return {
        "case_id": case_id,
        "person_id": decision.person_id,
        "action": decision.action,
        "new_score": new_score,
        "message": f"Flag {decision.action}d. Score updated to {new_score}.",
    }


@router.get("/hitl/{case_id}/appeals")
async def get_appeals(case_id: str) -> dict:
    """Return the current appeal log for a case session."""
    return {"case_id": case_id, "appeals": get_appeal_log()}
