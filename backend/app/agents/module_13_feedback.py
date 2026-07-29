"""
Module 13 — Feedback Loop (LangGraph node).
When an investigator rejects a flag:
  1. Reduces that indicator's weight for the current case session only
     (NOT persistent model retraining — session-scoped as designed)
  2. Recomputes the affected Person's risk score immediately
  3. Logs the appeal to a review queue

Session-scoped by design: live model retraining during an active case
investigation is not reliably buildable and the guarantee cannot be made.
Session reweighting has the same practical effect on the active case.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

_WEIGHT_REDUCTION_FACTOR = 0.5  # Rejected indicator weight halved for session

# In-memory review queue (persisted via ledger in production)
_appeal_log: list[dict] = []


def reject_flag(
    state: dict[str, Any],
    person_id: str,
    indicator: str,
) -> dict[str, Any]:
    """
    Called when an investigator rejects a flag via HITL.
    Updates session weights and recomputes the affected person's score.
    """
    # ── Reduce indicator weight for this session ───────────────────────────
    session_weights = state.setdefault("session_weights", {})
    person_session = session_weights.setdefault(person_id, {})

    from app.agents.module_12_risk import INDICATOR_WEIGHTS
    current_weight = person_session.get(indicator, INDICATOR_WEIGHTS.get(indicator, 1.0))
    person_session[indicator] = round(current_weight * _WEIGHT_REDUCTION_FACTOR, 4)

    # ── Log appeal ─────────────────────────────────────────────────────────
    appeal = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "person_id": person_id,
        "indicator": indicator,
        "original_weight": current_weight,
        "new_weight": person_session[indicator],
        "action": "REJECTED",
    }
    _appeal_log.append(appeal)

    # ── Recompute risk score immediately ────────────────────────────────────
    risk_scores = state.get("risk_scores", {})
    if person_id in risk_scores:
        from app.agents.module_12_risk import _compute_score, INDICATOR_WEIGHTS as IW
        indicators = risk_scores[person_id].get("indicators", [])
        new_score, new_tree = _compute_score(indicators, person_session)

        # Enforce multi-factor lock
        domain_scores = state.get("domain_scores", {})
        all_domains = set(domain_scores.get(person_id, []))
        indicator_domains = set(ind.split("_")[0] for ind in indicators)
        all_domains |= indicator_domains
        lock = state.get("multi_factor_lock_threshold", 40)
        if new_score > lock and len(all_domains) < 3:
            new_score = lock

        risk_scores[person_id]["score"] = new_score
        risk_scores[person_id]["justification_tree"] = new_tree
        state["risk_scores"] = risk_scores

    state["session_weights"] = session_weights
    return state


def approve_flag(state: dict[str, Any], person_id: str, indicator: str) -> dict[str, Any]:
    """Log an approved flag (no score change — investigator confirms the finding)."""
    appeal = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "person_id": person_id,
        "indicator": indicator,
        "action": "APPROVED",
    }
    _appeal_log.append(appeal)
    return state


def get_appeal_log() -> list[dict]:
    """Return the current session appeal log."""
    return list(_appeal_log)


def feedback_loop(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: pass-through — actual reweighting triggered by API endpoint."""
    state.setdefault("session_weights", {})
    return state
