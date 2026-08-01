"""
Module 12 — Risk Score Engine ⭐ (LangGraph node).

Formula: Risk = sigmoid(sum(W_i * x_i)) * 100

Where W_i = log(P(x_i|Perpetrator) / P(x_i|NonPerpetrator))
      x_i = binary indicator (1 if present, 0 if not)

Algorithm: Log-odds / Weight-of-Evidence — same evidential math used in
forensic science broadly. Every point traces to a named piece of evidence.
A black-box would be more "accurate" but legally useless — interpretability
is the requirement, not accuracy.

Important caveats (acknowledged):
  - Assumes conditional independence of indicators (not strictly true).
  - Weights below are ILLUSTRATIVE — hand-set for demo, not empirically calibrated.
  - Module 11's multi-factor lock is the practical mitigation for score inflation.
"""

from __future__ import annotations

import math
from typing import Any

from app.services.graph_db import write_person_properties

# ── NCMEC / ISO 27037 taxonomy mapping ─────────────────────────────────────
TAXONOMY_MAP = {
    "COERCIVE_COMMUNICATION": {
        "ncmec": "Grooming Indicator GI-01: Secrecy/isolation language",
        "iso27037": "ISO/IEC 27037:2012 §8.3.1 — Communication pattern evidence",
    },
    "TIMELINE_GAP": {
        "ncmec": "Grooming Indicator GI-04: Communication blackout period",
        "iso27037": "ISO/IEC 27037:2012 §8.3.2 — Temporal anomaly in digital evidence",
    },
    "HIGH_PAGERANK": {
        "ncmec": "Network Indicator NI-01: Central coordination role",
        "iso27037": "ISO/IEC 27037:2012 §8.4.1 — Network topology analysis",
    },
    "HIGH_BETWEENNESS": {
        "ncmec": "Network Indicator NI-02: Bridge/broker structural role",
        "iso27037": "ISO/IEC 27037:2012 §8.4.2 — Information flow analysis",
    },
    "ROLE_ORCHESTRATOR": {
        "ncmec": "Behavioral Indicator BI-01: Command and coordination pattern",
        "iso27037": "ISO/IEC 27037:2012 §8.5.1 — Behavioral pattern analysis",
    },
}

# ── Weight table (W_i = log-odds) ─────────────────────────────────────────
# ILLUSTRATIVE WEIGHTS — proof of methodology, not empirical calibration.
# Comments show the conceptual reasoning for each weight direction.
INDICATOR_WEIGHTS: dict[str, float] = {
    "COERCIVE_COMMUNICATION": 1.8,   # Strong: coercive language directly linked
    "TIMELINE_GAP":           1.4,   # Strong: unexplained silence in active window
    "HIGH_PAGERANK":          0.9,   # Moderate: central position, not proof alone
    "HIGH_BETWEENNESS":       1.1,   # Moderate-high: bridge role implies coordination
    "ROLE_ORCHESTRATOR":      1.3,   # Moderate-high: assigned coordination role
    "ROLE_RECRUITER":         0.8,   # Moderate
    "ROLE_ENFORCER":          1.0,   # Moderate-high
    "OUTBOUND_DOMINANCE":     0.4,   # Low: consistently initiates more messages
}


def _sigmoid(x: float) -> float:
    """Sigmoid squashing: maps raw log-odds sum to [0, 1]."""
    return 1.0 / (1.0 + math.exp(-x))


def _compute_score(
    indicators: list[str],
    session_weights: dict[str, float] | None = None,
) -> tuple[float, list[dict]]:
    """
    Compute risk score and justification tree for a given indicator set.

    Args:
        indicators: list of active indicator names for this person
        session_weights: optional per-session weight overrides (Module 13)

    Returns:
        (score_0_to_100, justification_tree)
    """
    weights = {**INDICATOR_WEIGHTS, **(session_weights or {})}
    total_log_odds = 0.0
    tree: list[dict] = []

    for indicator in indicators:
        w = weights.get(indicator, 0.0)
        if w == 0.0:
            continue
        total_log_odds += w
        taxonomy = TAXONOMY_MAP.get(indicator, {})
        tree.append({
            "indicator": indicator,
            "weight": round(w, 3),
            "ncmec_ref": taxonomy.get("ncmec", ""),
            "iso27037_ref": taxonomy.get("iso27037", ""),
        })

    # No observed indicator means no score contribution.  Returning sigmoid(0)
    # (50) made an absence of evidence look like a fixed risk assessment.
    raw_score = _sigmoid(total_log_odds) * 100 if tree else 0.0
    return round(raw_score, 1), tree


def risk_score_engine(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: compute 0-100 risk score per Person with justification tree.

    Reads:
        state['gap_alerts']         (Module 09 / 11)
        state['coercive_matches']   (Module 05)
        state['pagerank']           (Module 06)
        state['betweenness']        (Module 06)
        state['roles']              (Module 07)
        state['domain_scores']      (Module 11)
        state.get('session_weights') (Module 13, optional)

    State outputs:
        risk_scores: dict[person_id -> {score, indicators, justification_tree}]
    """
    gap_alerts: list[dict] = state.get("gap_alerts", [])
    coercive_matches: list[dict] = state.get("coercive_matches", [])
    pagerank: dict = state.get("pagerank", {})
    betweenness: dict = state.get("betweenness", {})
    roles: dict = state.get("roles", {})
    raw_messages: list[dict[str, Any]] = state.get("raw_messages", [])
    domain_scores: dict = state.get("domain_scores", {})
    session_weights: dict = state.get("session_weights", {})
    lock_threshold: int = state.get("multi_factor_lock_threshold", 40)

    # ── Collect active indicators per person ──────────────────────────────
    person_indicators: dict[str, set] = {}

    # Gap alerts (only non-suppressed ones contribute to score)
    for alert in gap_alerts:
        if alert.get("flagged_as_anomaly") and not alert.get("suppressed"):
            parts = alert["pair_key"].split("->")
            if parts:
                pid = parts[0]
                person_indicators.setdefault(pid, set()).add("TIMELINE_GAP")

    # Coercive communication matches
    for match in coercive_matches:
        pid = match.get("sender_id")
        if pid:
            person_indicators.setdefault(pid, set()).add("COERCIVE_COMMUNICATION")

    # Centrality indicators (top 30th percentile by score).  A tied minimum is
    # not a "high" centrality signal: applying >= to it had marked every node
    # in small graphs and flattened person-level scores.
    pr_values = list(pagerank.values())
    bc_values = list(betweenness.values())
    pr_threshold = sorted(pr_values)[int(len(pr_values) * 0.7)] if pr_values else 0
    bc_threshold = sorted(bc_values)[int(len(bc_values) * 0.7)] if bc_values else 0

    for pid, pr in pagerank.items():
        if pr >= pr_threshold and pr > min(pr_values, default=pr):
            person_indicators.setdefault(pid, set()).add("HIGH_PAGERANK")

    for pid, bc in betweenness.items():
        if bc >= bc_threshold and bc > min(bc_values, default=bc):
            person_indicators.setdefault(pid, set()).add("HIGH_BETWEENNESS")

    # Role indicators
    for pid, role_info in roles.items():
        tag = role_info.get("tag", "")
        role_indicator = f"ROLE_{tag.upper()}"
        if role_indicator in INDICATOR_WEIGHTS:
            person_indicators.setdefault(pid, set()).add(role_indicator)

    # Directionality is person-specific evidence, unlike a shared pair-level
    # graph edge.  It prevents a symmetric graph topology from erasing a clear
    # initiator/recipient asymmetry in the raw exhibits.
    sent_counts: dict[str, int] = {}
    received_counts: dict[str, int] = {}
    for message in raw_messages:
        if message.get("is_quarantined"):
            continue
        sender = message.get("sender_id")
        receiver = message.get("receiver_id")
        if sender:
            sent_counts[str(sender)] = sent_counts.get(str(sender), 0) + 1
        if receiver:
            received_counts[str(receiver)] = received_counts.get(str(receiver), 0) + 1
    for person_id, sent in sent_counts.items():
        if sent > received_counts.get(person_id, 0):
            person_indicators.setdefault(person_id, set()).add("OUTBOUND_DOMINANCE")

    # ── Compute scores ────────────────────────────────────────────────────
    risk_scores: dict[str, dict] = {}
    updates = []

    all_people = set(list(pagerank) + list(betweenness) + list(roles) + list(sent_counts) + list(received_counts))

    for person_id in all_people:
        indicators = list(person_indicators.get(person_id, set()))
        raw_score, justification_tree = _compute_score(
            indicators, session_weights.get(person_id)
        )

        # ── Multi-factor lock (Module 11) ──────────────────────────────────
        # Score above threshold requires 3+ independent evidence domains
        active_domains = domain_scores.get(person_id, [])
        # Also count indicators as evidence domains
        indicator_domains = set(ind.split("_")[0] for ind in indicators)
        all_domains = set(active_domains) | indicator_domains
        if raw_score > lock_threshold and len(all_domains) < 3:
            # Preserve the evidence-derived ordering while limiting a score
            # supported by fewer than three independent domains.  The former
            # hard clamp to exactly 40 was the source of the flatline.
            raw_score = min(raw_score, lock_threshold + 10 * len(all_domains))
            raw_score = round(raw_score, 1)

        role_info = roles.get(person_id, {})
        risk_scores[person_id] = {
            "person_id": person_id,
            "score": raw_score,
            "indicators": indicators,
            "justification_tree": justification_tree,
            "role_tag": role_info.get("tag", "Peripheral"),
            "role_justification": role_info.get("justification", ""),
            "pagerank": round(pagerank.get(person_id, 0.0), 6),
            "betweenness": round(betweenness.get(person_id, 0.0), 6),
        }

        updates.append({"id": person_id, "risk_score": raw_score})

    if updates:
        try:
            write_person_properties(updates, case_id=state.get("case_id", "default"))
        except Exception as exc:
            print(f"[WARN] Risk Score graph write failed: {exc}")

    state["risk_scores"] = risk_scores
    return state
