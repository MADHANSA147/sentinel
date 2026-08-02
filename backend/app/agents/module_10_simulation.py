"""
Module 10 — Case Simulation (LangGraph node).
One LLM call returning exactly 2 competing theories with likelihoods
summing to 100. Simple, no iterative simulation loop.
Falls back to rule-based if LLM unavailable.
"""

from __future__ import annotations

import json
import os
from typing import Any


def _build_prompt(
    case_id: str,
    risk_scores: dict,
    gap_alerts: list,
    roles: dict,
    raw_messages: list[dict[str, Any]],
) -> str:
    """Build a bounded, case-specific evidence brief for the LLM."""
    top_suspects = sorted(
        risk_scores.values(), key=lambda x: x["score"], reverse=True
    )[:3]

    flagged_gaps = [a for a in gap_alerts if a.get("flagged_as_anomaly") and not a.get("suppressed")]

    lines = [
        "You are assisting an investigator. Do not infer facts not present in the "
        "case evidence. Treat all indicators as leads, not conclusions.",
        f"Case identifier: {case_id}",
        "Evidence graph and message exhibits for this case:",
    ]
    lines.append(f"\nTop suspects by risk score:")
    for s in top_suspects:
        lines.append(
            f"  - {s['person_id']}: score={s['score']}, role={s['role_tag']}, "
            f"indicators={s['indicators']}; justification={s.get('justification_tree', [])}"
        )
    lines.append(f"\nUnexplained communication gaps: {len(flagged_gaps)}")
    for gap in flagged_gaps:
        lines.append(
            f"  - {gap.get('pair_key')}: {gap.get('before')} to {gap.get('after')} "
            f"({gap.get('gap_hours')}h; type={gap.get('alert_type')})"
        )
    lines.append("\nObserved roles:")
    for person_id, role in sorted(roles.items()):
        lines.append(f"  - {person_id}: {role.get('tag')} — {role.get('justification', '')}")

    lines.append("\nRaw message excerpts (chronological, capped at 40):")
    visible_messages = [m for m in raw_messages if not m.get("is_quarantined")][:40]
    for message in visible_messages:
        content = " ".join(str(message.get("content") or "").split())[:220]
        lines.append(
            f"  - {message.get('message_id')} | {message.get('timestamp')} | "
            f"{message.get('sender_id')}→{message.get('receiver_id')}: {content}"
        )
    lines.append(
        "\nGenerate exactly 2 competing investigative theories with relative "
        "likelihood percentages that sum to 100. Return as JSON: "
        '[{"theory": "...", "likelihood": 60}, {"theory": "...", "likelihood": 40}]'
    )
    return "\n".join(lines)


def _rule_based_theories(
    risk_scores: dict,
    gap_alerts: list,
    raw_messages: list[dict[str, Any]],
) -> list[dict]:
    """Fallback: generate theories from rule-based logic."""
    top = sorted(risk_scores.values(), key=lambda x: x["score"], reverse=True)
    flagged_gaps = [a for a in gap_alerts if a.get("flagged_as_anomaly") and not a.get("suppressed")]

    if not top:
        return [
            {"theory": "Insufficient evidence to form theories.", "likelihood": 50},
            {"theory": "Normal communication patterns, no suspicious activity.", "likelihood": 50},
        ]

    top_person = top[0]["person_id"]
    score = top[0]["score"]
    role = top[0].get("role_tag", "Unclassified")

    active_messages = [m for m in raw_messages if not m.get("is_quarantined")]
    focal_indicators = top[0].get("indicators", [])
    coercive_count = focal_indicators.count("COERCIVE_COMMUNICATION")
    gap_count = len(flagged_gaps)
    message_count = len(active_messages)
    participant_count = len({
        str(person_id)
        for message in active_messages
        for person_id in (message.get("sender_id"), message.get("receiver_id"))
        if person_id
    })
    adverse_likelihood = min(
        85,
        25
        + min(int(score / 5), 30)
        + 12 * gap_count
        + 10 * coercive_count
        + max(0, participant_count - 2) * 3
        + min(message_count // 5, 10),
    )
    benign_likelihood = 100 - adverse_likelihood

    if score > 60 and flagged_gaps:
        return [
            {
                "theory": (
                    f"{top_person} is coordinating covert activity — "
                    f"high risk score ({score}) combined with unexplained "
                    f"communication blackout periods suggests deliberate operational security."
                ),
                "likelihood": adverse_likelihood,
            },
            {
                "theory": (
                    f"Communication patterns are consistent with legitimate coordinated "
                    f"activity (e.g. shift work, restricted connectivity) — "
                    f"gaps may be explained by external scheduling constraints."
                ),
                "likelihood": benign_likelihood,
            },
        ]
    else:
        return [
            {
                "theory": (
                    f"{top_person} is the current focal person (score {score}; {role}) "
                    f"across {message_count} usable messages. The available evidence "
                    f"does not independently corroborate coercive communication or an "
                    f"unexplained blackout, so a benign explanation remains plausible."
                ),
                "likelihood": benign_likelihood,
            },
            {
                "theory": (
                    f"The observed structural pattern around {top_person} may reflect "
                    f"coordinated communication. Seek corroborating source material "
                    f"before escalation; the current score is an evidence trace, not a conclusion."
                ),
                "likelihood": adverse_likelihood,
            },
        ]


def case_simulation(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: generate 2 competing investigative theories.

    State outputs:
        theories: list of 2 dicts with 'theory' and 'likelihood' keys
    """
    risk_scores = state.get("risk_scores", {})
    gap_alerts = state.get("gap_alerts", [])
    roles = state.get("roles", {})
    raw_messages: list[dict[str, Any]] = state.get("raw_messages", [])
    case_id = str(state.get("case_id", "default"))

    # Groq exposes an OpenAI-compatible endpoint, so the existing client is retained.
    api_key = os.environ.get("GROQ_API_KEY")

    fallback_reason = "Groq API key is not configured."
    if api_key:
        try:
            prompt = _build_prompt(case_id, risk_scores, gap_alerts, roles, raw_messages)
            import openai

            client = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
                timeout=5.0,
                max_retries=0,
            )
            resp = client.chat.completions.create(
                model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
                messages=[
                    {
                        "role": "system",
                        "content": "Return only valid JSON. Ground each theory in the supplied case evidence.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=512,
                temperature=0.2,
            )
            raw = (resp.choices[0].message.content or "").strip()

            # Extract JSON from response
            if "[" in raw and "]" in raw:
                start = raw.index("[")
                end = raw.rindex("]") + 1
                theories = json.loads(raw[start:end])
                # Validate structure
                if len(theories) == 2 and all("theory" in t and "likelihood" in t for t in theories):
                    total = sum(t["likelihood"] for t in theories)
                    if total == 100:
                        state["theories"] = theories
                        state["simulation"] = {
                            "mode": "live",
                            "label": "Live Groq analysis",
                            "reason": None,
                        }
                        return state
            fallback_reason = "Groq returned an invalid theory response."
        except Exception as exc:
            print(f"[WARN] LLM theory generation failed, using rule-based: {exc}")
            fallback_reason = "Live Groq analysis request failed."

    # The fallback is deliberately marked so it cannot be mistaken for live reasoning.
    state["theories"] = _rule_based_theories(risk_scores, gap_alerts, raw_messages)
    state["simulation"] = {
        "mode": "fallback",
        "label": "⚠ Live analysis unavailable — showing placeholder",
        "reason": fallback_reason,
    }
    return state
