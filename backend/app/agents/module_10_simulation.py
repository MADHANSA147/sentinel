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


def _build_prompt(risk_scores: dict, gap_alerts: list, roles: dict) -> str:
    """Build the prompt summarising current evidence state."""
    top_suspects = sorted(
        risk_scores.values(), key=lambda x: x["score"], reverse=True
    )[:3]

    flagged_gaps = [a for a in gap_alerts if a.get("flagged_as_anomaly") and not a.get("suppressed")]

    lines = ["Given the following evidence from a digital forensic investigation:"]
    lines.append(f"\nTop suspects by risk score:")
    for s in top_suspects:
        lines.append(
            f"  - {s['person_id']}: score={s['score']}, role={s['role_tag']}, "
            f"indicators={s['indicators']}"
        )
    lines.append(f"\nUnexplained communication gaps: {len(flagged_gaps)}")
    lines.append(
        "\nGenerate exactly 2 competing investigative theories with relative "
        "likelihood percentages that sum to 100. Return as JSON: "
        '[{"theory": "...", "likelihood": 60}, {"theory": "...", "likelihood": 40}]'
    )
    return "\n".join(lines)


def _rule_based_theories(risk_scores: dict, gap_alerts: list) -> list[dict]:
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

    if score > 60 and flagged_gaps:
        return [
            {
                "theory": (
                    f"{top_person} is coordinating covert activity — "
                    f"high risk score ({score}) combined with unexplained "
                    f"communication blackout periods suggests deliberate operational security."
                ),
                "likelihood": 70,
            },
            {
                "theory": (
                    f"Communication patterns are consistent with legitimate coordinated "
                    f"activity (e.g. shift work, restricted connectivity) — "
                    f"gaps may be explained by external scheduling constraints."
                ),
                "likelihood": 30,
            },
        ]
    else:
        return [
            {
                "theory": (
                    f"Current evidence is inconclusive — network patterns show "
                    f"activity but without corroborating coercive language or "
                    f"unexplained blackouts, a benign explanation cannot be ruled out."
                ),
                "likelihood": 55,
            },
            {
                "theory": (
                    f"Preliminary indicators suggest coordinated communication — "
                    f"further evidence collection recommended before escalation."
                ),
                "likelihood": 45,
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

    # Try LLM first if API key is available
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")

    if api_key:
        try:
            prompt = _build_prompt(risk_scores, gap_alerts, roles)
            if os.environ.get("ANTHROPIC_API_KEY"):
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                message = client.messages.create(
                    model="claude-opus-4-5",
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = message.content[0].text.strip()
            else:
                import openai
                client = openai.OpenAI(api_key=api_key)
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=512,
                )
                raw = resp.choices[0].message.content.strip()

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
                        return state
        except Exception as exc:
            print(f"[WARN] LLM theory generation failed, using rule-based: {exc}")

    # Fallback
    state["theories"] = _rule_based_theories(risk_scores, gap_alerts)
    return state
