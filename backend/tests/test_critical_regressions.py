"""End-to-end regression guards for the Court Pack review findings."""

from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace
from pathlib import Path

from app.agents import module_05_words
from app.agents.module_10_simulation import (
    _build_prompt,
    _rule_based_theories,
    case_simulation,
)
from app.agents.module_12_risk import risk_score_engine
from app.services.pdf_export import _content_excerpt


ROOT = Path(__file__).parents[2]


def test_dataset1_operational_language_does_not_pass_grooming_gate() -> None:
    messages = json.loads(
        (ROOT / "data" / "synthetic" / "whatsapp_conversation_40_messages.json").read_text(
            encoding="utf-8"
        )
    )
    assert not any(
        module_05_words._has_explicit_coercive_language(message["content"])
        for message in messages
    )
    assert module_05_words._has_explicit_coercive_language(
        "Don't tell anyone about this; meet me alone."
    )


def test_all_synthetic_court_pack_content_hashes_are_unique() -> None:
    for dataset in (ROOT / "data" / "synthetic").glob("*.json"):
        messages = json.loads(dataset.read_text(encoding="utf-8"))
        hashes = [
            hashlib.sha256(message["content"].encode()).hexdigest()
            for message in messages
            if isinstance(message, dict) and message.get("content")
        ]
        assert len(hashes) == len(set(hashes)), dataset.name


def test_court_pack_uses_a_readable_escaped_excerpt() -> None:
    assert _content_excerpt("  Read <this> & preserve it.  ") == "Read &lt;this&gt; &amp; preserve it."


def test_risk_scores_remain_person_specific_when_evidence_differs(monkeypatch) -> None:
    monkeypatch.setattr("app.agents.module_12_risk.write_person_properties", lambda *args, **kwargs: None)
    state = risk_score_engine(
        {
            "case_id": "case-risk",
            "gap_alerts": [{"pair_key": "U-1->U-2", "flagged_as_anomaly": True}],
            "coercive_matches": [],
            "pagerank": {"U-1": 0.8, "U-2": 0.2},
            "betweenness": {"U-1": 0.5, "U-2": 0.0},
            "roles": {"U-1": {"tag": "Orchestrator"}, "U-2": {"tag": "Target"}},
        }
    )
    assert state["risk_scores"]["U-1"]["score"] > state["risk_scores"]["U-2"]["score"]
    assert "TIMELINE_GAP" in state["risk_scores"]["U-1"]["indicators"]
    assert not state["risk_scores"]["U-2"]["indicators"]


def test_simulation_prompt_and_fallback_change_with_case_evidence() -> None:
    scores = {"U-1": {"person_id": "U-1", "score": 61, "role_tag": "Bridge", "indicators": []}}
    prompt = _build_prompt(
        "case-a", scores, [], {"U-1": {"tag": "Bridge", "justification": "broker"}},
        [{"message_id": "M-1", "sender_id": "U-1", "receiver_id": "U-2", "content": "Case-specific excerpt"}],
    )
    assert "case-a" in prompt and "Case-specific excerpt" in prompt

    quiet = _rule_based_theories(scores, [], [{"is_quarantined": False}])
    gap = _rule_based_theories(
        scores,
        [{"flagged_as_anomaly": True, "suppressed": False}],
        [{"is_quarantined": False}] * 10,
    )
    assert quiet != gap
    assert sum(item["likelihood"] for item in gap) == 100


def test_case_simulation_uses_groq_with_case_specific_evidence(monkeypatch) -> None:
    """The Groq path must receive each case's real evidence brief, not a template."""
    prompts: list[str] = []

    class FakeCompletions:
        def create(self, **kwargs):
            prompt = kwargs["messages"][1]["content"]
            prompts.append(prompt)
            likelihood = 61 if "case-alpha" in prompt else 72
            raw = json.dumps([
                {"theory": f"Theory grounded in {prompt.splitlines()[1]}", "likelihood": likelihood},
                {"theory": "Competing evidence-based theory", "likelihood": 100 - likelihood},
            ])
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=raw))]
            )

    class FakeOpenAI:
        def __init__(self, *, api_key: str, base_url: str, timeout: float, max_retries: int) -> None:
            assert api_key == "test-key"
            assert base_url == "https://api.groq.com/openai/v1"
            assert timeout == 5.0
            assert max_retries == 0
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    def run(case_id: str, content: str) -> list[dict]:
        return case_simulation({
            "case_id": case_id,
            "risk_scores": {"U-1": {
                "person_id": "U-1", "score": 65, "role_tag": "Bridge", "indicators": [],
            }},
            "gap_alerts": [],
            "roles": {"U-1": {"tag": "Bridge", "justification": "broker"}},
            "raw_messages": [{
                "message_id": "MSG-1", "sender_id": "U-1", "receiver_id": "U-2", "content": content,
            }],
        })["theories"]

    alpha = run("case-alpha", "Alpha-only message exhibit")
    beta = run("case-beta", "Beta-only message exhibit")

    assert len(prompts) == 2
    assert "Alpha-only message exhibit" in prompts[0]
    assert "Beta-only message exhibit" in prompts[1]
    assert [item["likelihood"] for item in alpha] == [61, 39]
    assert [item["likelihood"] for item in beta] == [72, 28]
