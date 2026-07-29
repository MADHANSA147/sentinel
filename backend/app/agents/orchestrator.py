"""
SENTINEL — LangGraph orchestrator wiring all 12 agents into one pipeline.
Modules execute in dependency order; each updates the shared state dict.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import StateGraph, END

from app.agents.module_02_identity import identity_fusion
from app.agents.module_08_timeline import timeline_engine
from app.agents.module_04_baseline import behavioral_print
from app.agents.module_06_network import network_mapping
from app.agents.module_03_profiling import subject_profiling
from app.agents.module_07_roles import role_discovery
from app.agents.module_05_words import word_patterns
from app.agents.module_09_gap import gap_detector
from app.agents.module_11_context import exculpatory_context
from app.agents.module_12_risk import risk_score_engine
from app.agents.module_10_simulation import case_simulation
from app.agents.module_13_feedback import feedback_loop


def build_pipeline() -> Any:
    """Construct and compile the SENTINEL LangGraph pipeline."""

    graph = StateGraph(dict)

    # ── Register nodes ─────────────────────────────────────────────────────
    graph.add_node("identity_fusion",    identity_fusion)
    graph.add_node("timeline_engine",    timeline_engine)
    graph.add_node("behavioral_print",   behavioral_print)
    graph.add_node("network_mapping",    network_mapping)
    graph.add_node("subject_profiling",  subject_profiling)
    graph.add_node("role_discovery",     role_discovery)
    graph.add_node("word_patterns",      word_patterns)
    graph.add_node("gap_detector",       gap_detector)
    graph.add_node("exculpatory_context", exculpatory_context)
    graph.add_node("risk_score_engine",  risk_score_engine)
    graph.add_node("case_simulation",    case_simulation)
    graph.add_node("feedback_loop",      feedback_loop)

    # ── Define execution order ─────────────────────────────────────────────
    graph.set_entry_point("identity_fusion")

    graph.add_edge("identity_fusion",    "timeline_engine")
    graph.add_edge("timeline_engine",    "behavioral_print")
    graph.add_edge("behavioral_print",   "network_mapping")
    graph.add_edge("network_mapping",    "subject_profiling")
    graph.add_edge("subject_profiling",  "role_discovery")
    graph.add_edge("role_discovery",     "word_patterns")
    graph.add_edge("word_patterns",      "gap_detector")
    graph.add_edge("gap_detector",       "exculpatory_context")
    graph.add_edge("exculpatory_context","risk_score_engine")
    graph.add_edge("risk_score_engine",  "case_simulation")
    graph.add_edge("case_simulation",    "feedback_loop")
    graph.add_edge("feedback_loop",      END)

    return graph.compile()


# Singleton compiled pipeline
_pipeline = None


def get_pipeline() -> Any:
    global _pipeline
    if _pipeline is None:
        _pipeline = build_pipeline()
    return _pipeline


def run_pipeline(initial_state: dict | None = None) -> dict:
    """Run the full agent pipeline and return the final state."""
    state = initial_state or {}
    pipeline = get_pipeline()
    return pipeline.invoke(state)
