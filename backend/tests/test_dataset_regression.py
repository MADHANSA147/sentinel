"""End-to-end, offline regressions for the four synthetic demo datasets."""

from __future__ import annotations

import asyncio

import pytest

from app.agents import module_05_words, module_06_network, module_07_roles
from app.agents import module_09_gap, module_11_context, module_12_risk
from app.api import dashboard as dashboard_api
from app.api import pipeline as pipeline_api
from app.api import hitl as hitl_api


@pytest.fixture(autouse=True)
def offline_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep dataset behavior tests deterministic and independent of services."""
    pipeline_api._case_states.clear()
    monkeypatch.setattr(pipeline_api, "batch_load_graph", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline_api, "embed_messages", lambda *args, **kwargs: None)
    monkeypatch.setattr(module_06_network, "write_person_properties", lambda *args, **kwargs: None)
    monkeypatch.setattr(module_07_roles, "write_person_properties", lambda *args, **kwargs: None)
    monkeypatch.setattr(module_12_risk, "write_person_properties", lambda *args, **kwargs: None)
    monkeypatch.setattr(module_09_gap, "_write_gap_edge", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module_05_words,
        "get_collection",
        lambda: (_ for _ in ()).throw(RuntimeError("offline test")),
    )
    monkeypatch.setattr(
        module_11_context,
        "get_collection",
        lambda: (_ for _ in ()).throw(RuntimeError("offline test")),
    )


def _run_dashboard(case_id: str) -> dict:
    asyncio.run(pipeline_api.run_case(case_id))
    return asyncio.run(dashboard_api.get_dashboard(case_id))


def test_all_datasets_produce_scoped_distinct_outputs() -> None:
    dashboards = {
        case_id: _run_dashboard(case_id)
        for case_id in ("case-dataset1", "case-dataset2", "case-dataset3", "case-dataset4")
    }

    # Dataset 3 intentionally reuses Dataset 1's participants while exercising
    # malformed-record quarantine.  The other datasets have distinct graph
    # populations; no response may be a stale previous dashboard.
    assert len({tuple(node["id"] for node in data["graph"]["nodes"])
                for data in dashboards.values()}) == 3
    assert len({tuple((edge["source"], edge["target"], edge["weight"])
                for edge in data["graph"]["comm_edges"])
                for data in dashboards.values()}) == 4
    assert len({data["theories"][0]["theory"] for data in dashboards.values()}) == 4

    dataset1 = dashboards["case-dataset1"]
    assert dataset1["graph"]["gap_badge_count"] >= 1
    assert any(not alert["suppressed"] for alert in dataset1["alert_list"])

    dataset2 = dashboards["case-dataset2"]
    assert dataset2["graph"]["gap_badge_count"] == 0
    assert len(dataset2["alert_list"]) == 1
    assert dataset2["alert_list"][0]["suppressed"] is True
    assert "night shift" in dataset2["alert_list"][0]["suppression_reason"].lower()

    dataset3_state = pipeline_api.get_case_state("case-dataset3")
    assert len(dataset3_state["raw_messages"]) == 30
    assert len(dataset3_state["quarantined_ids"]) == 9

    dataset4 = dashboards["case-dataset4"]
    tags = {node["id"]: node["role"] for node in dataset4["graph"]["nodes"]}
    assert dataset4["graph"]["gap_badge_count"] == 0
    assert tags["U-301"] == "Orchestrator"
    assert tags["U-305"] == "Orchestrator"
    assert tags["U-304"] == "Bridge"


def test_run_endpoint_preserves_response_contract_for_each_dataset() -> None:
    for case_id in ("case-dataset1", "case-dataset2", "case-dataset3", "case-dataset4"):
        response = asyncio.run(pipeline_api.run_case(case_id))

        assert response["case_id"] == case_id
        assert response["status"] == "completed"
        assert set(response) == {
            "case_id",
            "status",
            "persons_analysed",
            "gap_alerts",
            "theories",
            "simulation",
        }


def test_hitl_rejection_recomputes_the_affected_person_score() -> None:
    _run_dashboard("case-dataset1")
    before = pipeline_api.get_case_state("case-dataset1")["risk_scores"]["U-101"]["score"]

    result = asyncio.run(hitl_api.submit_decision(
        "case-dataset1",
        hitl_api.FlagDecision(
            person_id="U-101", indicator="TIMELINE_GAP", action="reject"
        ),
    ))

    assert result["new_score"] < before
    refreshed = asyncio.run(dashboard_api.get_dashboard("case-dataset1"))
    assert next(
        person for person in refreshed["priority_board"] if person["person_id"] == "U-101"
    )["score"] == result["new_score"]
