"""SENTINEL — export endpoints (Court Pack PDF + Audit Bundle JSON)."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, JSONResponse

from app.api.pipeline import get_case_state
from app.services.ledger import get_ledger, verify_chain
from app.services.pdf_export import generate_court_pack

router = APIRouter(prefix="/api/v1", tags=["export"])


@router.get("/export/court-pack/{case_id}")
async def download_court_pack(case_id: str) -> Response:
    """
    Generate and return the Court Pack PDF — raw exhibits only, no AI scores.
    Track B output per the dual-track architecture.
    """
    state = get_case_state(case_id)
    if not state:
        raise HTTPException(status_code=404, detail="Case not found.")

    messages = state.get("raw_messages", [])
    ingestion_hash = state.get("ingestion_hash", "UNKNOWN")
    ledger_entries = get_ledger(case_id)
    coercive_matches = state.get("coercive_matches", [])

    # Build taxonomy match list for Part B
    from app.agents.module_12_risk import TAXONOMY_MAP
    taxonomy_exhibits = []
    for match in coercive_matches:
        taxonomy_exhibits.append({
            "message_id": match.get("message_id", ""),
            "ncmec_ref": TAXONOMY_MAP.get("COERCIVE_COMMUNICATION", {}).get("ncmec", ""),
            "iso27037_ref": TAXONOMY_MAP.get("COERCIVE_COMMUNICATION", {}).get("iso27037", ""),
        })

    pdf_bytes = generate_court_pack(
        case_id=case_id,
        messages=messages,
        ingestion_hash=ingestion_hash,
        ledger_entries=ledger_entries,
        taxonomy_matches=taxonomy_exhibits,
    )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="court_pack_{case_id}.pdf"'},
    )


@router.get("/export/audit-bundle/{case_id}")
async def download_audit_bundle(case_id: str) -> JSONResponse:
    """
    Export a plain JSON audit bundle: ledger, graph snapshot, ingestion hash.
    Simplified version — production would encrypt and sign this bundle.
    """
    state = get_case_state(case_id)
    if not state:
        raise HTTPException(status_code=404, detail="Case not found.")

    ledger = get_ledger(case_id)
    chain_valid = verify_chain(case_id)

    bundle = {
        "case_id": case_id,
        "ingestion_hash": state.get("ingestion_hash", "UNKNOWN"),
        "chain_integrity": chain_valid,
        "ledger": ledger,
        "graph_snapshot": {
            "persons": state.get("person_nodes", []),
            "centrality": state.get("centrality_summary", []),
        },
        "appeal_log": [],  # populated from Module 13 in full session
    }

    return JSONResponse(
        content=bundle,
        headers={
            "Content-Disposition": f'attachment; filename="audit_bundle_{case_id}.json"'
        },
    )


@router.get("/ledger/{case_id}")
async def get_case_ledger(case_id: str) -> dict:
    """Return the full execution ledger for a case."""
    ledger = get_ledger(case_id)
    chain_valid = verify_chain(case_id)
    return {
        "case_id": case_id,
        "chain_valid": chain_valid,
        "entries": ledger,
        "total_steps": len(ledger),
    }
