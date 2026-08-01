"""
SENTINEL — Execution Ledger (SHA-256 chained audit log).
Every agent step is hashed, chained to the previous entry and to the
Module 01 raw-file hash. Designed to maximize reproducibility.

Note: "designed to maximize reproducibility" is the correct claim —
not absolute determinism. GPU floating-point and cloud API calls are
not strictly bit-exact even at temperature 0.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

# In-memory ledger storage per case_id
_ledger: dict[str, list[dict]] = {}


def clear_ledger(case_id: str) -> None:
    """Start a fresh, case-specific execution chain for a new pipeline run."""
    _ledger.pop(case_id, None)


def _hash_entry(entry: dict) -> str:
    """SHA-256 hash of an entry's canonical JSON representation."""
    canonical = json.dumps(entry, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def log_step(
    case_id: str,
    module_name: str,
    input_summary: str,
    output_summary: str,
    raw_file_hash: str | None = None,
) -> dict:
    """
    Append a new ledger entry, chained to the previous entry's hash.

    Args:
        case_id:         Unique identifier for this investigation case
        module_name:     Which module/agent produced this entry
        input_summary:   Short description of inputs
        output_summary:  Short description of outputs
        raw_file_hash:   Module 01 SHA-256 of the original evidence file

    Returns:
        The completed ledger entry including its own hash
    """
    case_ledger = _ledger.setdefault(case_id, [])
    prev_hash = case_ledger[-1]["entry_hash"] if case_ledger else (raw_file_hash or "GENESIS")

    entry: dict[str, Any] = {
        "case_id": case_id,
        "step_index": len(case_ledger),
        "module": module_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_summary": input_summary,
        "output_summary": output_summary,
        "raw_file_hash": raw_file_hash,
        "prev_hash": prev_hash,
    }

    entry["entry_hash"] = _hash_entry(entry)
    case_ledger.append(entry)
    return entry


def get_ledger(case_id: str) -> list[dict]:
    """Return the full ledger for a case."""
    return list(_ledger.get(case_id, []))


def verify_chain(case_id: str) -> bool:
    """
    Verify ledger integrity: each entry's prev_hash must match the
    previous entry's entry_hash.
    """
    case_ledger = _ledger.get(case_id, [])
    for i in range(1, len(case_ledger)):
        expected_prev = case_ledger[i - 1]["entry_hash"]
        if case_ledger[i]["prev_hash"] != expected_prev:
            return False
    return True
