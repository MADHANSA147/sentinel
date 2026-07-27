"""
SENTINEL — Pydantic models for raw message ingestion (Module 01).
All timestamps are normalized to UTC immediately on parse.
Invalid/incomplete records are quarantined, never dropped.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, model_validator


# Regex that flags suspect content: repeated '?', replacement char, or C0 control bytes
SUSPECT_CONTENT_RE = re.compile(r"\?{2,}|\ufffd|[\x00-\x08\x0b\x0c\x0e-\x1f]")


class MessageMetadata(BaseModel):
    """Non-forensic metadata carried alongside each raw message."""

    deleted_flag: bool = False
    has_attachment: bool = False


class MessageFlags(BaseModel):
    """Quality / quarantine flags computed during ingestion."""

    timestamp_missing: bool = False
    sender_missing: bool = False
    content_missing: bool = False
    content_suspect: bool = False

    @property
    def is_quarantined(self) -> bool:
        """A record is quarantined if any flag is set."""
        return (
            self.timestamp_missing
            or self.sender_missing
            or self.content_missing
            or self.content_suspect
        )


class RawMessage(BaseModel):
    """
    Module 01 — canonical message record.

    Normalises timestamps to UTC regardless of source offset ("Z" or "+05:30").
    Quarantines rather than drops malformed records — chain-of-custody requirement.
    """

    message_id: str
    timestamp: Optional[datetime] = None
    sender_id: Optional[str] = None
    receiver_id: Optional[str] = None
    content: Optional[str] = None
    platform: str
    metadata: MessageMetadata = Field(default_factory=MessageMetadata)
    flags: MessageFlags = Field(default_factory=MessageFlags)

    @model_validator(mode="before")
    @classmethod
    def _normalise_and_flag(cls, data: dict) -> dict:  # type: ignore[override]
        """
        Run before Pydantic field validation so flags are set on the raw dict,
        allowing the model to construct cleanly even with missing fields.
        """
        # ── Timestamp: normalise any tz-aware datetime to UTC ─────────────────
        ts_raw = data.get("timestamp")
        ts_missing = ts_raw is None

        if not ts_missing:
            try:
                if isinstance(ts_raw, str):
                    # datetime.fromisoformat handles both "Z" (py 3.11+) and offsets.
                    # Fall back to manual 'Z' → '+00:00' replacement for older Pythons.
                    parsed = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                    data["timestamp"] = parsed.astimezone(timezone.utc)
                elif isinstance(ts_raw, datetime):
                    if ts_raw.tzinfo is not None:
                        data["timestamp"] = ts_raw.astimezone(timezone.utc)
                    else:
                        # Naive datetimes: assume UTC rather than silently misreading
                        data["timestamp"] = ts_raw.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                ts_missing = True
                data["timestamp"] = None

        # ── Build flags dict ───────────────────────────────────────────────────
        content = data.get("content")
        flags = {
            "timestamp_missing": ts_missing,
            "sender_missing": data.get("sender_id") is None,
            "content_missing": content is None,
            "content_suspect": (
                bool(SUSPECT_CONTENT_RE.search(content)) if content else False
            ),
        }
        data["flags"] = flags

        # Preserve existing metadata if already provided
        if "metadata" not in data:
            data["metadata"] = {}

        return data


class IngestResponse(BaseModel):
    """Response envelope for POST /api/v1/ingest."""

    filename: str
    sha256_hash: str
    total: int
    validated: int
    quarantined: int
    quarantined_ids: list[str]
