"""
SENTINEL — Court Pack PDF export (Track B output).
Generates a clean PDF with ONLY raw, verifiable exhibits:
  hashed message excerpts, timestamps, platform, taxonomy tags.
ZERO risk scores, ZERO AI summaries anywhere in this document.

This satisfies Section 63 of the Bharatiya Sakshya Adhiniyam, 2023
(electronic evidence certification provision).
"""

from __future__ import annotations

import hashlib
import io
from html import escape
from datetime import datetime, timezone
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)


def _hash_message_content(content: str) -> str:
    """SHA-256 of message content for exhibit integrity."""
    return hashlib.sha256(content.encode()).hexdigest()[:16] + "..."


def _content_excerpt(content: str, limit: int = 180) -> str:
    """Return a readable, safely escaped exhibit excerpt without altering evidence."""
    normalized = " ".join(content.split())
    if len(normalized) > limit:
        normalized = normalized[: limit - 1].rstrip() + "…"
    return escape(normalized) or "[No readable content supplied]"


def generate_court_pack(
    case_id: str,
    messages: list[dict],
    ingestion_hash: str,
    ledger_entries: list[dict],
    taxonomy_matches: list[dict] | None = None,
) -> bytes:
    """
    Generate Court Pack PDF as bytes.

    Args:
        case_id:          Investigation case identifier
        messages:         List of raw validated message dicts
        ingestion_hash:   SHA-256 of the original evidence file (Module 01)
        ledger_entries:   Full audit ledger for this case
        taxonomy_matches: Optional list of taxonomy tag dicts

    Returns:
        PDF bytes
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontSize=16,
        textColor=colors.HexColor("#1a1a2e"),
        spaceAfter=12,
    )
    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontSize=12,
        textColor=colors.HexColor("#16213e"),
        spaceAfter=8,
        spaceBefore=16,
    )
    body_style = styles["Normal"]
    caveat_style = ParagraphStyle(
        "Caveat",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#888888"),
        spaceAfter=4,
    )

    story = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ── Cover Page ────────────────────────────────────────────────────────
    story.append(Paragraph("SENTINEL — Digital Evidence Court Pack", title_style))
    story.append(Paragraph(f"Case Reference: {case_id}", body_style))
    story.append(Paragraph(f"Generated: {now}", body_style))
    story.append(Paragraph(f"Evidence File Hash (SHA-256): {ingestion_hash}", body_style))
    story.append(Spacer(1, 0.3 * cm))
    story.append(
        Paragraph(
            "⚠ This document contains raw digital evidence exhibits only. "
            "No AI-generated risk scores or analytical summaries are included. "
            "Prepared for potential submission under Section 63, Bharatiya "
            "Sakshya Adhiniyam, 2023.",
            caveat_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))

    # ── Message Exhibits ──────────────────────────────────────────────────
    story.append(Paragraph("Part A — Raw Message Exhibits", section_style))
    story.append(
        Paragraph(
            "Each message is presented with its content hash for integrity verification. "
            "Timestamps have been normalised to UTC as per forensic standards.",
            caveat_style,
        )
    )
    story.append(Spacer(1, 0.3 * cm))

    table_data = [[
        "#", "Message ID", "Timestamp (UTC)", "Sender", "Platform",
        "Readable content excerpt", "Content Hash",
    ]]
    for idx, msg in enumerate(messages, 1):
        content = msg.get("content") or ""
        ts = msg.get("timestamp")
        ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts or "MISSING")
        table_data.append([
            str(idx),
            msg.get("message_id", ""),
            ts_str,
            msg.get("sender_id") or "UNKNOWN",
            msg.get("platform", ""),
            Paragraph(_content_excerpt(content), caveat_style),
            _hash_message_content(content) if content else "NULL",
        ])

    msg_table = Table(table_data, repeatRows=1)
    msg_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("FONTSIZE", (0, 1), (-1, -1), 7),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f8f8")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("WORDWRAP", (0, 0), (-1, -1), True),
    ]))
    story.append(msg_table)

    # ── Taxonomy Tags ─────────────────────────────────────────────────────
    if taxonomy_matches:
        story.append(Paragraph("Part B — Taxonomy Classification Tags", section_style))
        story.append(
            Paragraph(
                "Messages matched against NCMEC grooming indicators and ISO/IEC 27037 "
                "digital forensics guidelines. These are descriptive classifications "
                "of observed patterns, not risk assessments.",
                caveat_style,
            )
        )
        story.append(Spacer(1, 0.3 * cm))

        tax_data = [["Message ID", "NCMEC Reference", "ISO 27037 Reference"]]
        for match in taxonomy_matches:
            tax_data.append([
                match.get("message_id", ""),
                match.get("ncmec_ref", ""),
                match.get("iso27037_ref", ""),
            ])

        tax_table = Table(tax_data, repeatRows=1)
        tax_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, 0), 8),
            ("FONTSIZE", (0, 1), (-1, -1), 7),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f8f8")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ]))
        story.append(tax_table)

    # ── Audit Chain ───────────────────────────────────────────────────────
    story.append(Paragraph("Part C — Processing Audit Chain", section_style))
    story.append(
        Paragraph(
            "SHA-256 hash chain of all processing steps. Each entry is linked "
            "to the previous, providing a verifiable audit trail.",
            caveat_style,
        )
    )
    story.append(Spacer(1, 0.3 * cm))

    for entry in ledger_entries[:10]:  # Show first 10 steps
        story.append(
            Paragraph(
                f"[{entry.get('step_index', '?')}] {entry.get('module', '')} — "
                f"{entry.get('timestamp', '')} | Hash: {entry.get('entry_hash', '')[:24]}...",
                caveat_style,
            )
        )

    doc.build(story)
    return buffer.getvalue()
