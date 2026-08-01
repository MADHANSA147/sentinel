"""
Module 05 — Word Patterns (LangGraph node).
Semantic search over ChromaDB for grooming/coercive language.
Uses a multilingual sentence-transformer so code-mixed Hinglish/Manglish/
Tanglish phrases are caught alongside English.

Why embeddings not keywords: "our little secret" and "don't tell anyone"
share no words but mean the same thing — embeddings catch this, keywords miss it.
"""

from __future__ import annotations

import re
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

# Multilingual model — catches English and code-mixed South Asian languages
_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_COLLECTION_NAME = "sentinel_messages"
_COERCIVE_THRESHOLD = 0.75  # cosine similarity threshold

# Grooming / coercive language seed queries (half English, half code-mixed)
GROOMING_QUERIES = [
    # English
    "don't tell anyone about this",
    "this is our little secret",
    "delete the messages after reading",
    "meet me alone without telling others",
    # Hinglish (Hindi-English mix)
    "yeh sirf hamare beech mein rehna chahiye",   # "this should stay between us"
    "kisi ko mat batana",                           # "don't tell anyone"
    # Manglish (Malayalam-English mix)
    "ithoru rahasyam aanu, parayande",              # "this is a secret, don't say"
    "ningal enikkoppam varanam, oru parayathe",     # "you must come with me, tell no one"
    # Tanglish (Tamil-English mix)
    "intha vishayam யாரிடமும் சொல்லாதே",          # "don't tell this to anyone"
    "நம்ம ரெண்டு பேர் மட்டும் தெரியணும்",          # "only we two should know"
]

# An embedding is a candidate generator, not evidence by itself.  The final
# tag needs an explicit secrecy/isolation instruction in the message.  This
# deliberately excludes ordinary operational phrases such as "don't share"
# or "delete this conversation", which caused Dataset 1 false positives.
_EXPLICIT_COERCIVE_PATTERNS = (
    r"\bdon['’]?t\s+tell\s+(?:anyone|anybody)\b",
    r"\b(?:our|this is a)\s+(?:little\s+)?secret\b",
    r"\bkeep\s+(?:this|it)\s+(?:a\s+)?secret\b",
    r"\bmeet\s+me\s+alone\b",
    r"\b(?:delete|erase)\s+(?:the\s+)?messages?\s+after\s+reading\b",
)


def _has_explicit_coercive_language(content: str) -> bool:
    """Require a reviewable, direct coercive-language match before tagging."""
    normalized = " ".join(content.casefold().split())
    return any(re.search(pattern, normalized) for pattern in _EXPLICIT_COERCIVE_PATTERNS)


def _get_chroma_collection() -> chromadb.Collection:
    """Return (or create) the ChromaDB message collection."""
    client = chromadb.Client()  # in-memory for MVP; swap for PersistentClient for prod
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=_MODEL_NAME
    )
    return client.get_or_create_collection(
        name=_COLLECTION_NAME, embedding_function=ef
    )


_chroma_collection: chromadb.Collection | None = None


def get_collection() -> chromadb.Collection:
    global _chroma_collection
    if _chroma_collection is None:
        _chroma_collection = _get_chroma_collection()
    return _chroma_collection


def embed_messages(messages: list[dict], case_id: str = "default") -> None:
    """
    Chunk and embed all messages into ChromaDB.
    Call this after ingestion, before running Module 05.
    """
    try:
        col = get_collection()
    except Exception as exc:
        print(f"[WARN] Word Patterns embedding unavailable: {exc}")
        return

    docs, ids, metas = [], [], []
    for msg in messages:
        if msg.get("content") and not msg.get("is_quarantined", False):
            message_id = str(msg["message_id"])
            scoped_case_id = str(msg.get("case_id") or case_id)
            docs.append(msg["content"])
            ids.append(f"{scoped_case_id}:{message_id}")
            metas.append({
                "case_id": scoped_case_id,
                "message_id": message_id,
                "sender_id": msg.get("sender_id") or "",
                "receiver_id": msg.get("receiver_id") or "",
                "platform": msg.get("platform", "unknown"),
                "timestamp": str(msg.get("timestamp") or ""),
            })

    if docs:
        try:
            col.upsert(documents=docs, ids=ids, metadatas=metas)
        except Exception as exc:
            print(f"[WARN] Word Patterns embedding write failed: {exc}")


def word_patterns(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: run semantic search for each grooming query and return
    matches above the similarity threshold.

    State outputs:
        coercive_matches: list of match dicts with keys
            query, message_id, content, sender_id, similarity
    """
    try:
        col = get_collection()
    except Exception as exc:
        print(f"[WARN] Word Patterns unavailable: {exc}")
        state["coercive_matches"] = []
        return state

    case_id = str(state.get("case_id", "default"))
    coercive_matches: list[dict] = []

    for query in GROOMING_QUERIES:
        try:
            results = col.query(
                query_texts=[query],
                n_results=5,
                include=["documents", "metadatas", "distances"],
                where={"case_id": case_id},
            )
            if not results["ids"] or not results["ids"][0]:
                continue

            for idx, msg_id in enumerate(results["ids"][0]):
                # ChromaDB distance is L2; convert to rough cosine similarity
                distance = results["distances"][0][idx]
                similarity = 1.0 - min(distance / 2.0, 1.0)

                content = results["documents"][0][idx]
                if (
                    similarity >= _COERCIVE_THRESHOLD
                    and _has_explicit_coercive_language(content)
                ):
                    metadata = results["metadatas"][0][idx]
                    coercive_matches.append({
                        "query": query,
                        "message_id": (
                            metadata.get("message_id")
                            or str(msg_id).split(":", 1)[-1]
                        ),
                        "content": content,
                        "sender_id": metadata.get("sender_id"),
                        "similarity": round(similarity, 4),
                    })
        except Exception as exc:
            print(f"[WARN] Word Patterns query failed: {exc}")

    state["coercive_matches"] = coercive_matches
    return state
