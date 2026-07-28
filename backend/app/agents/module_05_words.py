"""
Module 05 — Word Patterns (LangGraph node).
Semantic search over ChromaDB for grooming/coercive language.
Uses a multilingual sentence-transformer so code-mixed Hinglish/Manglish/
Tanglish phrases are caught alongside English.

Why embeddings not keywords: "our little secret" and "don't tell anyone"
share no words but mean the same thing — embeddings catch this, keywords miss it.
"""

from __future__ import annotations

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


def embed_messages(messages: list[dict]) -> None:
    """
    Chunk and embed all messages into ChromaDB.
    Call this after ingestion, before running Module 05.
    """
    col = get_collection()
    docs, ids, metas = [], [], []
    for msg in messages:
        if msg.get("content"):
            docs.append(msg["content"])
            ids.append(msg["message_id"])
            metas.append({
                "sender_id": msg.get("sender_id") or "",
                "receiver_id": msg.get("receiver_id") or "",
                "platform": msg.get("platform", "unknown"),
                "timestamp": str(msg.get("timestamp") or ""),
            })

    if docs:
        col.upsert(documents=docs, ids=ids, metadatas=metas)


def word_patterns(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: run semantic search for each grooming query and return
    matches above the similarity threshold.

    State outputs:
        coercive_matches: list of match dicts with keys
            query, message_id, content, sender_id, similarity
    """
    col = get_collection()
    coercive_matches: list[dict] = []

    for query in GROOMING_QUERIES:
        try:
            results = col.query(
                query_texts=[query],
                n_results=5,
                include=["documents", "metadatas", "distances"],
            )
            if not results["ids"] or not results["ids"][0]:
                continue

            for idx, msg_id in enumerate(results["ids"][0]):
                # ChromaDB distance is L2; convert to rough cosine similarity
                distance = results["distances"][0][idx]
                similarity = 1.0 - min(distance / 2.0, 1.0)

                if similarity >= _COERCIVE_THRESHOLD:
                    coercive_matches.append({
                        "query": query,
                        "message_id": msg_id,
                        "content": results["documents"][0][idx],
                        "sender_id": results["metadatas"][0][idx].get("sender_id"),
                        "similarity": round(similarity, 4),
                    })
        except Exception as exc:
            print(f"[WARN] Word Patterns query failed: {exc}")

    state["coercive_matches"] = coercive_matches
    return state
