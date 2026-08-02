"""
SENTINEL — FastAPI application root.
Registers all routers and wires startup/shutdown lifecycle.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# The project .env is the explicit backend configuration.  Override inherited
# shell values so a rotated key is not silently shadowed by a stale process env.
load_dotenv(override=True)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.ingestion import router as ingestion_router
from app.api.pipeline import router as pipeline_router
from app.api.dashboard import router as dashboard_router
from app.api.export import router as export_router
from app.api.hitl import router as hitl_router
from app.services.graph_db import close_driver
from app.agents.module_05_words import preload_collection, semantic_embeddings_enabled

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: startup → yield → shutdown."""
    # Nothing to init eagerly — Neo4j driver is lazy
    logger.info("Allowed CORS origins:\n- %s", "\n- ".join(allowed_origins))
    if semantic_embeddings_enabled():
        try:
            await asyncio.to_thread(preload_collection)
            logger.info("Semantic embedding model preloaded.")
        except Exception:
            logger.exception("Semantic embedding preload failed; using rule fallback.")
    yield
    close_driver()


app = FastAPI(
    title="SENTINEL",
    description=(
        "AI-powered investigation support platform for child-protection digital forensics. "
        "MVP demo — synthetic data only."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────
allowed_origins = [
    origin.strip()
    for origin in os.getenv("FRONTEND_URL", "http://localhost:5173").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────
app.include_router(ingestion_router)
app.include_router(pipeline_router)
app.include_router(dashboard_router)
app.include_router(export_router)
app.include_router(hitl_router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "SENTINEL"}
