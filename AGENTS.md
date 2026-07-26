# SENTINEL — Agent Instructions

## Project
AI-powered investigation support platform for child-protection digital
forensics. FastAPI + LangGraph backend, React + D3 frontend, Neo4j Aura
graph, ChromaDB vector search, Claude API for reasoning (synthetic data only).

## Non-negotiable conventions
- Normalize every timestamp to UTC immediately after parsing.
- Never drop an invalid or incomplete record — quarantine and keep it.
- Anomaly detection must be baseline-relative, never "flag the extreme value."
- No real evidence data anywhere in this repo or the deployed demo.

## Style
Python: type hints everywhere, Pydantic v2 for all models. Commit messages
reflect the actual build step. Every new module gets a test before it's done.
