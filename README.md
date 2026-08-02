# SENTINEL

AI-powered investigation support platform for child-protection digital forensics.

**MVP demo — synthetic data only. No real evidence data in this repository.**

## Architecture

| Layer | Tech |
|-------|------|
| Backend API | FastAPI + Python 3.11 |
| Agent Orchestration | LangGraph (12 concurrent agents) |
| Graph Database | Neo4j Aura (free tier) |
| Vector Search | ChromaDB (embedded) |
| Centrality Algorithms | networkx (PageRank + Betweenness) |
| LLM Reasoning | Groq API (OpenAI-compatible; synthetic data only) |
| Frontend | React + D3.js (Vite) |
| PDF Export | reportlab |
| Backend Deploy | Render |
| Frontend Deploy | Vercel |

## Setup

### Backend

```powershell
cd backend
pip install -r requirements.txt
copy .env.example .env   # fill in your Neo4j + Groq API keys
uvicorn app.main:app --reload
```

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

### Run Tests

```powershell
cd backend
pytest tests/test_gap_detector.py -v
```

## Synthetic Datasets

| File | Records | Quarantined | Tests |
|------|---------|-------------|-------|
| whatsapp_conversation_40_messages.json | 40 | 0 | Gap Detector (6h gap MSG-0028→MSG-0029) |
| whatsapp_synthetic_35_messages.json | 35 | 0 | Exculpatory Context (night-shift suppression) |
| corrupted_whatsapp_30_messages.json | 30 | 9 (30%) | Ingestion resilience |
| network_test_45_messages.json | 45 | 0 | Network Mapping (U-304 bridge betweenness) |

## Agents (LangGraph Pipeline)

```
identity_fusion → timeline_engine → behavioral_print → network_mapping
→ subject_profiling → role_discovery → word_patterns → gap_detector
→ exculpatory_context → risk_score_engine → case_simulation → feedback_loop
```

## Legal Layer

- **Execution Ledger**: SHA-256 chained audit log of every agent step
- **Court Pack PDF**: Score-free exhibits only (Section 63, BSA 2023)
- **Taxonomy Mapping**: NCMEC grooming indicators + ISO/IEC 27037
- **Audit Bundle**: JSON export of ledger + graph snapshot

## Definition of Done

- [ ] Deployed link opens with zero login
- [ ] Full flow runs start-to-finish without manual restart
- [ ] Deployed app matches GitHub repo exactly
- [ ] Repo is public with commit history spanning build days
