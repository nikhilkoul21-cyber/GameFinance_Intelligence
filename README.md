# GameFin Intelligence

A production-style **Retrieval-Augmented Generation (RAG)** system over
gaming-industry SEC filings (Take-Two Interactive and peers), built with an
enterprise **medallion architecture** (bronze → silver → gold), a real
**evaluation harness**, and trace-level observability.

> **Positioning:** This is an independent portfolio project built

---

## Status — Phase 1: Ingestion (bronze layer) ✅

The current build discovers and downloads real SEC filings and records full
lineage. Later phases add parsing/chunking (silver), embedding + Qdrant
(gold), retrieval + grounded generation, evaluation, and deployment.

| Phase | Scope | State |
|------|-------|-------|
| 1 | Repo + config + **bronze ingestion** from EDGAR | **done** |
| 2 | Silver (parse/chunk) + Gold (embed → Qdrant) + retrieval + generation | next |
| 3 | Golden dataset + eval harness (RAGAS + LLM-as-judge) + eval gate | planned |
| 4 | FastAPI + UI, observability, deploy to a public URL | planned |

---

## Architecture (target)

```
        SEC EDGAR (public REST API, no key)
                    │
   ┌────────────────▼─────────────────┐
   │ BRONZE  raw filings + manifest    │  ← Phase 1 (this build)
   │  data/bronze/<TICKER>/<FORM>/...  │
   │  data/bronze/manifest.jsonl       │  ← lineage: company/form/date/url/sha256
   └────────────────┬─────────────────┘
   ┌────────────────▼─────────────────┐
   │ SILVER  cleaned, section-labelled │  ← Phase 2
   │         chunks (Item 1A, MD&A…)   │
   └────────────────┬─────────────────┘
   ┌────────────────▼─────────────────┐
   │ GOLD    embeddings (bge-base) →   │  ← Phase 2
   │         Qdrant vector index       │
   └────────────────┬─────────────────┘
   ┌────────────────▼─────────────────┐
   │ SERVE   retrieve → cite-or-refuse │  ← Phase 2/4
   │         LLM → FastAPI + UI        │
   └────────────────┬─────────────────┘
   ┌────────────────▼─────────────────┐
   │ ASSURE  eval harness + tracing    │  ← Phase 3/4
   └───────────────────────────────────┘
```

**Locked stack decisions:** local `BAAI/bge-base-en-v1.5` embeddings (data
stays in-house), **Qdrant** self-hosted vector DB, provider-agnostic LLM
client (default OpenAI, swappable to Anthropic/Azure). Rationale is in
`docs/ARCHITECTURE.md` and the Solution Scope document.

---

## Quick start (Phase 1)

Prerequisites: **Python 3.10+** and **git**.

```bash
# 1. Enter the project
cd gamefin-intelligence

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install Phase 1 dependencies
pip install -r requirements.txt

# 4. Configure your SEC identity (REQUIRED — EDGAR blocks anonymous bots)
cp .env.example .env
#   then edit .env and set:
#   SEC_USER_AGENT="GameFin-Intelligence your.email@domain.com"

# 5. Run the ingestion
python scripts/run_ingest.py
```

You should see it discover and download real filings, then print a summary:

```
Done. 30 filings in the manifest.
Raw docs:  data/bronze
Manifest:  data/bronze/manifest.jsonl
  EA   10-K                3
  EA   10-Q                6
  ...
  TTWO 10-K                4
  TTWO 10-Q                6
  TTWO 8-K                 6
```

Inspect what you pulled:

```bash
# how many docs, and a peek at the lineage record
wc -l data/bronze/manifest.jsonl
head -n 1 data/bronze/manifest.jsonl | python -m json.tool
```

Run the tests (no network needed):

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

---

## Configuration

Edit **`config/ingestion.yaml`** to change scope — companies, form types,
date range, per-form caps, rate limit. The code discovers matching filings
live from EDGAR; nothing about *which* documents to pull is hardcoded.

Secrets and machine-specific values live in **`.env`** (never committed).

---

## Project layout

```
gamefin-intelligence/
├── config/ingestion.yaml         # WHAT to ingest (committed, auditable)
├── src/gamefin/
│   ├── config.py                 # typed config (YAML + .env)
│   ├── logging_conf.py
│   └── ingest/
│       ├── edgar_client.py       # rate-limited, retrying EDGAR client
│       └── fetch_filings.py      # bronze download + lineage manifest
├── scripts/run_ingest.py         # CLI entrypoint
├── tests/test_ingest.py          # unit tests (mocked, no network)
├── data/{bronze,silver,gold}/    # medallion layers (gitignored contents)
├── docs/ARCHITECTURE.md
├── requirements.txt
└── .env.example
```

## Data source & compliance

All data is from the U.S. SEC **EDGAR** system via its official REST API
(`data.sec.gov`). No API key is required, but the SEC requires a descriptive
`User-Agent` and asks callers to stay under 10 requests/second — this client
self-throttles (default 5 req/s) and retries transient failures with backoff.
SEC filings are U.S. government public-domain data.
