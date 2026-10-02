# Architecture notes

Short, living companion to the full **Solution Scope** document. Records the
*why* behind the build so it survives interview questioning.

## Why medallion (bronze / silver / gold)?

A staged data flow, borrowed from lakehouse practice:

- **Bronze** — raw, immutable, exactly as fetched. If parsing logic changes
  we re-derive everything downstream *without re-hitting the source*. Bronze
  is also our audit copy.
- **Silver** — cleaned, section-labelled, chunked text. Deterministic from
  bronze.
- **Gold** — embeddings + vector index, ready to serve.

Each layer is reproducible from the one above it, which is what makes runs
auditable and cheap to iterate.

## Why lineage on every chunk?

Every bronze row (and every downstream chunk) carries company, form, period,
accession, source URL and a content `sha256`. A generated answer must resolve
back to these. This is the same control as tracing a regulatory figure to its
system of record — accuracy, completeness, traceability — applied to
unstructured text.

## Locked stack decisions

| # | Decision | Choice | Why |
|---|----------|--------|-----|
| D1 | Embeddings | local `BAAI/bge-base-en-v1.5` | data residency (text never leaves your infra); £0; swappable to a managed API via config |
| D2 | Vector DB | Qdrant (self-hosted, Docker) | real production system; separates vector workload from transactional DB; scales without code change |
| D3 | Generation LLM | provider-agnostic; default OpenAI, swap Anthropic/Azure | avoid vendor lock-in; the abstraction is the senior move |

## Discovery-driven ingestion

Which filings to pull is derived at runtime from the EDGAR submissions API
and filtered by `config/ingestion.yaml` (forms, date range, caps). We do not
hardcode accession numbers — a real pipeline re-discovers the current filing
set every run. Verified reference accessions are kept in the config only so a
run's output can be eyeballed for correctness.

## Compliance posture

- Descriptive `User-Agent` (SEC requirement).
- Self-throttled below the SEC's 10 req/s limit (default 5).
- Retry-with-backoff on transient errors.
- Idempotent downloads (re-runs skip files already stored).
