"""Silver layer: bronze filings -> clean, section-labelled, chunked text.

THE LINK TO PHASE 1: this reads data/bronze/manifest.jsonl (Phase 1's output).
For each filing it opens the raw HTML we downloaded, parses it into sections,
chunks each section, and writes one row per chunk to data/silver/chunks.jsonl.

Crucially, every chunk row carries the SAME lineage fields as its bronze
filing (company, ticker, form, accession, dates, url) PLUS its section label
and position. That is how a future answer traces back to an exact filing and
section — the governance thread running through the whole system.
"""
from __future__ import annotations

import json
from pathlib import Path

from gamefin.config import REPO_ROOT
from gamefin.logging_conf import get_logger
from gamefin.parse.chunker import chunk_text
from gamefin.parse.html_parser import parse_filing
from gamefin.pipeline_config import SilverConfig, load_pipeline_config

log = get_logger(__name__)

BRONZE_MANIFEST = REPO_ROOT / "data" / "bronze" / "manifest.jsonl"


def _read_bronze_manifest(path: Path | None = None) -> list[dict]:
    # Resolve at call time (not definition time) so tests/config can override
    # the module-level BRONZE_MANIFEST.
    path = path or BRONZE_MANIFEST
    if not path.exists():
        raise FileNotFoundError(
            f"Bronze manifest not found at {path}. Run Phase 1 first: "
            "python scripts/run_ingest.py"
        )
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_silver(cfg: SilverConfig) -> list[dict]:
    """Parse + chunk every bronze filing into silver chunk rows."""
    filings = _read_bronze_manifest()
    cfg.silver_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.silver_dir / "chunks.jsonl"

    rows: list[dict] = []
    for f in filings:
        # local_path in the manifest is relative to the repo root.
        raw_path = REPO_ROOT / f["local_path"]
        if not raw_path.exists():
            log.warning("Missing raw file, skipping: %s", raw_path)
            continue

        html = raw_path.read_bytes()
        sections = parse_filing(html)

        seq = 0  # running chunk index within this filing
        n_before = len(rows)
        for section in sections:
            chunks = chunk_text(
                section.text,
                chunk_chars=cfg.chunk_chars,
                overlap_chars=cfg.overlap_chars,
                min_chunk_chars=cfg.min_chunk_chars,
            )
            for local_i, chunk in enumerate(chunks):
                rows.append(
                    {
                        # --- lineage carried from bronze (Phase 1) ---
                        "company": f["company"],
                        "ticker": f["ticker"],
                        "cik": f["cik"],
                        "form": f["form"],
                        "accession": f["accession"],
                        "filing_date": f["filing_date"],
                        "report_date": f["report_date"],
                        "url": f["url"],
                        # --- new silver-layer fields ---
                        "chunk_id": f"{f['ticker']}_{f['accession'].replace('-', '')}_{seq:04d}",
                        "section": section.item,
                        "section_chunk_index": local_i,
                        "chunk_index": seq,
                        "text": chunk,
                        "chars": len(chunk),
                    }
                )
                seq += 1
        log.info("%s %s (%s): %d chunks", f["ticker"], f["form"], f["report_date"], len(rows) - n_before)

    with open(out_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    log.info("Silver complete: %d chunks from %d filings -> %s", len(rows), len(filings), out_path)
    return rows


def main() -> None:
    cfg = load_pipeline_config().silver
    build_silver(cfg)


if __name__ == "__main__":
    main()
