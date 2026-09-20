"""Bronze layer: download raw filings and record lineage.

Output for a run:
  data/bronze/<TICKER>/<FORM>/<accession>__<primaryDocument>   (raw file)
  data/bronze/manifest.jsonl                                    (one line per doc)

The manifest is the point of this layer. Every downstream chunk, embedding
and answer traces back to a manifest row: which company, which filing, which
period, which URL, and a sha256 of exactly what we stored. That lineage is
what makes an answer auditable — the same discipline as tracing a regulatory
figure back to its system of record.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from gamefin.config import IngestionConfig, load_ingestion_config
from gamefin.ingest.edgar_client import EdgarClient, Filing
from gamefin.logging_conf import get_logger

log = get_logger(__name__)


def _safe(name: str) -> str:
    """Filesystem-safe fragment."""
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in name)


def _bronze_path(bronze_dir: Path, f: Filing) -> Path:
    return (
        bronze_dir
        / _safe(f.ticker)
        / _safe(f.form)
        / f"{f.accession}__{_safe(f.primary_document)}"
    )


def _already_have(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def run_ingestion(config: IngestionConfig) -> list[dict]:
    """Execute a full bronze ingestion run. Returns the manifest rows written."""
    client = EdgarClient(
        user_agent=config.user_agent,
        rate_limit_per_sec=config.rate_limit_per_sec,
    )
    config.bronze_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.bronze_dir / "manifest.jsonl"

    rows: list[dict] = []
    for company in config.companies:
        filings = client.discover_filings(
            company_name=company.name,
            ticker=company.ticker,
            cik=company.cik,
            forms=config.forms,
            since_date=config.since_date,
            max_per_form=config.max_per_form,
        )
        for f in filings:
            dest = _bronze_path(config.bronze_dir, f)
            dest.parent.mkdir(parents=True, exist_ok=True)

            if _already_have(dest):
                # Idempotent: re-running a completed download is a no-op.
                content = dest.read_bytes()
                downloaded = False
            else:
                content = client.download_document(f)
                dest.write_bytes(content)
                downloaded = True

            row = {
                **asdict(f),
                "local_path": str(dest.relative_to(config.bronze_dir.parent.parent))
                if config.bronze_dir.parent.parent in dest.parents
                else str(dest),
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
                "newly_downloaded": downloaded,
            }
            rows.append(row)

    # Rewrite the manifest fresh each run (it reflects current bronze state).
    with open(manifest_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    log.info(
        "Ingestion complete: %d filings across %d companies. Manifest: %s",
        len(rows),
        len(config.companies),
        manifest_path,
    )
    return rows


def main() -> None:
    config = load_ingestion_config()
    run_ingestion(config)


if __name__ == "__main__":
    main()
