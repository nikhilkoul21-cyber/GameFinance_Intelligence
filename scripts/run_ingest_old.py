#!/usr/bin/env python
"""CLI entrypoint for a bronze ingestion run.

Usage:
    python scripts/run_ingest.py
    python scripts/run_ingest.py --config config/ingestion.yaml

Run from the repo root with the venv active. Requires SEC_USER_AGENT in .env.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `src/` importable without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gamefin.config import load_ingestion_config  # noqa: E402
from gamefin.ingest.fetch_filings import run_ingestion  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="GameFin bronze ingestion")
    parser.add_argument(
        "--config",
        default="config/ingestion.yaml",
        help="Path to the ingestion YAML config.",
    )
    args = parser.parse_args()

    config = load_ingestion_config(args.config)
    rows = run_ingestion(config)

    print(f"\nDone. {len(rows)} filings in the manifest.")
    print(f"Raw docs:  {config.bronze_dir}")
    print(f"Manifest:  {config.bronze_dir / 'manifest.jsonl'}")
    # Small human-readable summary so you can eyeball the run.
    by_key: dict[str, int] = {}
    for r in rows:
        key = f"{r['ticker']} {r['form']}"
        by_key[key] = by_key.get(key, 0) + 1
    for key in sorted(by_key):
        print(f"  {key:20s} {by_key[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
