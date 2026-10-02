#!/usr/bin/env python
"""CLI: build the silver layer (parse + chunk bronze filings).

Usage:
    python scripts/run_silver.py

Prerequisite: Phase 1 must have run (data/bronze/manifest.jsonl exists).
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gamefin.parse.build_silver import build_silver  # noqa: E402
from gamefin.pipeline_config import load_pipeline_config  # noqa: E402


def main() -> int:
    cfg = load_pipeline_config().silver
    rows = build_silver(cfg)

    print(f"\nDone. {len(rows)} chunks written to {cfg.silver_dir / 'chunks.jsonl'}")
    by_ticker = Counter(r["ticker"] for r in rows)
    for t in sorted(by_ticker):
        print(f"  {t:6s} {by_ticker[t]:5d} chunks")
    # Show the richest sections so you can eyeball quality.
    top_sections = Counter(r["section"] for r in rows).most_common(6)
    print("\nTop sections by chunk count:")
    for name, n in top_sections:
        print(f"  {n:5d}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
