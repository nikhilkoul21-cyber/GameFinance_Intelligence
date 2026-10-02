#!/usr/bin/env python
"""CLI: build the gold layer (embed silver chunks + load into Qdrant).

Prerequisites:
  1. Qdrant running:  docker compose up -d
  2. Silver built:    python scripts/run_silver.py

Usage:
    python scripts/run_gold.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gamefin.index.build_gold import build_gold  # noqa: E402
from gamefin.pipeline_config import load_pipeline_config  # noqa: E402


def main() -> int:
    cfg = load_pipeline_config()
    total = build_gold(cfg.gold, cfg.silver.silver_dir)
    print(f"\nDone. {total} vectors indexed in Qdrant collection '{cfg.gold.collection}'.")
    print("Next: ask a question ->  python scripts/ask.py \"What are Take-Two's key risk factors?\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
