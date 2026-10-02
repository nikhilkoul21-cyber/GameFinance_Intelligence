#!/usr/bin/env python
"""CLI: ask the RAG system a question and get a cited answer.

Prerequisites: Qdrant running + gold index built + an LLM key in .env.

Usage:
    python scripts/ask.py "What are Take-Two's key risk factors?"
    python scripts/ask.py --ticker TTWO "How did net bookings change?"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gamefin.pipeline_config import load_pipeline_config  # noqa: E402
from gamefin.serve.rag import RagPipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask GameFin Intelligence a question")
    parser.add_argument("question", help="Your question, in quotes")
    parser.add_argument("--ticker", default=None, help="Restrict to one company, e.g. TTWO")
    args = parser.parse_args()

    cfg = load_pipeline_config()
    pipeline = RagPipeline(cfg)
    result = pipeline.ask(args.question, ticker=args.ticker)

    print("\n" + "=" * 70)
    print("Q:", result.question)
    print("=" * 70)
    if result.detected_ticker and not args.ticker:
        print(f"(auto-detected company: {result.detected_ticker} — pass --ticker to override)")
    print(result.answer)
    if result.sources:
        print("\nSources:")
        for s in result.sources:
            print(f"  [{s.n}] {s.company} {s.form} {s.report_date} — {s.section}")
            print(f"      score={s.score:.3f}  {s.url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
