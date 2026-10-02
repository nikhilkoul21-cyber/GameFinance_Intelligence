"""Unit tests for the silver layer (parsing + chunking).

These are pure-logic tests: no network, no models, no Qdrant. They prove the
parser finds sections and the chunker respects size/overlap/minimum rules —
the parts most likely to hide bugs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gamefin.parse import build_silver as bs  # noqa: E402
from gamefin.parse.chunker import chunk_text  # noqa: E402
from gamefin.parse.html_parser import html_to_text, parse_filing, split_into_sections  # noqa: E402
from gamefin.pipeline_config import SilverConfig  # noqa: E402


SAMPLE_HTML = b"""
<html><head><style>.x{color:red}</style></head><body>
<p>TAKE-TWO INTERACTIVE SOFTWARE, INC. Annual Report</p>
<p>Item 1. Business</p>
<p>We develop and publish interactive entertainment. Our labels include
Rockstar Games and 2K.</p>
<p>Item 1A. Risk Factors</p>
<p>Our business depends on a small number of key franchises. If a major title
underperforms, our results could be materially harmed. We also face risks from
cybersecurity incidents and evolving privacy regulation.</p>
<p>Item 7. Management's Discussion and Analysis</p>
<p>Net bookings increased year over year driven by Grand Theft Auto and NBA 2K.</p>
</body></html>
"""


def test_html_to_text_strips_tags_and_styles():
    text = html_to_text(SAMPLE_HTML)
    assert "color:red" not in text          # style content removed
    assert "Risk Factors" in text
    assert "<p>" not in text                 # tags gone


def test_split_into_sections_finds_items():
    sections = parse_filing(SAMPLE_HTML)
    labels = [s.item for s in sections]
    # Should detect the three item headings (order preserved).
    assert any("Item 1A" in l and "Risk Factors" in l for l in labels)
    assert any("Item 7" in l for l in labels)
    # The risk section text lands under the risk label, not business.
    risk = next(s for s in sections if "1A" in s.item)
    assert "key franchises" in risk.text


def test_8k_without_items_becomes_one_section():
    html = b"<html><body><p>On this date, the company announced a new CEO.</p></body></html>"
    sections = parse_filing(html)
    assert len(sections) == 1
    assert sections[0].item in ("FULL_DOCUMENT", "PREAMBLE")


def test_chunk_respects_size_overlap_and_min():
    text = "A. " * 3000  # ~9000 chars, forces multiple chunks
    chunks = chunk_text(text, chunk_chars=2000, overlap_chars=200, min_chunk_chars=100)
    assert len(chunks) >= 4
    # No chunk wildly exceeds target + overlap.
    assert all(len(c) <= 2000 + 200 + 50 for c in chunks)
    # Overlap: the start of chunk 2 should share text with the end of chunk 1.
    assert chunks[0][-50:][:10] in chunks[1]


def test_tiny_fragments_dropped():
    chunks = chunk_text("short", chunk_chars=2000, overlap_chars=0, min_chunk_chars=100)
    assert chunks == []          # below minimum -> discarded


def test_build_silver_carries_lineage(tmp_path, monkeypatch):
    # Fake a bronze filing on disk + a manifest pointing at it.
    raw = tmp_path / "ttwo-10k.htm"
    raw.write_bytes(SAMPLE_HTML)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({
        "company": "Take-Two Interactive", "ticker": "TTWO", "cik": 946581,
        "form": "10-K", "accession": "0001628280-25-026694",
        "filing_date": "2025-05-15", "report_date": "2025-03-31",
        "url": "https://sec.gov/x", "local_path": str(raw),
    }) + "\n")

    # Point the module at our temp manifest and repo root.
    monkeypatch.setattr(bs, "BRONZE_MANIFEST", manifest)
    monkeypatch.setattr(bs, "REPO_ROOT", tmp_path)

    cfg = SilverConfig(chunk_chars=500, overlap_chars=50, min_chunk_chars=20,
                       silver_dir=tmp_path / "silver")
    rows = bs.build_silver(cfg)

    assert len(rows) > 0
    r = rows[0]
    # Lineage from bronze is present on every chunk.
    for field in ("company", "ticker", "form", "accession", "report_date", "url",
                  "chunk_id", "section", "chunk_index", "text"):
        assert field in r
    assert r["ticker"] == "TTWO"
    assert r["chunk_id"].startswith("TTWO_000162828025026694_")
    # The silver file was written and is valid JSONL.
    out = (tmp_path / "silver" / "chunks.jsonl").read_text().splitlines()
    assert len(out) == len(rows)
    assert json.loads(out[0])["ticker"] == "TTWO"
