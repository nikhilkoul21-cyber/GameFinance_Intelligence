"""Unit tests for the ingestion logic.

These run WITHOUT network access: we feed the client a fake submissions
record and a fake document downloader, then assert the discovery filter and
the manifest writer behave. This is exactly how you test an ingestion
pipeline in CI, where you can't (and shouldn't) hit the live SEC API.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Make src/ importable when running pytest from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gamefin.config import Company, IngestionConfig  # noqa: E402
from gamefin.ingest import edgar_client as ec  # noqa: E402
from gamefin.ingest import fetch_filings as ff  # noqa: E402


# A tiny fake of the EDGAR submissions `filings.recent` structure:
# parallel arrays, newest-first, mixing form types and dates.
FAKE_SUBMISSIONS = {
    "filings": {
        "recent": {
            "accessionNumber": [
                "0001628280-25-026694",  # 10-K, in range
                "0001628280-25-000111",  # 10-Q, in range
                "0001628280-24-000222",  # 8-K, in range
                "0001628280-20-000333",  # 10-K, OUT of range (too old)
                "0001628280-25-000444",  # DEF 14A, wrong form
            ],
            "form": ["10-K", "10-Q", "8-K", "10-K", "DEF 14A"],
            "filingDate": [
                "2025-05-15",
                "2025-08-01",
                "2024-11-10",
                "2020-05-20",
                "2025-07-01",
            ],
            "reportDate": [
                "2025-03-31",
                "2025-06-30",
                "2024-11-10",
                "2020-03-31",
                "2025-07-01",
            ],
            "primaryDocument": [
                "ttwo-20250331.htm",
                "ttwo-20250630.htm",
                "ttwo-8k.htm",
                "ttwo-20200331.htm",
                "ttwo-def14a.htm",
            ],
        }
    }
}


def test_discover_filters_by_form_and_date(monkeypatch):
    client = ec.EdgarClient(user_agent="test test@example.org")
    monkeypatch.setattr(client, "get_submissions", lambda cik: FAKE_SUBMISSIONS)

    filings = client.discover_filings(
        company_name="Take-Two Interactive",
        ticker="TTWO",
        cik=946581,
        forms=["10-K", "10-Q", "8-K"],
        since_date="2021-01-01",
        max_per_form=6,
    )

    # DEF 14A dropped (wrong form); 2020 10-K dropped (too old). 3 remain.
    forms = sorted(f.form for f in filings)
    assert forms == ["10-K", "10-Q", "8-K"]

    # URL is constructed correctly (accession de-dashed, CIK not padded here).
    tenk = next(f for f in filings if f.form == "10-K")
    assert tenk.accession == "0001628280-25-026694"
    assert tenk.accession_nodash == "000162828025026694"
    assert tenk.url == (
        "https://www.sec.gov/Archives/edgar/data/946581/"
        "000162828025026694/ttwo-20250331.htm"
    )


def test_max_per_form_cap(monkeypatch):
    client = ec.EdgarClient(user_agent="test test@example.org")
    monkeypatch.setattr(client, "get_submissions", lambda cik: FAKE_SUBMISSIONS)
    filings = client.discover_filings(
        "Take-Two", "TTWO", 946581,
        forms=["10-K"], since_date="2021-01-01", max_per_form=1,
    )
    assert len([f for f in filings if f.form == "10-K"]) == 1


def test_run_ingestion_writes_manifest(monkeypatch, tmp_path):
    # Mock discovery + download so no network is touched.
    fake = ec.Filing(
        company="Take-Two Interactive", ticker="TTWO", cik=946581,
        form="10-K", accession="0001628280-25-026694",
        filing_date="2025-05-15", report_date="2025-03-31",
        primary_document="ttwo-20250331.htm",
        url="https://www.sec.gov/Archives/edgar/data/946581/"
            "000162828025026694/ttwo-20250331.htm",
    )
    monkeypatch.setattr(
        ec.EdgarClient, "discover_filings",
        lambda self, **kwargs: [fake],
    )
    monkeypatch.setattr(
        ec.EdgarClient, "download_document",
        lambda self, filing: b"<html>fake 10-K body</html>",
    )

    config = IngestionConfig(
        companies=[Company("Take-Two Interactive", "TTWO", 946581, primary=True)],
        forms=["10-K"], since_date="2021-01-01", max_per_form=6,
        rate_limit_per_sec=5, bronze_dir=tmp_path / "bronze",
        user_agent="test test@example.org",
    )

    rows = ff.run_ingestion(config)

    assert len(rows) == 1
    row = rows[0]
    # Lineage fields present on every manifest row.
    for field in ("company", "ticker", "form", "accession", "report_date",
                  "url", "sha256", "bytes", "downloaded_at"):
        assert field in row and row[field] != ""

    # Raw file actually written, and manifest is valid JSONL.
    manifest = config.bronze_dir / "manifest.jsonl"
    assert manifest.exists()
    parsed = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert parsed[0]["accession"] == "0001628280-25-026694"
    assert parsed[0]["bytes"] == len(b"<html>fake 10-K body</html>")

    # Idempotency: a second run re-uses the file, doesn't re-download.
    rows2 = ff.run_ingestion(config)
    assert rows2[0]["newly_downloaded"] is False
