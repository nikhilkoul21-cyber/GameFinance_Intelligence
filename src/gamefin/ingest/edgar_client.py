"""A small, well-behaved client for the SEC EDGAR REST API.

Design goals (these are the things an interviewer will poke at):
  * Compliant: sends the required User-Agent, self-throttles below the
    SEC's 10 req/s limit, retries transient errors with backoff.
  * Discovery-driven: which filings to download is derived at runtime from
    the submissions API, not hardcoded.
  * Traceable: every filing dict carries the metadata (company, form, date,
    accession, source URL) that becomes lineage on every downstream chunk.

EDGAR facts this relies on:
  * Submissions API: https://data.sec.gov/submissions/CIK##########.json
    -> `filings.recent` holds PARALLEL arrays (accessionNumber[i], form[i],
       filingDate[i], reportDate[i], primaryDocument[i], ...).
  * Document URL: https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/{primaryDocument}
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from gamefin.logging_conf import get_logger

log = get_logger(__name__)

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik_padded}.json"
ARCHIVE_URL = (
    "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"
)


@dataclass(frozen=True)
class Filing:
    """One filing's metadata — the unit of lineage we carry everywhere."""
    company: str
    ticker: str
    cik: int
    form: str
    accession: str      # with dashes, e.g. 0001628280-25-026694
    filing_date: str    # YYYY-MM-DD (date submitted to SEC)
    report_date: str    # YYYY-MM-DD (period the filing covers)
    primary_document: str
    url: str

    @property
    def accession_nodash(self) -> str:
        return self.accession.replace("-", "")


class EdgarClient:
    """Rate-limited, retrying HTTP client scoped to EDGAR."""

    def __init__(self, user_agent: str, rate_limit_per_sec: float = 5.0):
        if not user_agent:
            raise ValueError("EDGAR requires a non-empty User-Agent.")
        self._min_interval = 1.0 / max(rate_limit_per_sec, 0.1)
        self._last_request_ts = 0.0
        self._session = requests.Session()
        self._session.headers.update(
            {
                # SEC asks for a descriptive UA (name + contact). Also send
                # standard headers so the archive server serves us normally.
                "User-Agent": user_agent,
                "Accept-Encoding": "gzip, deflate",
                "Host": "www.sec.gov",
            }
        )

    # --- low-level, throttled GET -----------------------------------------
    def _throttle(self) -> None:
        """Sleep just enough to stay under the configured request rate."""
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_ts = time.monotonic()

    @retry(
        retry=retry_if_exception_type(requests.RequestException),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        reraise=True,
    )
    def _get(self, url: str, host: str) -> requests.Response:
        self._throttle()
        # The Host header must match the URL's host (data.sec.gov vs www.sec.gov).
        headers = {"Host": host}
        resp = self._session.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp

    # --- API methods -------------------------------------------------------
    def get_submissions(self, cik: int) -> dict:
        """Fetch the full submissions record for a company (CIK)."""
        url = SUBMISSIONS_URL.format(cik_padded=f"{cik:010d}")
        log.info("Fetching submissions index: %s", url)
        return self._get(url, host="data.sec.gov").json()

    def discover_filings(
        self,
        company_name: str,
        ticker: str,
        cik: int,
        forms: list[str],
        since_date: str,
        max_per_form: int,
    ) -> list[Filing]:
        """Turn the submissions record into a filtered list of Filing objects.

        Iterates the parallel arrays in `filings.recent`, keeps only the
        requested form types filed on/after `since_date`, and caps each form
        type at `max_per_form` (most recent first).
        """
        data = self.get_submissions(cik)
        recent = data.get("filings", {}).get("recent", {})

        accession = recent.get("accessionNumber", [])
        form_arr = recent.get("form", [])
        fdate_arr = recent.get("filingDate", [])
        rdate_arr = recent.get("reportDate", [])
        pdoc_arr = recent.get("primaryDocument", [])

        wanted = {f.upper() for f in forms}
        counts: dict[str, int] = {f: 0 for f in wanted}
        out: list[Filing] = []

        # The arrays are already newest-first in EDGAR's response.
        for i in range(len(accession)):
            form = (form_arr[i] or "").upper()
            if form not in wanted:
                continue
            if fdate_arr[i] < since_date:
                continue
            if counts[form] >= max_per_form:
                continue
            doc = pdoc_arr[i] if i < len(pdoc_arr) else ""
            if not doc:
                # No primary document listed (rare) — skip; nothing to fetch.
                continue
            acc = accession[i]
            url = ARCHIVE_URL.format(
                cik=cik, acc_nodash=acc.replace("-", ""), doc=doc
            )
            out.append(
                Filing(
                    company=company_name,
                    ticker=ticker,
                    cik=cik,
                    form=form,
                    accession=acc,
                    filing_date=fdate_arr[i],
                    report_date=rdate_arr[i] if i < len(rdate_arr) else "",
                    primary_document=doc,
                    url=url,
                )
            )
            counts[form] += 1

        log.info(
            "%s: discovered %d filings (%s)",
            ticker,
            len(out),
            ", ".join(f"{k}={v}" for k, v in counts.items()),
        )
        return out

    def download_document(self, filing: Filing) -> bytes:
        """Fetch the raw primary document bytes for a filing."""
        log.info("Downloading %s %s -> %s", filing.ticker, filing.form, filing.url)
        return self._get(filing.url, host="www.sec.gov").content
