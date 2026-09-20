"""Typed configuration loading for GameFin Intelligence.

Two sources, kept deliberately separate:
  * config/ingestion.yaml  — WHAT to ingest (companies, forms, scope). Safe to commit.
  * .env                   — SECRETS and machine-specific values. Never committed.

Keeping "what" in version control and "secrets" out of it is a basic
governance control: the scope of a run is auditable in git history, while
credentials never leak into the repo.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv


# Repo root = three parents up from this file (src/gamefin/config.py -> repo/).
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Company:
    name: str
    ticker: str
    cik: int
    primary: bool = False

    @property
    def cik_padded(self) -> str:
        """EDGAR requires the CIK zero-padded to 10 digits in API paths."""
        return f"{self.cik:010d}"


@dataclass(frozen=True)
class IngestionConfig:
    companies: list[Company]
    forms: list[str]
    since_date: str
    max_per_form: int
    rate_limit_per_sec: float
    bronze_dir: Path
    user_agent: str

    # Convenience: only the companies flagged primary (Take-Two).
    @property
    def primary_companies(self) -> list[Company]:
        return [c for c in self.companies if c.primary] or self.companies


def load_ingestion_config(
    config_path: str | Path = "config/ingestion.yaml",
) -> IngestionConfig:
    """Load and validate the ingestion config from YAML + .env.

    Raises a clear error if the SEC User-Agent is missing, because EDGAR
    will reject anonymous automated requests and the failure is otherwise
    cryptic (403s).
    """
    load_dotenv(REPO_ROOT / ".env")

    path = Path(config_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    user_agent = os.getenv("SEC_USER_AGENT", "").strip()
    if not user_agent or "example.com" in user_agent:
        raise RuntimeError(
            "SEC_USER_AGENT is not set (or still the placeholder). "
            "Copy .env.example to .env and set a real value like "
            "'GameFin-Intelligence you@yourdomain.com'. The SEC blocks "
            "requests without a descriptive User-Agent."
        )

    companies = [
        Company(
            name=c["name"],
            ticker=c["ticker"],
            cik=int(c["cik"]),
            primary=bool(c.get("primary", False)),
        )
        for c in raw["companies"]
    ]

    bronze_dir = Path(raw.get("bronze_dir", "data/bronze"))
    if not bronze_dir.is_absolute():
        bronze_dir = REPO_ROOT / bronze_dir

    return IngestionConfig(
        companies=companies,
        forms=list(raw["forms"]),
        since_date=str(raw["since_date"]),
        max_per_form=int(raw.get("max_per_form", 6)),
        rate_limit_per_sec=float(raw.get("rate_limit_per_sec", 5)),
        bronze_dir=bronze_dir,
        user_agent=user_agent,
    )
