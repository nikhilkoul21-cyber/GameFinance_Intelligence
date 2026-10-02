"""Typed loading of config/pipeline.yaml (Phase 2 settings).

Kept separate from config.py (Phase 1's ingestion config) so each phase's
configuration is independent and easy to reason about. Reuses REPO_ROOT so
paths resolve no matter where you run from.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

from gamefin.config import REPO_ROOT


@dataclass(frozen=True)
class SilverConfig:
    chunk_chars: int
    overlap_chars: int
    min_chunk_chars: int
    silver_dir: Path


@dataclass(frozen=True)
class GoldConfig:
    embedding_model: str
    embedding_dim: int
    query_instruction: str
    embed_batch_size: int
    qdrant_url: str
    collection: str
    distance: str


@dataclass(frozen=True)
class ServeConfig:
    top_k: int
    min_score: float
    llm_provider: str
    llm_model: str
    temperature: float
    max_tokens: int


@dataclass(frozen=True)
class PipelineConfig:
    silver: SilverConfig
    gold: GoldConfig
    serve: ServeConfig
    companies: dict[str, list[str]]


def load_pipeline_config(
    config_path: str | Path = "config/pipeline.yaml",
) -> PipelineConfig:
    load_dotenv(REPO_ROOT / ".env")

    path = Path(config_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    s = raw["silver"]
    silver_dir = Path(s.get("silver_dir", "data/silver"))
    if not silver_dir.is_absolute():
        silver_dir = REPO_ROOT / silver_dir
    silver = SilverConfig(
        chunk_chars=int(s["chunk_chars"]),
        overlap_chars=int(s["overlap_chars"]),
        min_chunk_chars=int(s["min_chunk_chars"]),
        silver_dir=silver_dir,
    )

    g = raw["gold"]
    qdrant_url = os.getenv(g.get("qdrant_url_env", "QDRANT_URL"), "http://localhost:6333")
    gold = GoldConfig(
        embedding_model=g["embedding_model"],
        embedding_dim=int(g["embedding_dim"]),
        query_instruction=g.get("query_instruction", ""),
        embed_batch_size=int(g.get("embed_batch_size", 32)),
        qdrant_url=qdrant_url,
        collection=g["collection"],
        distance=g.get("distance", "Cosine"),
    )

    v = raw["serve"]
    serve = ServeConfig(
        top_k=int(v["top_k"]),
        min_score=float(v["min_score"]),
        llm_provider=v.get("llm_provider", "openai"),
        llm_model=v.get("llm_model", "gpt-4o-mini"),
        temperature=float(v.get("temperature", 0.0)),
        max_tokens=int(v.get("max_tokens", 700)),
    )

    companies = {ticker: list(aliases) for ticker, aliases in raw.get("companies", {}).items()}

    return PipelineConfig(silver=silver, gold=gold, serve=serve, companies=companies)
