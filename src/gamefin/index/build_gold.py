"""Gold layer: silver chunks -> embeddings -> Qdrant vector index.

THE LINK TO SILVER: reads data/silver/chunks.jsonl, embeds each chunk's text
with the local bge model, and upserts the vector + full lineage payload into
Qdrant. After this runs, the corpus is searchable by meaning.
"""
from __future__ import annotations

import json
from pathlib import Path

from gamefin.index.embedder import Embedder
from gamefin.index.vector_store import VectorStore
from gamefin.logging_conf import get_logger
from gamefin.pipeline_config import GoldConfig, load_pipeline_config

log = get_logger(__name__)


def _expand_oversized(window: list[dict], embedder: Embedder) -> tuple[list[dict], int]:
    """Re-split any chunk that still exceeds the model's token limit.

    See Embedder.split_oversized for why this is necessary even after
    silver's char-based chunking. Each oversized row is replaced by N rows
    that share its lineage/section/etc. but carry a suffixed chunk_id, so a
    cited source still traces back to the exact filing and section.
    """
    expanded: list[dict] = []
    n_split = 0
    for row in window:
        pieces = embedder.split_oversized(row["text"])
        if len(pieces) == 1:
            expanded.append(row)
            continue
        n_split += 1
        for i, piece in enumerate(pieces):
            expanded.append({**row, "text": piece, "chunk_id": f"{row['chunk_id']}_t{i:02d}", "chars": len(piece)})
    return expanded, n_split


def _read_chunks(silver_dir: Path) -> list[dict]:
    path = silver_dir / "chunks.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Silver chunks not found at {path}. Run Phase 2 silver first: "
            "python scripts/run_silver.py"
        )
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_gold(gold: GoldConfig, silver_dir: Path) -> int:
    """Embed all silver chunks and load them into Qdrant. Returns count."""
    chunks = _read_chunks(silver_dir)
    log.info("Loaded %d chunks from silver", len(chunks))

    embedder = Embedder(
        model_name=gold.embedding_model,
        query_instruction=gold.query_instruction,
        batch_size=gold.embed_batch_size,
    )
    store = VectorStore(
        url=gold.qdrant_url,
        collection=gold.collection,
        dim=gold.embedding_dim,
        distance=gold.distance,
    )
    store.recreate_collection()

    # Process in batches to bound memory.
    batch = gold.embed_batch_size
    total_split = 0
    for start in range(0, len(chunks), batch):
        window = chunks[start : start + batch]
        window, n_split = _expand_oversized(window, embedder)
        total_split += n_split

        # Fail loudly rather than let sentence-transformers silently
        # truncate: after the split above, every text handed to the
        # embedder must fit, or something is wrong with split_oversized
        # itself and the resulting vector would quietly ignore part of the
        # chunk forever.
        for c in window:
            n_tok = embedder.count_tokens(c["text"])
            if n_tok > embedder.max_seq_length:
                raise RuntimeError(
                    f"Chunk {c['chunk_id']} is {n_tok} tokens after splitting "
                    f"(limit {embedder.max_seq_length}) — split_oversized did not "
                    "converge; fix before indexing silently truncates data."
                )

        vectors = embedder.embed_documents([c["text"] for c in window])
        store.upsert(vectors=vectors, payloads=window)
        log.info("Indexed %d / %d", min(start + batch, len(chunks)), len(chunks))

    if total_split:
        log.info("Token-split %d oversized chunk(s) before embedding", total_split)

    total = store.count()
    log.info("Gold complete: %d vectors in collection '%s'", total, gold.collection)
    return total


def main() -> None:
    cfg = load_pipeline_config()
    build_gold(cfg.gold, cfg.silver.silver_dir)


if __name__ == "__main__":
    main()
