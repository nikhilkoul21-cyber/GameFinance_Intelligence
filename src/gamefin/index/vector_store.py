"""A thin wrapper around Qdrant, our vector database.

A vector database stores vectors + their metadata and answers the question
"which stored vectors are most similar to this query vector?" extremely fast,
even over millions of items. Qdrant runs as a local Docker container (see
docker-compose.yml); this class hides its API behind three simple methods:
recreate_collection, upsert, and search.

Each stored point = one chunk: an id, its embedding vector, and a payload
(the chunk text + all its lineage metadata). The payload is what lets a
search result cite its exact source filing and section.
"""
from __future__ import annotations

import uuid

from gamefin.logging_conf import get_logger

log = get_logger(__name__)


class VectorStore:
    def __init__(self, url: str, collection: str, dim: int, distance: str = "Cosine"):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance

        self._client = QdrantClient(url=url)
        self._collection = collection
        self._dim = dim
        self._distance = getattr(Distance, distance.upper())

    def recreate_collection(self) -> None:
        """Create the collection fresh (drops it first if it exists).

        Used at the start of an indexing run so re-indexing is clean and
        idempotent — you always get exactly what's in the current silver file.
        """
        from qdrant_client.models import VectorParams

        self._client.recreate_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=self._dim, distance=self._distance),
        )
        log.info("Recreated Qdrant collection '%s' (dim=%d, %s)", self._collection, self._dim, self._distance.name)

    def upsert(self, vectors: list[list[float]], payloads: list[dict]) -> None:
        """Insert a batch of chunk vectors + their metadata payloads."""
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                # Deterministic id from the chunk_id so re-runs overwrite cleanly.
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, p["chunk_id"])),
                vector=vec,
                payload=p,
            )
            for vec, p in zip(vectors, payloads)
        ]
        self._client.upsert(collection_name=self._collection, points=points)

    def search(self, query_vector: list[float], top_k: int = 6, flt: dict | None = None):
        """Return the top_k most similar chunks, optionally filtered.

        `flt` is a simple {field: value} exact-match filter, e.g.
        {"ticker": "TTWO"} to restrict the search to one company.
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        qdrant_filter = None
        if flt:
            qdrant_filter = Filter(
                must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in flt.items()]
            )

        # query_points replaced the old .search() (removed in qdrant-client 1.13+).
        return self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        ).points

    def count(self) -> int:
        return self._client.count(collection_name=self._collection).count
