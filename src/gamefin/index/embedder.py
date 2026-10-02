"""Turn text into vectors with a local embedding model (BAAI/bge-base-en-v1.5).

An 'embedding' is a list of numbers (here 768 of them) that represents the
MEANING of a piece of text. Texts with similar meaning get similar vectors,
so we can find relevant chunks by comparing vectors instead of matching
keywords. Running the model locally means filing text never leaves your
machine — the data-residency choice from our architecture (decision D1).

Important bge detail: these models are trained so that SEARCH QUERIES get a
short instruction prefix, but STORED PASSAGES do not. We therefore expose two
methods — embed_documents (no prefix) and embed_query (prefixed) — and using
the right one for each side measurably improves retrieval.
"""
from __future__ import annotations

from gamefin.logging_conf import get_logger

log = get_logger(__name__)


class Embedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        query_instruction: str = "Represent this sentence for searching relevant passages: ",
        batch_size: int = 32,
    ):
        # Imported here (not at top) so the rest of the package doesn't require
        # the heavy sentence-transformers install unless you actually embed.
        from sentence_transformers import SentenceTransformer

        log.info("Loading embedding model: %s (first run downloads it)", model_name)
        self._model = SentenceTransformer(model_name)
        self._query_instruction = query_instruction
        self._batch_size = batch_size

    @property
    def dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    @property
    def max_seq_length(self) -> int:
        return int(self._model.max_seq_length)

    def count_tokens(self, text: str) -> int:
        return len(self._model.tokenizer(text, add_special_tokens=False)["input_ids"])

    def split_oversized(self, text: str) -> list[str]:
        """Re-split one chunk on real tokens if it exceeds max_seq_length.

        The silver-layer chunker (gamefin.parse.chunker) sizes chunks in
        characters, calibrated for prose. Dense financial tables tokenize at
        under half the chars-per-token ratio of prose (measured as low as
        2.4 chars/token vs. the ~4 assumed), so a char budget tuned for
        prose still lets some chunks through well over this model's 512-
        token limit. Past that limit, sentence-transformers silently
        truncates — the embedding only ever sees the first 512 tokens, with
        no error — so a number sitting past that point is invisible to
        search forever. This is the backstop: anything still oversized after
        silver's char-based split gets sliced on the model's own tokenizer,
        a guarantee a char-based guess can't give.
        """
        tok = self._model.tokenizer
        ids = tok(text, add_special_tokens=False)["input_ids"]
        limit = self.max_seq_length - 2  # leave room for [CLS]/[SEP]
        if len(ids) <= limit:
            return [text]
        windows = [ids[i : i + limit] for i in range(0, len(ids), limit)]
        return [tok.decode(w, skip_special_tokens=True) for w in windows]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed stored passages (no instruction prefix). Normalised vectors."""
        vecs = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,   # unit length -> cosine == dot product
            show_progress_bar=len(texts) > 200,
        )
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query (WITH the bge instruction prefix)."""
        vec = self._model.encode(
            self._query_instruction + text,
            normalize_embeddings=True,
        )
        return vec.tolist()
