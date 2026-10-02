"""The RAG pipeline: question -> retrieve -> grounded, cited answer (or refuse).

This ties the whole system together:
  1. Embed the user's question (bge query embedding).
  2. Retrieve the top-k most similar chunks from Qdrant.
  3. GUARD: if the best match is too weak, refuse instead of guessing.
  4. Build a prompt that shows the model ONLY those chunks, numbered.
  5. Ask the model to answer using only that context and cite sources [n].
  6. Return the answer plus the source list for traceability.

Steps 3 and 5 are the "grounded" part — the difference between a system that
says "I don't know" honestly and one that hallucinates a plausible-sounding
number. For financial data, that honesty is the whole point.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from gamefin.index.embedder import Embedder
from gamefin.index.vector_store import VectorStore
from gamefin.logging_conf import get_logger
from gamefin.pipeline_config import PipelineConfig
from gamefin.serve.llm import LLMClient

log = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are a financial research assistant answering questions about company "
    "SEC filings. You must answer ONLY using the numbered sources provided. "
    "Cite the sources you use inline with bracketed numbers like [1], [2]. "
    "If the sources do not contain the answer, reply exactly: "
    "\"I could not find this in the available filings.\" Do not use outside "
    "knowledge. Do not guess numbers."
)


@dataclass
class Source:
    n: int
    company: str
    form: str
    report_date: str
    section: str
    url: str
    score: float
    text: str


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[Source]
    refused: bool
    detected_ticker: str | None = None  # set when auto-detected from the question text


def _detect_ticker(question: str, companies: dict[str, list[str]]) -> str | None:
    """Find a single company named in the question, for an automatic filter.

    Unfiltered search runs across every company's filings at once. Pure
    semantic similarity doesn't reliably favour the company actually named
    in the question — competitors' risk-factors sections, for instance, are
    structurally similar enough (competition, platform dependency, ...) to
    outrank the right company's own chunks. Naming the company and NOT
    filtering to it was exactly the failure mode that motivated this: a
    question asking for Take-Two's risks returned mostly Roblox's.

    Only returns a ticker when exactly one company's aliases match — a
    question naming two companies ("compare TTWO and EA") must stay
    unfiltered, or we'd silently hide one side of the comparison.
    """
    q = question.lower().replace("-", " ")
    matched = {
        ticker
        for ticker, aliases in companies.items()
        if any(re.search(r"\b" + re.escape(alias.lower().replace("-", " ")) + r"\b", q) for alias in aliases)
    }
    return matched.pop() if len(matched) == 1 else None


class RagPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.embedder = Embedder(
            model_name=cfg.gold.embedding_model,
            query_instruction=cfg.gold.query_instruction,
        )
        self.store = VectorStore(
            url=cfg.gold.qdrant_url,
            collection=cfg.gold.collection,
            dim=cfg.gold.embedding_dim,
            distance=cfg.gold.distance,
        )
        self.llm = LLMClient(
            provider=cfg.serve.llm_provider,
            model=cfg.serve.llm_model,
            temperature=cfg.serve.temperature,
            max_tokens=cfg.serve.max_tokens,
        )

    def ask(self, question: str, ticker: str | None = None) -> Answer:
        # 1. embed the question
        qvec = self.embedder.embed_query(question)

        # 2. retrieve top-k, filtered to one company if the caller named one
        # explicitly (--ticker) or the question names exactly one on its own.
        detected = ticker or _detect_ticker(question, self.cfg.companies)
        flt = {"ticker": detected} if detected else None
        hits = self.store.search(qvec, top_k=self.cfg.serve.top_k, flt=flt)

        # 3. refusal guard: nothing retrieved, or best match too weak
        if not hits or hits[0].score < self.cfg.serve.min_score:
            return Answer(
                question=question,
                answer="I could not find this in the available filings.",
                sources=[],
                refused=True,
                detected_ticker=detected,
            )

        sources = [
            Source(
                n=i + 1,
                company=h.payload["company"],
                form=h.payload["form"],
                report_date=h.payload["report_date"],
                section=h.payload["section"],
                url=h.payload["url"],
                score=float(h.score),
                text=h.payload["text"],
            )
            for i, h in enumerate(hits)
        ]

        # 4. build the numbered context block
        context = "\n\n".join(
            f"[{s.n}] ({s.company} {s.form} {s.report_date} — {s.section})\n{s.text}"
            for s in sources
        )
        user_prompt = f"Question: {question}\n\nSources:\n{context}\n\nAnswer:"

        # 5. generate a grounded, cited answer
        answer_text = self.llm.complete(SYSTEM_PROMPT, user_prompt)

        # 6. detect an explicit refusal from the model too
        refused = "could not find this in the available filings" in answer_text.lower()
        return Answer(
            question=question, answer=answer_text, sources=sources, refused=refused, detected_ticker=detected
        )
