"""Split long section text into overlapping chunks for embedding.

Why chunk at all? An embedding model turns a piece of text into ONE vector.
Feed it a whole 100-page 10-K and you get one blurry vector that means
"annual report" and retrieves nothing precisely. Split it into ~1000-token
passages and each vector captures one specific idea, so retrieval is sharp.

Why overlap? A fact can straddle a boundary ("...net bookings were $5.6bn |
up 12%..."). Overlapping consecutive chunks by a few hundred characters means
the whole fact still appears intact in at least one chunk.

The splitter is recursive: it prefers to break on paragraph breaks, then
sentences, then spaces — so chunks end at natural boundaries, not mid-word.
Pure-Python, no heavy dependencies, fully testable.
"""
from __future__ import annotations

# Break points in order of preference: paragraph, line, sentence, word.
_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split_recursive(text: str, target: int, seps: list[str]) -> list[str]:
    """Split text into pieces <= target chars, breaking on the best separator."""
    if len(text) <= target:
        return [text]
    if not seps:
        # No separators left: hard-cut at target (rare fallback).
        return [text[i : i + target] for i in range(0, len(text), target)]

    sep = seps[0]
    parts = text.split(sep)
    if len(parts) == 1:
        # This separator doesn't occur; try the next finer one.
        return _split_recursive(text, target, seps[1:])

    # Greedily pack parts back together up to the target size.
    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = part if not current else current + sep + part
        if len(candidate) <= target:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # A single part may still exceed target -> recurse with finer seps.
            if len(part) > target:
                chunks.extend(_split_recursive(part, target, seps[1:]))
                current = ""
            else:
                current = part
    if current:
        chunks.append(current)
    return chunks


def chunk_text(
    text: str,
    chunk_chars: int = 4000,
    overlap_chars: int = 600,
    min_chunk_chars: int = 200,
) -> list[str]:
    """Turn one section's text into overlapping, sensibly-bounded chunks."""
    if not text or not text.strip():
        return []

    base = _split_recursive(text.strip(), chunk_chars, _SEPARATORS)

    # Add trailing overlap: prepend the tail of the previous chunk to each next.
    with_overlap: list[str] = []
    for i, chunk in enumerate(base):
        if i == 0 or overlap_chars <= 0:
            with_overlap.append(chunk)
        else:
            tail = base[i - 1][-overlap_chars:]
            with_overlap.append((tail + " " + chunk).strip())

    # Drop tiny fragments (page numbers, stray boilerplate).
    return [c for c in with_overlap if len(c) >= min_chunk_chars]
