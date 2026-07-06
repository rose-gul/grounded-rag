"""Token-aware chunking. Pure logic (no network), so it's fully unit-tested —
chunking quietly determines retrieval quality, so it deserves real tests.
"""

from __future__ import annotations

import re

# Split on sentence boundaries so a chunk never cuts a sentence in half.
_SENT = re.compile(r"(?<=[.!?])\s+")


def _approx_tokens(text: str) -> int:
    """Cheap token estimate (~0.75 words/token) so tests run without tiktoken.
    In production, swap for tiktoken.encoding_for_model(...).encode()."""
    words = len(text.split())
    return max(1, round(words / 0.75))


def chunk_text(
    text: str, max_tokens: int = 400, overlap_tokens: int = 60
) -> list[str]:
    """Greedy sentence-packing into ~max_tokens chunks with token overlap.

    Overlap keeps context continuous across chunk boundaries so a fact split
    across two sentences is still retrievable.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be < max_tokens")

    sentences = [s.strip() for s in _SENT.split(text.strip()) if s.strip()]
    chunks: list[str] = []
    cur: list[str] = []
    cur_tok = 0

    for sent in sentences:
        st = _approx_tokens(sent)
        if cur and cur_tok + st > max_tokens:
            chunks.append(" ".join(cur))
            # start next chunk with a tail overlap from the previous one
            cur, cur_tok = _tail_overlap(cur, overlap_tokens)
        cur.append(sent)
        cur_tok += st

    if cur:
        chunks.append(" ".join(cur))
    return chunks


def _tail_overlap(sentences: list[str], overlap_tokens: int) -> tuple[list[str], int]:
    """Take sentences from the end until we've accumulated ~overlap_tokens."""
    tail: list[str] = []
    tok = 0
    for sent in reversed(sentences):
        st = _approx_tokens(sent)
        if tok + st > overlap_tokens and tail:
            break
        tail.insert(0, sent)
        tok += st
    return tail, tok
