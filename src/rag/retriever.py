"""Retrieve candidate chunks from Qdrant, optionally rerank, and apply the
abstention threshold. Defined against a small Protocol so tests can inject a
fake retriever without a live vector DB.
"""

from __future__ import annotations

from typing import Protocol

from rag.schemas import Chunk


class VectorStore(Protocol):
    def search(self, query: str, top_k: int) -> list[Chunk]:
        """Return top_k candidate chunks with similarity scores."""
        ...


class Retriever:
    def __init__(self, store: VectorStore, top_k: int, final_k: int,
                 min_score: float, rerank: bool = True) -> None:
        self.store = store
        self.top_k = top_k
        self.final_k = final_k
        self.min_score = min_score
        self.rerank = rerank

    def retrieve(self, query: str) -> list[Chunk]:
        candidates = self.store.search(query, self.top_k)
        if self.rerank:
            candidates = self._rerank(query, candidates)
        # drop weak matches; the pipeline abstains if nothing survives
        kept = [c for c in candidates if c.score >= self.min_score]
        return kept[: self.final_k]

    def _rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """TODO(you): call a cross-encoder (bge-reranker) or Cohere rerank and
        overwrite chunk.score. For now, keep vector order (identity rerank)."""
        return sorted(chunks, key=lambda c: c.score, reverse=True)
