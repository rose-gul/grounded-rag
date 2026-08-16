"""Offline ingest: read the corpus, chunk it, embed, and upsert to Qdrant.

Network-touching pieces are marked TODO(you) — fill them in during the study plan.
The chunking + id-assignment logic is real and testable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from rag.chunking import chunk_text
from rag.schemas import Chunk


def load_corpus(corpus_dir: str) -> dict[str, str]:
    """doc_id -> raw text for every .md/.txt file in the corpus."""
    docs: dict[str, str] = {}
    for p in sorted(Path(corpus_dir).glob("**/*")):
        if p.suffix.lower() in {".md", ".txt"}:
            docs[p.stem] = p.read_text(encoding="utf-8")
    return docs


def build_chunks(docs: dict[str, str], max_tokens: int, overlap: int) -> list[Chunk]:
    """Assign every chunk a stable, human-citable source id (S1, S2, ...)."""
    chunks: list[Chunk] = []
    n = 0
    for doc_id, text in docs.items():
        for piece in chunk_text(text, max_tokens, overlap):
            n += 1
            chunks.append(Chunk(source_id=f"S{n}", doc_id=doc_id, text=piece))
    return chunks


def embed(texts: list[str], model: str) -> list[list[float]]:
    """TODO(you): call OpenAI embeddings (or a local model) and return vectors."""
    raise NotImplementedError("wire up embeddings in Week 8 of the study plan")


def upsert(chunks: list[Chunk], vectors: list[list[float]], collection: str) -> None:
    """TODO(you): qdrant_client.upsert(...) with chunk payloads (doc_id, source_id, text)."""
    raise NotImplementedError("wire up Qdrant upsert")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/app.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    docs = load_corpus(cfg["corpus_dir"])
    chunks = build_chunks(docs, cfg["chunking"]["max_tokens"], cfg["chunking"]["overlap_tokens"])
    print(f"loaded {len(docs)} docs -> {len(chunks)} chunks")
    # vectors = embed([c.text for c in chunks], cfg["embedding"]["model"])
    # upsert(chunks, vectors, cfg["collection"])
    print("TODO(you): embed + upsert (see Week 8)")


if __name__ == "__main__":
    main()
