"""End-to-end answer(): retrieve -> generate. The single entry point the API,
the UI, and the eval harness all call, so they exercise identical behaviour.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from rag.generator import generate
from rag.retriever import Retriever
from rag.schemas import Answer


class RagPipeline:
    def __init__(self, retriever: Retriever, gen_cfg: dict) -> None:
        self.retriever = retriever
        self.gen_cfg = gen_cfg

    def answer(self, query: str) -> Answer:
        chunks = self.retriever.retrieve(query)
        return generate(
            query, chunks,
            system=self.gen_cfg["system"],
            model=self.gen_cfg["model"],
            temperature=self.gen_cfg.get("temperature", 0.0),
        )


def load_config(path: str = "configs/app.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def build_pipeline(config_path: str = "configs/app.yaml") -> RagPipeline:
    """TODO(you): construct the Qdrant-backed VectorStore and Retriever here."""
    cfg = load_config(config_path)
    # store = QdrantVectorStore(cfg["collection"], cfg["embedding"])   # you implement
    # retriever = Retriever(store, **cfg["retrieval"])
    # return RagPipeline(retriever, cfg["generation"])
    raise NotImplementedError("assemble the real pipeline in Week 8")
