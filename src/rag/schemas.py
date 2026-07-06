"""Typed contracts for the pipeline. A retrieved chunk carries a stable source id
so citations can be verified against exactly what was shown to the model.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    source_id: str          # e.g. "S1" — the label the model must cite
    doc_id: str             # originating document
    text: str
    score: float = 0.0      # retrieval / rerank score


class Query(BaseModel):
    text: str
    top_k: int | None = None


class Answer(BaseModel):
    text: str
    citations: list[str] = Field(default_factory=list)   # source_ids referenced
    contexts: list[Chunk] = Field(default_factory=list)  # what was retrieved
    abstained: bool = False


class EvalCase(BaseModel):
    """One golden item: a question, its reference answer, and the doc(s) that
    should be retrieved to answer it."""

    id: str
    question: str
    reference_answer: str
    expected_doc_ids: list[str] = Field(default_factory=list)
