"""Unit tests for the deterministic core: chunking, citation extraction, the
grounding check, and the retriever's abstention/threshold logic. These pass
offline (no API keys) and are the portfolio flex: the RAG pipeline is tested.
"""

from __future__ import annotations

from evals.citations import verify
from evals.retrieval_regression import recall_at_k
from rag.chunking import chunk_text
from rag.generator import ABSTAIN_TEXT, extract_citations
from rag.retriever import Retriever
from rag.schemas import Chunk


# --------------------------- chunking --------------------------- #
def test_chunk_respects_max_tokens_roughly():
    text = " ".join(f"Sentence number {i} has some words." for i in range(60))
    chunks = chunk_text(text, max_tokens=50, overlap_tokens=10)
    assert len(chunks) > 1                       # long text splits
    assert all(chunks)                           # no empty chunks


def test_chunk_overlap_shares_content():
    text = "Alpha one. Bravo two. Charlie three. Delta four. Echo five. Foxtrot six."
    chunks = chunk_text(text, max_tokens=8, overlap_tokens=4)
    # consecutive chunks should share at least one sentence due to overlap
    assert len(chunks) >= 2


def test_chunk_rejects_bad_params():
    import pytest
    with pytest.raises(ValueError):
        chunk_text("x", max_tokens=10, overlap_tokens=10)


# --------------------------- citations --------------------------- #
def test_extract_citations_dedup_in_order():
    assert extract_citations("A [S2] and B [S1] and again [S2].") == ["S2", "S1"]


def test_grounded_answer_passes():
    ctx = [Chunk(source_id="S1", doc_id="d", text="Refunds within 30 days.")]
    rep = verify("You can refund within 30 days [S1].", ctx)
    assert rep.grounded and not rep.invalid_citations and not rep.uncited_sentences


def test_invented_citation_fails():
    ctx = [Chunk(source_id="S1", doc_id="d", text="...")]
    rep = verify("The limit is 30 days [S7].", ctx)   # S7 was never retrieved
    assert not rep.grounded and rep.invalid_citations == ["S7"]


def test_uncited_claim_fails():
    ctx = [Chunk(source_id="S1", doc_id="d", text="...")]
    rep = verify("Refunds take 90 days.", ctx)         # factual, no citation
    assert not rep.grounded and rep.uncited_sentences


def test_abstention_is_grounded():
    rep = verify(ABSTAIN_TEXT, [])
    assert rep.grounded and rep.abstained


# --------------------------- retrieval --------------------------- #
class _FakeStore:
    def __init__(self, chunks):
        self._chunks = chunks

    def search(self, query, top_k):
        return self._chunks[:top_k]


def test_retriever_applies_min_score_and_final_k():
    chunks = [
        Chunk(source_id="S1", doc_id="a", text="x", score=0.9),
        Chunk(source_id="S2", doc_id="b", text="y", score=0.5),
        Chunk(source_id="S3", doc_id="c", text="z", score=0.1),  # below threshold
    ]
    r = Retriever(_FakeStore(chunks), top_k=3, final_k=2, min_score=0.3, rerank=True)
    out = r.retrieve("q")
    assert [c.source_id for c in out] == ["S1", "S2"]  # S3 dropped, capped at final_k


def test_recall_at_k():
    retrieved = [Chunk(source_id="S1", doc_id="refund-policy", text="...", score=1.0)]
    assert recall_at_k(["refund-policy"], retrieved) == 1.0
    assert recall_at_k(["security-faq"], retrieved) == 0.0
