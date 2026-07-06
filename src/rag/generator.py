"""Generation with mandatory citations + abstention. The prompt is assembled so
each context carries its citable [S#] label; downstream, evals/citations.py
verifies the model actually cited real sources.
"""

from __future__ import annotations

import re

from rag.schemas import Answer, Chunk

_CITE = re.compile(r"\[(S\d+)\]")

ABSTAIN_TEXT = "I don't know based on the provided sources."


def build_context_block(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{c.source_id}] {c.text}" for c in chunks)


def extract_citations(answer_text: str) -> list[str]:
    """Pull the [S#] markers the model emitted, de-duplicated, in order."""
    seen: list[str] = []
    for m in _CITE.finditer(answer_text):
        if m.group(1) not in seen:
            seen.append(m.group(1))
    return seen


def generate(query: str, chunks: list[Chunk], system: str, model: str,
             temperature: float = 0.0) -> Answer:
    # Abstain when retrieval returned nothing usable — never hallucinate.
    if not chunks:
        return Answer(text=ABSTAIN_TEXT, citations=[], contexts=[], abstained=True)

    _context = build_context_block(chunks)                # noqa: F841 (used in real call)
    text = _call_llm(system, query, _context, model, temperature)
    return Answer(
        text=text,
        citations=extract_citations(text),
        contexts=chunks,
        abstained=text.strip() == ABSTAIN_TEXT,
    )


def _call_llm(system: str, query: str, context: str, model: str, temperature: float) -> str:
    """TODO(you): real OpenAI chat call with messages=[system, context+query].
    Kept stubbed so pure-logic tests (extract_citations, abstention) run offline."""
    # from openai import OpenAI
    # msg = f"Sources:\n{context}\n\nQuestion: {query}"
    # ...
    raise NotImplementedError("wire up the LLM call in Week 8")
