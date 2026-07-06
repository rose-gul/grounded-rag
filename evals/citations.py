"""Citation verification — the grounding check.

An answer is "grounded" only if every citation it makes points at a source that
was actually retrieved (no invented [S#]) AND every factual sentence carries at
least one citation (no uncited claims). Pure logic -> fully unit-tested.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.generator import ABSTAIN_TEXT, extract_citations
from rag.schemas import Chunk

_CITE = re.compile(r"\[S\d+\]")
_SENT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class CitationReport:
    grounded: bool
    invalid_citations: list[str]     # cited sources that weren't retrieved
    uncited_sentences: list[str]     # factual sentences with no citation
    abstained: bool


def verify(answer_text: str, retrieved: list[Chunk]) -> CitationReport:
    if answer_text.strip() == ABSTAIN_TEXT:
        return CitationReport(grounded=True, invalid_citations=[],
                              uncited_sentences=[], abstained=True)

    valid_ids = {c.source_id for c in retrieved}
    cited = extract_citations(answer_text)
    invalid = [c for c in cited if c not in valid_ids]

    uncited: list[str] = []
    for sent in _SENT.split(answer_text.strip()):
        s = sent.strip()
        if not s or not _is_factual(s):
            continue
        if not _CITE.search(s):
            uncited.append(s)

    grounded = not invalid and not uncited
    return CitationReport(grounded=grounded, invalid_citations=invalid,
                          uncited_sentences=uncited, abstained=False)


def _is_factual(sentence: str) -> bool:
    """Heuristic: skip hedges/meta sentences that don't assert a fact.
    Good enough for a grounding gate; refine as you see false positives."""
    lowered = sentence.lower()
    hedges = ("i don't know", "as an ai", "note that", "in summary", "overall,")
    return not any(h in lowered for h in hedges)
