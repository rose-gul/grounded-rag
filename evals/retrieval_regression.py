"""Retrieval regression suite. For each golden question, the expected source
document(s) MUST appear in what the retriever returns. This is the RAG analogue
of a golden-file test: a chunking/embedding/top-k change that stops retrieving
the right doc fails the build.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag.schemas import Chunk, EvalCase


def load_eval(path: str) -> list[EvalCase]:
    cases = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(EvalCase(**json.loads(line)))
    return cases


def recall_at_k(expected_doc_ids: list[str], retrieved: list[Chunk]) -> float:
    """Fraction of expected docs that showed up in the retrieved chunks."""
    if not expected_doc_ids:
        return 1.0
    got = {c.doc_id for c in retrieved}
    hits = sum(1 for d in expected_doc_ids if d in got)
    return hits / len(expected_doc_ids)


def run(data_path: str, retrieve_fn, threshold: float = 1.0) -> int:
    """retrieve_fn: query -> list[Chunk]. Returns process exit code."""
    cases = load_eval(data_path)
    failures = []
    for case in cases:
        r = recall_at_k(case.expected_doc_ids, retrieve_fn(case.question))
        if r < threshold:
            failures.append((case.id, r))
    if failures:
        print("❌ RETRIEVAL REGRESSION — expected docs not retrieved:")
        for cid, r in failures:
            print(f"   {cid}: recall {r:.2f} < {threshold}")
        return 1
    print(f"✅ retrieval regression passed on {len(cases)} cases (recall ≥ {threshold})")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/eval/rag_eval.jsonl")
    args = ap.parse_args()
    # TODO(you): build the real pipeline and pass its retriever.retrieve here.
    from rag.pipeline import build_pipeline
    pipe = build_pipeline()
    sys.exit(run(args.data, pipe.retriever.retrieve))


if __name__ == "__main__":
    main()
