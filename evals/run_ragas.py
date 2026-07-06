"""Faithfulness / groundedness scoring with Ragas — the headline metric.

Ragas needs: question, generated answer, retrieved contexts, and (for some
metrics) a reference answer. This wires the pipeline's outputs into that shape.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.schemas import EvalCase


def load_eval(path: str) -> list[EvalCase]:
    return [EvalCase(**json.loads(l)) for l in Path(path).read_text().splitlines() if l.strip()]


def build_ragas_dataset(cases: list[EvalCase], pipeline) -> list[dict]:
    """Run each question through the pipeline and collect Ragas-shaped rows."""
    rows = []
    for case in cases:
        ans = pipeline.answer(case.question)
        rows.append({
            "question": case.question,
            "answer": ans.text,
            "contexts": [c.text for c in ans.contexts],
            "ground_truth": case.reference_answer,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/app.yaml")
    ap.add_argument("--data", default="data/eval/rag_eval.jsonl")
    args = ap.parse_args()

    # TODO(you) — once the pipeline is wired:
    #   from datasets import Dataset
    #   from ragas import evaluate
    #   from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    #   from rag.pipeline import build_pipeline
    #   cases = load_eval(args.data)
    #   rows = build_ragas_dataset(cases, build_pipeline(args.config))
    #   result = evaluate(Dataset.from_list(rows),
    #                     metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
    #   print(result)   # <- your headline numbers; write them into the README
    print("TODO(you): wire Ragas evaluate() — see Week 7 of the study plan.")


if __name__ == "__main__":
    main()
