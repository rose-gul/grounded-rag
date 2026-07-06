# 🔎 Grounded — a deployed RAG assistant that proves it isn't hallucinating

> A production-shaped retrieval-augmented assistant over a real document corpus,
> deployed as a containerized API + web UI, with a **groundedness / faithfulness
> eval harness** that turns "trust me" into a measured hallucination rate.

<!-- Replace with your real numbers once you've run the eval. This before/after is what gets the repo forwarded. -->
**Headline result (example — replace with yours):** On a 120-question eval set over a
regulatory corpus, adding a reranker + a citation-required prompt raised **faithfulness
0.71 → 0.94** and cut the **hallucination rate 6.2% → 0.7%**, with **context recall 0.89**.
Every claim in an answer is traced to a retrieved source or the model abstains.

---

## Why this project exists

RAG demos are everywhere. **Evaluated, deployed** RAG with a published hallucination
number is not — and in 2026 an *unevaluated* LLM feature is a hiring red flag. This repo
treats a RAG pipeline the way a test engineer treats any system with many failure points:

| RAG failure point | How this repo tests it |
|---|---|
| Retrieval misses the right document | **Retrieval regression suite** — fixed queries whose expected source docs *must* keep being retrieved (`evals/retrieval_regression.py`) |
| Model invents facts not in context | **Faithfulness / groundedness** scoring (Ragas) + **citation verification** (`evals/citations.py`) |
| Model answers when it shouldn't | **Abstention test** — when context is weak, it must say "I don't know" |
| A prompt/chunking change silently degrades quality | **Evals-in-CI** gate on every PR |

This is also the **"app under test"** for [Project 5 (eval harness)](../project-05-eval-bakeoff)
and [Project 9 (EvalOps)](../project-09-evalops) — build it, then evaluate and attack it.

---

## Architecture

```
  ingest (offline)                          query (online)
  ┌────────────┐   ┌──────────┐             ┌─────────────┐   ┌──────────┐   ┌─────────────┐
  │ corpus     │──▶│ chunk +  │──▶ Qdrant   │ user query  │──▶│ retrieve │──▶│ rerank      │
  │ (docs)     │   │ embed    │   (vectors) │ (FastAPI)   │   │ (top-k)  │   │ (bge/Cohere)│
  └────────────┘   └──────────┘             └─────────────┘   └──────────┘   └──────┬──────┘
                                                                                     ▼
                                              answer + citations ◀── generate (citation-required prompt)
                                                     │
                                              ┌──────▼───────────────────────────────┐
                                              │ EVAL: Ragas (faithfulness, context    │
                                              │ precision/recall) + citation check +  │
                                              │ retrieval regression  → CI gate       │
                                              └───────────────────────────────────────┘
```

## Quickstart

```bash
make install
cp .env.example .env            # add OPENAI_API_KEY (+ optional COHERE_API_KEY)
make up                         # start Qdrant (docker compose)
make ingest                     # chunk + embed data/corpus/ into Qdrant
make api                        # serve the RAG API at :8000
make ui                         # Streamlit demo at :8501

make eval                       # run Ragas + retrieval regression + citation checks
make test                       # deterministic unit tests (chunking, citation logic)
```

## Repo layout

```
grounded-rag/
├── README.md · pyproject.toml · Makefile · Dockerfile · docker-compose.yml
├── .env.example · .github/workflows/evals.yml
├── configs/app.yaml                # chunk size, top-k, models, thresholds
├── data/
│   ├── corpus/                      # source docs (swap in a real, messy corpus)
│   └── eval/rag_eval.jsonl          # golden Q&A with expected source + reference answer
├── src/rag/
│   ├── chunking.py                  # pure chunking logic  ✓tested
│   ├── ingest.py                    # embed + upsert to Qdrant
│   ├── retriever.py                 # retrieve + rerank
│   ├── generator.py                 # citation-required generation + abstention
│   ├── pipeline.py                  # end-to-end answer()
│   ├── api.py                       # FastAPI service
│   └── schemas.py
├── evals/
│   ├── citations.py                 # verify every claim maps to a source  ✓tested
│   ├── retrieval_regression.py      # expected docs must be retrieved
│   └── run_ragas.py                 # faithfulness / context precision-recall
├── ui/streamlit_app.py              # the live demo
└── tests/test_rag.py                # chunking + citation verification tests
```

## Design decisions (talking points)
- **Citations are mandatory.** The generator prompt forces `[source_id]` markers; an answer
  with an unsupported claim fails the citation check. Grounding is enforced, not hoped for.
- **Abstention over hallucination.** Below a retrieval-confidence threshold the assistant
  returns "I don't know" — and the eval set measures abstention accuracy.
- **The eval is the product.** Lead the README with the faithfulness before/after; the
  pipeline is table stakes, the measurement is the differentiator.

## Stretch goals
- [ ] LLM-as-judge faithfulness with human-calibrated agreement (bridge to Project 5)
- [ ] Per-answer confidence + measured abstention accuracy
- [ ] Nightly eval dashboard; hybrid (BM25 + dense) retrieval; query rewriting

---

🚧 **Scaffold.** Structural files + pure-logic pieces are implemented and unit-tested;
network-touching pieces (embeddings, Qdrant, LLM calls) are stubbed with `TODO(you)`
markers to fill in as you work through the Month-2 study plan.
