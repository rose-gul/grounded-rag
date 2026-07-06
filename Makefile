PY ?= python

.PHONY: help install up down ingest api ui eval test lint fmt clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## Install deps (dev + eval + ui)
	@if command -v uv >/dev/null 2>&1; then uv pip install -e ".[dev,eval,ui]"; \
	else $(PY) -m pip install -e ".[dev,eval,ui]"; fi

up:  ## Start Qdrant (vector DB) via docker compose
	docker compose up -d qdrant

down:  ## Stop services
	docker compose down

ingest:  ## Chunk + embed data/corpus/ into Qdrant
	$(PY) -m rag.ingest --config configs/app.yaml

api:  ## Serve the RAG API at :8000
	uvicorn rag.api:app --reload --port 8000

ui:  ## Run the Streamlit demo at :8501
	streamlit run ui/streamlit_app.py

eval:  ## Run Ragas + retrieval regression + citation checks
	$(PY) -m evals.run_ragas --config configs/app.yaml --data data/eval/rag_eval.jsonl
	$(PY) -m evals.retrieval_regression --data data/eval/rag_eval.jsonl

test:  ## Deterministic unit tests (chunking + citation logic)
	$(PY) -m pytest

lint:  ## Ruff + mypy
	ruff check src evals tests && mypy src

fmt:  ## Auto-format
	ruff check --fix src evals tests && ruff format src evals tests

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
