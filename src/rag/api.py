"""FastAPI service. `POST /ask` returns an answer with its citations and the
contexts that were retrieved — so a caller (or an eval) can verify grounding.
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from rag.pipeline import build_pipeline
from rag.schemas import Answer

app = FastAPI(title="Grounded RAG", version="0.1.0")

# Built once at startup. Wrapped so the app can boot for /health before the
# vector DB is wired up during development.
_pipeline = None


class AskRequest(BaseModel):
    query: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=Answer)
def ask(req: AskRequest) -> Answer:
    global _pipeline
    if _pipeline is None:
        _pipeline = build_pipeline()   # TODO(you): implement build_pipeline()
    return _pipeline.answer(req.query)
