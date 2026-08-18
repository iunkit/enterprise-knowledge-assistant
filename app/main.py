"""FastAPI surface for the knowledge assistant."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.models import AskRequest, AskResponse, HealthResponse, SearchHit
from app.rag import answer_question
from app.retrieval import retrieve

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enterprise Knowledge Assistant",
    description="RAG question answering over internal documents, grounded in "
    "Azure AI Search and answered by Azure OpenAI.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _require_config() -> None:
    if not get_settings().configured:
        raise HTTPException(
            status_code=503,
            detail="Azure OpenAI / Azure AI Search credentials are not configured.",
        )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        index=settings.azure_search_index,
        chat_deployment=settings.azure_openai_chat_deployment,
        embedding_deployment=settings.azure_openai_embedding_deployment,
        azure_configured=settings.configured,
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Answer a question against the indexed documents, with citations."""
    _require_config()
    try:
        return answer_question(
            request.question, top_k=request.top_k, source=request.source
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("ask failed")
        raise HTTPException(status_code=502, detail=f"Upstream failure: {exc}") from exc


@app.get("/search", response_model=list[SearchHit])
def search(q: str, top_k: int | None = None, source: str | None = None) -> list[SearchHit]:
    """Retrieval only -- useful for debugging which chunks a question pulls."""
    _require_config()
    try:
        hits = retrieve(q, top_k=top_k, source=source)
    except Exception as exc:
        logger.exception("search failed")
        raise HTTPException(status_code=502, detail=f"Upstream failure: {exc}") from exc
    return [SearchHit(**hit) for hit in hits]
