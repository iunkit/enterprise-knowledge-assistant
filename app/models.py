"""Request/response schemas for the API."""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    # Restrict retrieval to one source document, e.g. "hr-handbook.md".
    source: str | None = None


class Citation(BaseModel):
    n: int
    source: str
    title: str
    chunk_index: int
    score: float
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    grounded: bool
    latency_ms: int


class SearchHit(BaseModel):
    source: str
    title: str
    chunk_index: int
    score: float
    content: str


class HealthResponse(BaseModel):
    status: str
    index: str
    chat_deployment: str
    embedding_deployment: str
    azure_configured: bool
