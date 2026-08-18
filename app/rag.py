"""The RAG pipeline: retrieve -> ground -> generate."""

from __future__ import annotations

import logging
import time

from app.azure_clients import get_openai_client
from app.config import get_settings
from app.models import AskResponse, Citation
from app.prompts import NO_ANSWER, SYSTEM_PROMPT, build_user_message
from app.retrieval import retrieve

logger = logging.getLogger(__name__)

EXCERPT_CHARS = 300


def answer_question(
    question: str, top_k: int | None = None, source: str | None = None
) -> AskResponse:
    settings = get_settings()
    started = time.perf_counter()

    chunks = retrieve(question, top_k=top_k, source=source)

    if not chunks:
        # Nothing retrieved means nothing to ground on -- short-circuit instead of
        # asking the model, which would otherwise answer from pretraining.
        return AskResponse(
            answer=NO_ANSWER,
            citations=[],
            grounded=False,
            latency_ms=_elapsed_ms(started),
        )

    completion = get_openai_client().chat.completions.create(
        model=settings.azure_openai_chat_deployment,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(question, chunks)},
        ],
        temperature=settings.temperature,
        max_tokens=settings.max_answer_tokens,
    )
    answer = (completion.choices[0].message.content or "").strip()

    return AskResponse(
        answer=answer,
        citations=_citations(chunks),
        grounded=NO_ANSWER not in answer,
        latency_ms=_elapsed_ms(started),
    )


def _citations(chunks: list[dict]) -> list[Citation]:
    return [
        Citation(
            n=i,
            source=c["source"],
            title=c["title"],
            chunk_index=c["chunk_index"],
            score=round(c["score"], 4),
            excerpt=c["content"][:EXCERPT_CHARS].strip(),
        )
        for i, c in enumerate(chunks, start=1)
    ]


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
