"""Ingestion: read documents -> chunk -> embed -> upload to Azure AI Search."""

from __future__ import annotations

import logging
from pathlib import Path

from app.azure_clients import embed_texts, get_search_client
from app.chunking import Chunk, chunk_document, iter_documents
from app.config import get_settings
from app.index_setup import ensure_index

logger = logging.getLogger(__name__)

# Azure OpenAI caps inputs per embeddings call; batching also keeps a large
# corpus from hitting the per-minute token limit in one burst.
EMBED_BATCH = 32
UPLOAD_BATCH = 500


def build_chunks(docs_dir: Path) -> list[Chunk]:
    settings = get_settings()
    chunks: list[Chunk] = []
    for path in iter_documents(docs_dir):
        doc_chunks = chunk_document(
            path, settings.chunk_size_tokens, settings.chunk_overlap_tokens
        )
        logger.info("%s -> %d chunks", path.name, len(doc_chunks))
        chunks.extend(doc_chunks)
    return chunks


def embed_chunks(chunks: list[Chunk]) -> None:
    for start in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[start : start + EMBED_BATCH]
        for chunk, vector in zip(batch, embed_texts([c.content for c in batch])):
            chunk.embedding = vector


def upload_chunks(chunks: list[Chunk]) -> int:
    client = get_search_client()
    uploaded = 0
    for start in range(0, len(chunks), UPLOAD_BATCH):
        batch = chunks[start : start + UPLOAD_BATCH]
        results = client.upload_documents(
            documents=[c.to_search_document() for c in batch]
        )
        failed = [r for r in results if not r.succeeded]
        if failed:
            raise RuntimeError(
                f"{len(failed)} documents failed to upload; first error: "
                f"{failed[0].error_message}"
            )
        uploaded += len(batch)
    return uploaded


def ingest(docs_dir: Path, recreate_index: bool = True) -> dict:
    if recreate_index:
        ensure_index()
    chunks = build_chunks(docs_dir)
    if not chunks:
        return {"documents": 0, "chunks": 0, "uploaded": 0}
    embed_chunks(chunks)
    uploaded = upload_chunks(chunks)
    return {
        "documents": len({c.source for c in chunks}),
        "chunks": len(chunks),
        "uploaded": uploaded,
    }
