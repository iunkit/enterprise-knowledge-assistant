"""Thin factories for the Azure OpenAI and Azure AI Search clients.

Clients are cached per-process: both are thread-safe and hold connection pools,
so rebuilding them per request measurably slowed the endpoint down.
"""

from __future__ import annotations

from functools import lru_cache

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from openai import AzureOpenAI

from app.config import Settings, get_settings


@lru_cache
def get_openai_client() -> AzureOpenAI:
    settings = get_settings()
    return AzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
    )


@lru_cache
def get_search_client() -> SearchClient:
    settings = get_settings()
    return SearchClient(
        endpoint=settings.azure_search_endpoint,
        index_name=settings.azure_search_index,
        credential=AzureKeyCredential(settings.azure_search_api_key),
    )


def get_index_client(settings: Settings | None = None) -> SearchIndexClient:
    settings = settings or get_settings()
    return SearchIndexClient(
        endpoint=settings.azure_search_endpoint,
        credential=AzureKeyCredential(settings.azure_search_api_key),
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings with the configured embedding deployment."""
    if not texts:
        return []
    settings = get_settings()
    client = get_openai_client()
    response = client.embeddings.create(
        model=settings.azure_openai_embedding_deployment,
        input=texts,
    )
    # The API preserves input order, but sort by index defensively.
    return [item.embedding for item in sorted(response.data, key=lambda d: d.index)]
