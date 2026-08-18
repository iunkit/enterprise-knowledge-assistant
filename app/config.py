"""Application settings, loaded from environment / .env."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Azure OpenAI ---
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_chat_deployment: str = "gpt-4o-mini"
    azure_openai_embedding_deployment: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # --- Azure AI Search ---
    azure_search_endpoint: str = ""
    azure_search_api_key: str = ""
    azure_search_index: str = "enterprise-knowledge"
    # Semantic ranker needs a semantic configuration on the index; disable if the
    # search tier does not support it.
    use_semantic_ranker: bool = True

    # --- Retrieval / chunking ---
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 100
    top_k: int = 5
    # Retrieved chunks scoring below this are dropped before they reach the prompt.
    min_score: float = 0.0

    # --- Generation ---
    max_answer_tokens: int = 800
    temperature: float = 0.0

    @property
    def configured(self) -> bool:
        return bool(
            self.azure_openai_endpoint
            and self.azure_openai_api_key
            and self.azure_search_endpoint
            and self.azure_search_api_key
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
