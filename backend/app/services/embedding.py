"""Embedding Service — pluggable embedding provider.

Supports OpenAI and Gemini embedding models.
Used by QueryEmbeddingNode and Embedding Pipeline.
"""

from __future__ import annotations

import abc
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class BaseEmbeddingProvider(abc.ABC):
    """Abstract embedding provider."""

    @abc.abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        ...

    @abc.abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        ...

    @abc.abstractmethod
    def get_dimension(self) -> int:
        """Return embedding dimension."""
        ...


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider (text-embedding-3-small/large)."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.api_key = api_key
        self.model = model
        self._dimension = 1536 if "small" in model else 3072

    async def embed(self, text: str) -> list[float]:
        import openai

        client = openai.AsyncOpenAI(api_key=self.api_key)
        response = await client.embeddings.create(
            input=text,
            model=self.model,
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        import openai

        client = openai.AsyncOpenAI(api_key=self.api_key)
        response = await client.embeddings.create(
            input=texts,
            model=self.model,
        )
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

    def get_dimension(self) -> int:
        return self._dimension


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """Google Gemini embedding provider."""

    def __init__(self, api_key: str, model: str = "text-embedding-004"):
        self.api_key = api_key
        self.model = model
        self._dimension = 768

    async def embed(self, text: str) -> list[float]:
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        result = genai.embed_content(
            model=f"models/{self.model}",
            content=text,
        )
        return result["embedding"]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Gemini doesn't support true batch — sequential
        return [await self.embed(text) for text in texts]

    def get_dimension(self) -> int:
        return self._dimension


def create_embedding_provider() -> BaseEmbeddingProvider | None:
    """Factory: create embedding provider from settings."""
    settings = get_settings()

    if settings.openai_api_key:
        logger.info("embedding_provider", provider="openai", model=settings.embedding_model)
        return OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
        )

    if settings.gemini_api_key:
        logger.info("embedding_provider", provider="gemini", model=settings.embedding_model)
        return GeminiEmbeddingProvider(
            api_key=settings.gemini_api_key,
            model=settings.embedding_model,
        )

    logger.warning("no_embedding_provider", message="No API key configured")
    return None
