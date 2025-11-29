"""Factory for creating embedding service instances based on configuration."""

from app.core.config import settings
from app.interfaces.embedding_service import IEmbeddingService
from app.services.embedding.google_embedding_service import GoogleEmbeddingService
from app.services.embedding.openai_embedding_service import OpenAIEmbeddingService


def create_embedding_service() -> IEmbeddingService:
    """
    Create embedding service instance based on settings.embedding_provider.

    :return: Google or OpenAI embedding service
    :raises ValueError: If provider is not supported or API key is missing
    """
    if settings.embedding_provider == "google":
        if not settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY is required when embedding_provider='google'")

        return GoogleEmbeddingService(
            api_key=settings.google_api_key,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )

    elif settings.embedding_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when embedding_provider='openai'")

        return OpenAIEmbeddingService(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )

    else:
        raise ValueError(
            f"Unsupported embedding provider: {settings.embedding_provider}. Supported providers: 'google', 'openai'"
        )
