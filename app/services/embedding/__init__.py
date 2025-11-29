"""Embedding services for vector search (Google Gemini, OpenAI)."""

from app.services.embedding.embedding_factory import create_embedding_service
from app.services.embedding.google_embedding_service import GoogleEmbeddingService
from app.services.embedding.openai_embedding_service import OpenAIEmbeddingService

__all__ = [
    "create_embedding_service",
    "GoogleEmbeddingService",
    "OpenAIEmbeddingService",
]
