"""Unit tests for embedding factory and Google embedding service."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.services.embedding.embedding_factory import create_embedding_service
from app.services.embedding.google_embedding_service import GoogleEmbeddingService


@pytest.mark.unit
class TestEmbeddingFactory:
    """Verify factory selects provider correctly and validates keys."""

    def test_creates_google_service(self, monkeypatch):
        from app.core.config import settings

        monkeypatch.setattr(settings, "embedding_provider", "google")
        monkeypatch.setattr(settings, "google_api_key", "test-key")
        monkeypatch.setattr(settings, "embedding_model", "gemini-embedding-001")
        monkeypatch.setattr(settings, "embedding_dimensions", 1024)

        created = create_embedding_service()
        assert isinstance(created, GoogleEmbeddingService)

    def test_missing_google_key_raises(self, monkeypatch):
        from app.core.config import settings

        monkeypatch.setattr(settings, "embedding_provider", "google")
        monkeypatch.setattr(settings, "google_api_key", None)

        with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
            create_embedding_service()

    def test_missing_openai_key_raises(self, monkeypatch):
        from app.core.config import settings

        monkeypatch.setattr(settings, "embedding_provider", "openai")
        monkeypatch.setattr(settings, "openai_api_key", None)

        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            create_embedding_service()


@pytest.mark.unit
class TestGoogleEmbeddingService:
    """Covers embed_text/embed_batch helper logic."""

    @pytest.fixture
    def fake_client(self, monkeypatch):
        """Patch google genai client + types."""

        fake_models = MagicMock()

        class FakeClient:
            def __init__(self, api_key: str):
                self.api_key = api_key
                self.models = fake_models

        monkeypatch.setattr("app.services.embedding.google_embedding_service.genai.Client", FakeClient)
        monkeypatch.setattr(
            "app.services.embedding.google_embedding_service.types.EmbedContentConfig",
            lambda output_dimensionality: SimpleNamespace(dim=output_dimensionality),
        )

        return fake_models

    @pytest.mark.asyncio
    async def test_embed_text_returns_values(self, fake_client):
        fake_client.embed_content.return_value = SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1, 0.2])])

        service = GoogleEmbeddingService(api_key="abc", model="gemini", dimensions=3)

        result = await service.embed_text("hello")

        assert result == [0.1, 0.2]
        fake_client.embed_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_batch_handles_empty_input(self):
        service = GoogleEmbeddingService(api_key="abc", model="gemini", dimensions=3)

        with pytest.raises(ValueError, match="Batch cannot be empty"):
            await service.embed_batch([])

    @pytest.mark.asyncio
    async def test_embed_batch_returns_vectors(self, fake_client):
        fake_client.embed_content.return_value = SimpleNamespace(
            embeddings=[
                SimpleNamespace(values=[0.1]),
                SimpleNamespace(values=[0.2]),
            ]
        )

        service = GoogleEmbeddingService(api_key="abc", model="gemini", dimensions=3)

        result = await service.embed_batch(["a", "b"])

        assert result == [[0.1], [0.2]]
        fake_client.embed_content.assert_called_once()
