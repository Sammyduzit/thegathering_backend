import structlog
from google import genai
from google.genai import types

from app.core.decorators import standard_retry
from app.interfaces.embedding_service import IEmbeddingService

logger = structlog.get_logger(__name__)


class GoogleEmbeddingService(IEmbeddingService):
    """Google Gemini embedding service using gemini-embedding-001."""

    def __init__(self, api_key: str, model: str = "gemini-embedding-001", dimensions: int = 1536):
        """
        Initialize Google Gemini embedding service.

        :param api_key: Google API key
        :param model: Embedding model name (default: gemini-embedding-001)
        :param dimensions: Embedding dimensions (default: 1536, supported: 768, 1536, 3072)
        """
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.dimensions = dimensions

    @standard_retry
    async def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text with retry logic.

        :param text: Text to embed
        :return: Embedding vector as list of floats
        :raises Exception: If embedding fails after retries
        """
        try:
            response = self.client.models.embed_content(
                model=self.model,
                contents=text,
                config=types.EmbedContentConfig(output_dimensionality=self.dimensions),
            )
            return response.embeddings[0].values
        except Exception as e:
            logger.error("embedding_failed", text_length=len(text), error=str(e))
            raise

    @standard_retry
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in a batch.

        Note: Google Gemini API supports batch embedding via the contents parameter.

        :param texts: List of texts to embed
        :return: List of embedding vectors
        :raises Exception: If batch embedding fails after retries
        :raises ValueError: If batch is empty
        """
        if not texts:
            raise ValueError("Batch cannot be empty")

        try:
            response = self.client.models.embed_content(
                model=self.model,
                contents=texts,
                config=types.EmbedContentConfig(output_dimensionality=self.dimensions),
            )
            return [emb.values for emb in response.embeddings]
        except Exception as e:
            logger.error("batch_embedding_failed", batch_size=len(texts), error=str(e))
            raise
