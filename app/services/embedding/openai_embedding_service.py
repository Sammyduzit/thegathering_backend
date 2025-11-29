import structlog
from openai import AsyncOpenAI

from app.core.decorators import standard_retry
from app.interfaces.embedding_service import IEmbeddingService

logger = structlog.get_logger(__name__)


class OpenAIEmbeddingService(IEmbeddingService):
    """OpenAI embedding service using text-embedding-3-small."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small", dimensions: int = 1536):
        """
        Initialize OpenAI embedding service.

        :param api_key: OpenAI API key
        :param model: Embedding model name
        :param dimensions: Embedding dimensions
        """
        self.client = AsyncOpenAI(api_key=api_key)
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
            response = await self.client.embeddings.create(
                input=text,
                model=self.model,
                dimensions=self.dimensions,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error("embedding_failed", text_length=len(text), error=str(e))
            raise

    @standard_retry
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in a batch (max 100).

        :param texts: List of texts to embed
        :return: List of embedding vectors
        :raises Exception: If batch embedding fails after retries
        :raises ValueError: If batch size exceeds 100
        """
        if len(texts) > 100:
            raise ValueError(f"Batch size {len(texts)} exceeds maximum of 100")

        try:
            response = await self.client.embeddings.create(
                input=texts,
                model=self.model,
                dimensions=self.dimensions,
            )
            # Sort by index to ensure order matches input
            sorted_data = sorted(response.data, key=lambda x: x.index)
            return [item.embedding for item in sorted_data]
        except Exception as e:
            logger.error("batch_embedding_failed", batch_size=len(texts), error=str(e))
            raise
