from abc import ABC, abstractmethod


class EmbeddingServiceError(Exception):
    """Exception raised when embedding generation fails."""

    def __init__(self, message: str, original_error: Exception | None = None):
        """
        Initialize embedding service error.

        :param message: Error message
        :param original_error: Original exception that caused this error
        """
        super().__init__(message)
        self.original_error = original_error


class IEmbeddingService(ABC):
    """Interface for text embedding services."""

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        :param text: Text to embed
        :return: List of floats representing the embedding vector
        """
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in a batch.

        :param texts: List of texts to embed (max 100)
        :return: List of embedding vectors
        """
        pass
