"""
Memory Retriever interface for retrieving AI memories from storage.

This interface abstracts memory retrieval functionality, allowing different
retrieval strategies (keyword-based, vector-based, hybrid) to be used
interchangeably through dependency injection.
"""

from abc import ABC, abstractmethod

from app.models.ai_memory import AIMemory


class IMemoryRetriever(ABC):
    """Abstract interface for AI memory retrieval services."""

    @abstractmethod
    async def retrieve_candidates(
        self,
        entity_id: int,
        query: str | None = None,
        keywords: list[str] | None = None,
        limit: int = 20,
    ) -> list[AIMemory]:
        """
        Retrieve memory candidates for further filtering/ranking.

        This method over-fetches candidates that can be filtered and ranked
        by subsequent verifier stages.

        :param entity_id: AI entity ID to retrieve memories for
        :param query: Optional query text for semantic search (used by vector retrievers)
        :param keywords: Optional keywords for keyword-based filtering
        :param limit: Maximum number of candidates to retrieve (default: 20)
        :return: List of memory candidates (may contain more than needed); memories are NOT pre-filtered by relevance
        :raises MemoryRetrievalError: If retrieval fails
        """
        pass

    @abstractmethod
    async def retrieve_tiered(
        self,
        entity_id: int,
        user_id: int,
        conversation_id: int | None,
        query: str,
    ) -> list[AIMemory]:
        """
        Tiered retrieval across short-term, long-term, and personality memory layers.

        Returns mixed memories from all 3 layers, ordered by relevance.

        :param entity_id: AI entity ID to retrieve memories for
        :param user_id: User ID for user-specific memories
        :param conversation_id: Conversation ID for short-term context
        :param query: Query text for retrieval
        :return: List of memories from all layers, ordered by relevance
        :raises MemoryRetrievalError: If retrieval fails
        """
        pass


class MemoryRetrievalError(Exception):
    """Exception raised when memory retrieval operations fail."""

    def __init__(self, message: str, original_error: Exception | None = None):
        super().__init__(message)
        self.original_error = original_error
