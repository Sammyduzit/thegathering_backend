"""
Keyword-based Memory Retriever implementation.

Provides keyword-based memory retrieval from database using repository pattern.
Future-proof design allows easy extension with vector search.
"""

import structlog

from app.core.config import settings
from app.interfaces.memory_retriever import IMemoryRetriever, MemoryRetrievalError
from app.models.ai_memory import AIMemory
from app.repositories.ai_memory_repository import IAIMemoryRepository

logger = structlog.get_logger(__name__)


class KeywordMemoryRetriever(IMemoryRetriever):
    """Keyword-based memory retriever implementation."""

    def __init__(self, memory_repo: IAIMemoryRepository):
        """
        Initialize keyword retriever with memory repository.

        :param memory_repo: AI memory repository instance
        """
        self.memory_repo = memory_repo

    async def retrieve_candidates(
        self,
        entity_id: int,
        query: str | None = None,
        keywords: list[str] | None = None,
        limit: int = 20,
    ) -> list[AIMemory]:
        """
        Retrieve memory candidates using keyword-based filtering.

        Strategy:
        1. If keywords provided: Use repository's search_by_keywords()
        2. Else: Use repository's get_entity_memories() (importance + recency)

        :param entity_id: AI entity ID to retrieve memories for
        :param query: Optional query text (ignored by keyword retriever, used by vector retrievers)
        :param keywords: Optional keywords for keyword-based filtering
        :param limit: Maximum number of candidates to retrieve (default: 20)
        :return: List of memory candidates ordered by relevance; with keywords ordered by importance_score, otherwise by importance_score DESC then created_at DESC
        :raises MemoryRetrievalError: If retrieval fails
        """
        try:
            if keywords:
                # Keyword-based search
                memories = await self.memory_repo.search_by_keywords(
                    entity_id=entity_id,
                    keywords=keywords,
                    limit=limit,
                )

                logger.debug(
                    "retrieved_memories_by_keywords",
                    entity_id=entity_id,
                    keywords=keywords,
                    count=len(memories),
                )
            else:
                # Fallback: Get all memories ordered by importance + recency
                memories = await self.memory_repo.get_entity_memories(
                    entity_id=entity_id,
                    room_id=None,  # Get all, not room-specific
                    limit=limit,
                )

                logger.debug(
                    "retrieved_memories_all",
                    entity_id=entity_id,
                    count=len(memories),
                )

            return memories

        except Exception as e:
            logger.error("memory_retrieval_failed", entity_id=entity_id, error=str(e))
            raise MemoryRetrievalError(f"Failed to retrieve memories: {str(e)}", original_error=e)

    async def retrieve_tiered(
        self,
        entity_id: int,
        user_id: int,
        conversation_id: int | None,
        query: str,
    ) -> list[AIMemory]:
        """
        Tiered retrieval using importance + recency (no vector search).

        Retrieves memories from all layers based on importance_score and creation time.
        Simple fallback when vector search is disabled.

        :param entity_id: AI entity ID
        :param user_id: User ID for user-specific memories
        :param conversation_id: Conversation ID for short-term context
        :param query: Query text (unused by keyword retriever)
        :return: Mixed memories from all layers, ordered by importance and recency
        """
        try:
            # Retrieve from each layer separately
            short_term = await self._retrieve_layer(
                entity_id=entity_id,
                user_id=user_id,
                conversation_id=conversation_id,
                memory_type="short_term",
                limit=settings.short_term_candidates,
            )

            long_term = await self._retrieve_layer(
                entity_id=entity_id,
                user_id=user_id,
                conversation_id=None,  # Long-term crosses conversations
                memory_type="long_term",
                limit=settings.long_term_candidates,
            )

            personality = await self._retrieve_layer(
                entity_id=entity_id,
                user_id=None,  # Personality is global
                conversation_id=None,
                memory_type="personality",
                limit=settings.personality_candidates,
            )

            # Simple combination with guaranteed minimums
            final_memories = self._combine_layers(
                short_term=short_term,
                long_term=long_term,
                personality=personality,
                total_limit=settings.total_memory_limit,
            )

            logger.debug(
                "tiered_retrieval_completed",
                entity_id=entity_id,
                user_id=user_id,
                conversation_id=conversation_id,
                count=len(final_memories),
            )

            return final_memories

        except Exception as e:
            logger.error("tiered_retrieval_failed", entity_id=entity_id, error=str(e))
            raise MemoryRetrievalError(f"Failed tiered retrieval: {str(e)}", original_error=e)

    async def _retrieve_layer(
        self,
        entity_id: int,
        user_id: int | None,
        conversation_id: int | None,
        memory_type: str,
        limit: int,
    ) -> list[AIMemory]:
        """
        Retrieve memories from a specific layer using repository filters.

        :param entity_id: AI entity ID
        :param user_id: User ID filter (None for global)
        :param conversation_id: Conversation ID filter (None for cross-conversation)
        :param memory_type: Layer type (short_term, long_term, personality)
        :param limit: Maximum memories to retrieve
        :return: List of memories from the layer
        """
        # Get all memories for entity, then filter
        all_memories = await self.memory_repo.get_entity_memories(
            entity_id=entity_id,
            room_id=None,
            limit=limit * 3,  # Over-fetch for filtering
        )

        # Filter by type
        filtered = [m for m in all_memories if m.memory_metadata and m.memory_metadata.get("type") == memory_type]

        # Apply user filter
        if user_id is not None:
            filtered = [m for m in filtered if user_id in m.user_ids]

        # Apply conversation filter
        if conversation_id is not None:
            filtered = [m for m in filtered if m.conversation_id == conversation_id]

        return filtered[:limit]

    def _combine_layers(
        self,
        short_term: list[AIMemory],
        long_term: list[AIMemory],
        personality: list[AIMemory],
        total_limit: int,
    ) -> list[AIMemory]:
        """
        Combine layers with guaranteed minimums, ordered by importance.

        Strategy:
        1. Take guaranteed minimums from each layer
        2. Pool remaining candidates
        3. Sort pool by importance_score
        4. Fill up to total_limit

        :param short_term: Short-term memories
        :param long_term: Long-term memories
        :param personality: Personality memories
        :param total_limit: Maximum total memories
        :return: Combined and limited memory list
        """
        guaranteed = []
        pool = []

        # Extract guaranteed short-term
        if settings.guaranteed_short_term > 0 and short_term:
            guaranteed.extend(short_term[: settings.guaranteed_short_term])
            pool.extend(short_term[settings.guaranteed_short_term :])
        else:
            pool.extend(short_term)

        # Extract guaranteed long-term
        if settings.guaranteed_long_term > 0 and long_term:
            guaranteed.extend(long_term[: settings.guaranteed_long_term])
            pool.extend(long_term[settings.guaranteed_long_term :])
        else:
            pool.extend(long_term)

        # Extract guaranteed personality
        if settings.guaranteed_personality > 0 and personality:
            guaranteed.extend(personality[: settings.guaranteed_personality])
            pool.extend(personality[settings.guaranteed_personality :])
        else:
            pool.extend(personality)

        # Calculate remaining budget
        remaining_budget = total_limit - len(guaranteed)
        if remaining_budget <= 0:
            return guaranteed[:total_limit]

        # Sort pool by importance_score (descending)
        pool_sorted = sorted(
            pool,
            key=lambda m: (m.importance_score or 0.0),
            reverse=True,
        )

        # Combine guaranteed + top of pool
        final = guaranteed + pool_sorted[:remaining_budget]

        return final[:total_limit]
