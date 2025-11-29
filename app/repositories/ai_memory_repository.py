from abc import abstractmethod
from datetime import datetime, timedelta

from sqlalchemy import delete, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.ai_memory import AIMemory

from .base_repository import BaseRepository


class IAIMemoryRepository(BaseRepository[AIMemory]):
    """Interface for AI Memory repository."""

    @abstractmethod
    async def get_entity_memories(self, entity_id: int, room_id: int | None = None, limit: int = 10) -> list[AIMemory]:
        """Get memories for entity, optionally filtered by room."""
        pass

    @abstractmethod
    async def search_by_keywords(self, entity_id: int, keywords: list[str], limit: int = 5) -> list[AIMemory]:
        """Simple keyword-based memory search."""
        pass

    @abstractmethod
    async def vector_search(
        self,
        entity_id: int,
        embedding: list[float],
        user_id: int | None = None,
        conversation_id: int | None = None,
        exclude_conversation_id: int | None = None,
        memory_type: str | None = None,
        limit: int = 20,
    ) -> list[AIMemory]:
        """Vector similarity search using pgvector."""
        pass


class AIMemoryRepository(IAIMemoryRepository):
    """SQLAlchemy implementation of AI Memory repository."""

    def __init__(self, db: AsyncSession):
        super().__init__(db)

    async def get_by_id(self, id: int) -> AIMemory | None:
        query = select(AIMemory).where(AIMemory.id == id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_all(self, limit: int = 100, offset: int = 0) -> list[AIMemory]:
        query = select(AIMemory).limit(limit).offset(offset)
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_entity_memories(self, entity_id: int, room_id: int | None = None, limit: int = 10) -> list[AIMemory]:
        """Get recent memories for entity, ordered by importance and recency."""
        query = select(AIMemory).where(AIMemory.entity_id == entity_id)

        if room_id is not None:
            query = query.where(AIMemory.room_id == room_id)

        query = query.order_by(desc(AIMemory.importance_score), desc(AIMemory.created_at))
        query = query.limit(limit)

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def search_by_keywords(self, entity_id: int, keywords: list[str], limit: int = 5) -> list[AIMemory]:
        """
        Simple keyword matching.
        Returns memories ordered by importance score.
        """
        query = select(AIMemory).where(AIMemory.entity_id == entity_id)
        query = query.order_by(desc(AIMemory.importance_score))
        query = query.limit(limit * 3)  # Fetch more for filtering

        result = await self.db.execute(query)
        all_memories = list(result.scalars().all())

        # Simple keyword filtering in Python (Phase 2)
        # Phase 3: Move to database query with proper GIN index
        filtered = []
        for memory in all_memories:
            memory_keywords = memory.keywords or []
            if any(kw.lower() in [mk.lower() for mk in memory_keywords] for kw in keywords):
                filtered.append(memory)

        return filtered[:limit]

    async def delete(self, id: int) -> bool:
        """Hard delete for memories."""
        memory = await self.get_by_id(id)
        if memory:
            await self.db.delete(memory)
            await self.db.commit()
            return True
        return False

    async def exists(self, id: int) -> bool:
        """Check if memory exists by ID."""
        return await self._check_exists_where(AIMemory.id == id)

    async def vector_search(
        self,
        entity_id: int,
        embedding: list[float],
        user_id: int | None = None,
        conversation_id: int | None = None,
        exclude_conversation_id: int | None = None,
        memory_type: str | None = None,
        limit: int = 20,
    ) -> list[AIMemory]:
        """
        Vector similarity search using pgvector cosine distance.

        :param entity_id: AI entity ID
        :param embedding: Query embedding vector
        :param user_id: Filter by user (for short-term/long-term)
        :param conversation_id: Filter by conversation
        :param exclude_conversation_id: Exclude specific conversation
        :param memory_type: Filter by type (short_term, long_term, personality)
        :param limit: Maximum results
        :return: List of memories ordered by similarity (ascending distance)
        """
        query = select(AIMemory).where(AIMemory.entity_id == entity_id)

        # Filter by user_id if provided (PostgreSQL array containment check)
        if user_id is not None:
            query = query.where(AIMemory.user_ids.contains([user_id]))

        # Filter by conversation_id if provided
        if conversation_id is not None:
            query = query.where(AIMemory.conversation_id == conversation_id)

        # Exclude conversation if provided (NULL-safe: include personality memories with conversation_id=NULL)
        if exclude_conversation_id is not None:
            query = query.where(
                (AIMemory.conversation_id != exclude_conversation_id) | (AIMemory.conversation_id.is_(None))
            )

        # Filter by memory type if provided
        if memory_type is not None:
            query = query.where(AIMemory.memory_metadata["type"].as_string() == memory_type)

        # Order by cosine distance (ascending = most similar first)
        query = query.order_by(AIMemory.embedding.cosine_distance(embedding))
        query = query.limit(limit)

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def delete_old_short_term_memories(self, ttl_days: int) -> int:
        """
        Delete short-term memories older than TTL.

        :param ttl_days: Time-to-live in days (e.g., 7 for 7 days)
        :return: Number of deleted memories
        """
        cutoff_date = datetime.now() - timedelta(days=ttl_days)

        # Delete short-term memories older than cutoff
        stmt = delete(AIMemory).where(
            AIMemory.created_at < cutoff_date,
            AIMemory.memory_metadata["type"].as_string() == "short_term",
        )

        result = await self.db.execute(stmt)
        await self.db.commit()

        return result.rowcount or 0
