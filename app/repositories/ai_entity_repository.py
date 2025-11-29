from abc import abstractmethod

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.ai_entity import AIEntity, AIEntityStatus

from .base_repository import BaseRepository


class IAIEntityRepository(BaseRepository[AIEntity]):
    """Interface for AI Entity repository."""

    @abstractmethod
    async def get_by_username(self, username: str) -> AIEntity | None:
        """Get AI entity by unique username."""
        pass

    @abstractmethod
    async def get_available_entities(self) -> list[AIEntity]:
        """Get all available AI entities (online and not deleted)."""
        pass

    @abstractmethod
    async def username_exists(self, username: str, exclude_id: int | None = None) -> bool:
        """Check if username exists (for validation)."""
        pass

    @abstractmethod
    async def get_available_in_room(self, room_id: int) -> list[AIEntity]:
        """Get available AIs in specific room (optimized with JOIN)."""
        pass

    @abstractmethod
    async def get_ai_in_conversation(self, conversation_id: int) -> AIEntity | None:
        """Get AI entity in specific conversation (optimized with JOIN)."""
        pass

    @abstractmethod
    async def get_ai_in_room(self, room_id: int) -> AIEntity | None:
        """Get AI entity currently assigned to a specific room."""
        pass


class AIEntityRepository(IAIEntityRepository):
    """SQLAlchemy implementation of AI Entity repository."""

    def __init__(self, db: AsyncSession):
        super().__init__(db)

    async def get_by_id(self, id: int) -> AIEntity | None:
        query = (
            select(AIEntity).options(selectinload(AIEntity.current_room)).where(AIEntity.id == id, AIEntity.is_active)
        )
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_by_username(self, username: str) -> AIEntity | None:
        query = select(AIEntity).where(AIEntity.username == username, AIEntity.is_active)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_all(self, limit: int = 100, offset: int = 0) -> list[AIEntity]:
        query = (
            select(AIEntity)
            .options(selectinload(AIEntity.current_room))
            .where(AIEntity.is_active)
            .limit(limit)
            .offset(offset)
        )
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_available_entities(self) -> list[AIEntity]:
        """Get all available AI entities (online and not deleted)."""
        query = (
            select(AIEntity)
            .options(selectinload(AIEntity.current_room))
            .where(AIEntity.status == AIEntityStatus.ONLINE, AIEntity.is_active)
        )
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def delete(self, id: int) -> bool:
        """Soft delete - set is_active to False and status to OFFLINE."""
        query = select(AIEntity).where(AIEntity.id == id)
        result = await self.db.execute(query)
        entity = result.scalar_one_or_none()

        if entity:
            entity.is_active = False
            entity.status = AIEntityStatus.OFFLINE
            await self.db.commit()
            return True
        return False

    async def set_status(self, id: int, status: AIEntityStatus) -> bool:
        """Change AI entity status (ONLINE/OFFLINE)."""
        entity = await self.get_by_id(id)
        if entity:
            entity.status = status
            await self.db.commit()
            return True
        return False

    async def exists(self, id: int) -> bool:
        """Check if AI entity exists by ID (active entities only)."""
        return await self._check_exists_where(and_(AIEntity.id == id, AIEntity.is_active))

    async def username_exists(self, username: str, exclude_id: int | None = None) -> bool:
        """Check if username exists, optionally excluding specific entity."""
        where_clauses = [AIEntity.username == username]
        if exclude_id:
            where_clauses.append(AIEntity.id != exclude_id)
        return await self._check_exists_where(*where_clauses)

    async def get_available_in_room(self, room_id: int) -> list[AIEntity]:
        """
        Get available AIs in specific room.

        Available = ACTIVE status + not in any active conversation + in this room
        Uses LEFT JOIN to check conversation participation efficiently.
        """
        from app.models.conversation_participant import ConversationParticipant

        query = (
            select(AIEntity)
            .outerjoin(
                ConversationParticipant,
                and_(
                    ConversationParticipant.ai_entity_id == AIEntity.id,
                    ConversationParticipant.left_at.is_(None),
                ),
            )
            .where(
                and_(
                    AIEntity.current_room_id == room_id,
                    AIEntity.status == AIEntityStatus.ONLINE,
                    AIEntity.is_active,
                    ConversationParticipant.id.is_(None),
                )
            )
        )
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_ai_in_conversation(self, conversation_id: int) -> AIEntity | None:
        """
        Get AI entity in specific conversation (optimized with JOIN).

        Returns None if no AI in conversation.
        """
        from app.models.conversation_participant import ConversationParticipant

        query = (
            select(AIEntity)
            .join(
                ConversationParticipant,
                ConversationParticipant.ai_entity_id == AIEntity.id,
            )
            .where(
                and_(
                    ConversationParticipant.conversation_id == conversation_id,
                    ConversationParticipant.left_at.is_(None),
                )
            )
        )
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_ai_in_room(self, room_id: int) -> AIEntity | None:
        """
        Get AI entity currently assigned to a specific room.

        Returns active AI entity with current_room_id == room_id and status == ONLINE.
        Returns None if no AI assigned to room.
        """
        query = select(AIEntity).where(
            and_(
                AIEntity.current_room_id == room_id,
                AIEntity.status == AIEntityStatus.ONLINE,
                AIEntity.is_active,
            )
        )
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
