from abc import abstractmethod
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.ai_cooldown import AICooldown

from .base_repository import BaseRepository


class IAICooldownRepository(BaseRepository[AICooldown]):
    """Interface for AI Cooldown repository."""

    @abstractmethod
    async def get_cooldown(
        self,
        ai_entity_id: int,
        room_id: int | None = None,
        conversation_id: int | None = None,
    ) -> AICooldown | None:
        """Get cooldown for specific AI entity in room or conversation."""
        pass

    @abstractmethod
    async def upsert_cooldown(
        self,
        ai_entity_id: int,
        room_id: int | None = None,
        conversation_id: int | None = None,
    ) -> AICooldown:
        """Atomic upsert of cooldown timestamp."""
        pass

    @abstractmethod
    async def is_on_cooldown(
        self,
        ai_entity_id: int,
        cooldown_seconds: int,
        room_id: int | None = None,
        conversation_id: int | None = None,
    ) -> bool:
        """Check if AI entity is currently on cooldown in given context."""
        pass


class AICooldownRepository(IAICooldownRepository):
    """SQLAlchemy implementation of AI Cooldown repository."""

    def __init__(self, db: AsyncSession):
        super().__init__(db)

    async def get_by_id(self, id: int) -> AICooldown | None:
        query = select(AICooldown).where(AICooldown.id == id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_all(self, limit: int = 100, offset: int = 0) -> list[AICooldown]:
        query = select(AICooldown).limit(limit).offset(offset)
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_cooldown(
        self,
        ai_entity_id: int,
        room_id: int | None = None,
        conversation_id: int | None = None,
    ) -> AICooldown | None:
        """Get cooldown for specific AI entity in room or conversation."""
        query = (
            select(AICooldown)
            .where(
                AICooldown.ai_entity_id == ai_entity_id,
                AICooldown.room_id == room_id,
                AICooldown.conversation_id == conversation_id,
            )
            # Defensive ordering: SQLite allows multiple NULLs in UNIQUE constraints,
            # so pick the most recent if duplicates ever exist.
            .order_by(AICooldown.last_response_at.desc())
        )
        result = await self.db.execute(query)
        return result.scalars().first()

    async def upsert_cooldown(
        self,
        ai_entity_id: int,
        room_id: int | None = None,
        conversation_id: int | None = None,
    ) -> AICooldown:
        """
        Atomic upsert of cooldown timestamp using PostgreSQL ON CONFLICT.

        Uses database-level UPSERT to prevent race conditions under high concurrency.
        The unique constraint (ai_entity_id, room_id, conversation_id) ensures atomicity.

        :param ai_entity_id: AI entity ID
        :param room_id: Room ID (XOR with conversation_id)
        :param conversation_id: Conversation ID (XOR with room_id)
        :return: Upserted AICooldown record
        """
        now = datetime.now(timezone.utc)

        # PostgreSQL UPSERT: INSERT ... ON CONFLICT DO UPDATE
        upsert_statement = insert(AICooldown).values(
            ai_entity_id=ai_entity_id,
            room_id=room_id,
            conversation_id=conversation_id,
            last_response_at=now,
        )

        # On conflict on unique constraint, update the timestamp
        upsert_statement = upsert_statement.on_conflict_do_update(
            constraint="uq_ai_cooldown_context",
            set_={"last_response_at": now},
        )

        await self.db.execute(upsert_statement)
        await self.db.commit()

        # Fetch and return the upserted row
        cooldown = await self.get_cooldown(ai_entity_id, room_id, conversation_id)
        if not cooldown:
            raise RuntimeError("Upsert succeeded but cooldown not found")
        return cooldown

    async def is_on_cooldown(
        self,
        ai_entity_id: int,
        cooldown_seconds: int,
        room_id: int | None = None,
        conversation_id: int | None = None,
    ) -> bool:
        """
        Check if AI entity is currently on cooldown in given context.

        :param ai_entity_id: AI entity ID
        :param cooldown_seconds: Cooldown duration in seconds
        :param room_id: Room ID (for room context)
        :param conversation_id: Conversation ID (for conversation context)
        :return: True if on cooldown (cannot respond yet), False otherwise
        """
        cooldown = await self.get_cooldown(ai_entity_id, room_id, conversation_id)

        if not cooldown:
            # No cooldown record exists - can respond
            return False

        # Calculate time elapsed since last response
        now = datetime.now(timezone.utc)
        elapsed_seconds = (now - cooldown.last_response_at).total_seconds()

        # On cooldown if elapsed time is less than required cooldown duration
        return elapsed_seconds < cooldown_seconds

    async def delete(self, id: int) -> bool:
        cooldown = await self.get_by_id(id)
        if cooldown:
            await self.db.delete(cooldown)
            await self.db.commit()
            return True
        return False

    async def exists(self, id: int) -> bool:
        """Check if cooldown record exists by ID."""
        return await self._check_exists_where(AICooldown.id == id)
