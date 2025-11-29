from abc import abstractmethod

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.conversation import Conversation, ConversationType
from app.models.conversation_participant import ConversationParticipant
from app.repositories.base_repository import BaseRepository


class IConversationRepository(BaseRepository[Conversation]):
    """Abstract interface for Conversation repository."""

    @abstractmethod
    async def create_private_conversation(self, room_id: int, user_ids: list[int], ai_ids: list[int]) -> Conversation:
        """Create a private conversation (2 participants: human and/or AI)."""
        pass

    @abstractmethod
    async def create_group_conversation(self, room_id: int, user_ids: list[int], ai_ids: list[int]) -> Conversation:
        """Create a group conversation (2+ participants: human and/or AI)."""
        pass

    @abstractmethod
    async def add_participant(
        self,
        conversation_id: int,
        user_id: int | None = None,
        ai_entity_id: int | None = None,
    ) -> ConversationParticipant:
        """Add participant (human or AI) to conversation."""
        pass

    @abstractmethod
    async def remove_participant(
        self,
        conversation_id: int,
        user_id: int | None = None,
        ai_entity_id: int | None = None,
    ) -> bool:
        """Remove participant (human or AI) from conversation (set left_at)."""
        pass

    @abstractmethod
    async def is_participant(self, conversation_id: int, user_id: int) -> bool:
        """Check if user is active participant in conversation."""
        pass

    @abstractmethod
    async def get_participants(self, conversation_id: int) -> list[ConversationParticipant]:
        """Get all active participants in conversation (polymorphic: User + AI)."""
        pass

    @abstractmethod
    async def get_user_conversations(self, user_id: int) -> list[Conversation]:
        """Get all active conversations for a user."""
        pass

    @abstractmethod
    async def get_room_conversations(self, room_id: int) -> list[Conversation]:
        """Get all active conversations in a room."""
        pass

    @abstractmethod
    async def get_active_conversations_for_ai(self, ai_entity_id: int) -> list[Conversation]:
        """Get all active conversations where AI is still a participant."""
        pass

    @abstractmethod
    async def count_active_participants(self, conversation_id: int) -> int:
        """Count active participants in conversation."""
        pass


class ConversationRepository(IConversationRepository):
    """SQLAlchemy implementation of Conversation repository."""

    def __init__(self, db: AsyncSession):
        """
        Initialize with async database session.
        :param db: SQLAlchemy async database session
        """
        super().__init__(db)

    async def get_by_id(self, id: int) -> Conversation | None:
        """Get conversation by ID."""
        query = select(Conversation).where(and_(Conversation.id == id, Conversation.is_active.is_(True)))
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def _create_conversation_base(
        self,
        room_id: int,
        user_ids: list[int],
        ai_ids: list[int],
        conversation_type: ConversationType,
        max_participants: int | None,
    ) -> Conversation:
        """
        Shared helper for creating conversations with participants.

        :param room_id: Room ID
        :param user_ids: List of user IDs
        :param ai_ids: List of AI entity IDs
        :param conversation_type: Type of conversation (PRIVATE or GROUP)
        :param max_participants: Maximum participants (2 for PRIVATE, None for GROUP)
        :return: Created conversation with participants
        """
        new_conversation = Conversation(
            room_id=room_id,
            conversation_type=conversation_type,
            max_participants=max_participants,
        )

        self.db.add(new_conversation)
        await self.db.flush()

        # Add human participants
        for user_id in user_ids:
            participant = ConversationParticipant(conversation_id=new_conversation.id, user_id=user_id)
            self.db.add(participant)

        # Add AI participants
        for ai_id in ai_ids:
            participant = ConversationParticipant(conversation_id=new_conversation.id, ai_entity_id=ai_id)
            self.db.add(participant)

        await self.db.commit()
        await self.db.refresh(new_conversation)
        return new_conversation

    async def create_private_conversation(self, room_id: int, user_ids: list[int], ai_ids: list[int]) -> Conversation:
        """Create a private conversation (2 participants: human and/or AI)."""
        total_participants = len(user_ids) + len(ai_ids)

        if total_participants != 2:
            raise ValueError(f"Private conversations require exactly 2 participants, got {total_participants}")

        if len(user_ids) < 1:
            raise ValueError("Private conversations require at least 1 human participant")

        return await self._create_conversation_base(
            room_id=room_id,
            user_ids=user_ids,
            ai_ids=ai_ids,
            conversation_type=ConversationType.PRIVATE,
            max_participants=2,
        )

    async def create_group_conversation(self, room_id: int, user_ids: list[int], ai_ids: list[int]) -> Conversation:
        """Create a group conversation (2+ participants: human and/or AI)."""
        total_participants = len(user_ids) + len(ai_ids)

        if total_participants < 2:
            raise ValueError(f"Group conversations require at least 2 participants, got {total_participants}")

        if len(user_ids) < 1:
            raise ValueError("Group conversations require at least 1 human participant")

        return await self._create_conversation_base(
            room_id=room_id,
            user_ids=user_ids,
            ai_ids=ai_ids,
            conversation_type=ConversationType.GROUP,
            max_participants=None,
        )

    async def add_participant(
        self,
        conversation_id: int,
        user_id: int | None = None,
        ai_entity_id: int | None = None,
    ) -> ConversationParticipant:
        """Add participant (human or AI) to conversation."""
        # XOR validation: exactly one must be set
        if (user_id is None) == (ai_entity_id is None):
            raise ValueError("Exactly one of user_id or ai_entity_id must be provided")

        # Check if already participant (human or AI)
        if user_id:
            if await self.is_participant(conversation_id, user_id):
                raise ValueError("User is already a participant in this conversation")
        elif ai_entity_id:
            # Check if AI already in conversation
            query = select(ConversationParticipant).where(
                and_(
                    ConversationParticipant.conversation_id == conversation_id,
                    ConversationParticipant.ai_entity_id == ai_entity_id,
                    ConversationParticipant.left_at.is_(None),
                )
            )
            result = await self.db.execute(query)
            if result.scalar_one_or_none():
                raise ValueError("AI is already a participant in this conversation")

        # Create participant
        participant = ConversationParticipant(
            conversation_id=conversation_id,
            user_id=user_id,
            ai_entity_id=ai_entity_id,
        )

        self.db.add(participant)
        await self.db.commit()
        await self.db.refresh(participant)
        return participant

    async def remove_participant(
        self,
        conversation_id: int,
        user_id: int | None = None,
        ai_entity_id: int | None = None,
    ) -> bool:
        """Remove participant (human or AI) from conversation (set left_at)."""
        # XOR validation: exactly one must be set
        if (user_id is None) == (ai_entity_id is None):
            raise ValueError("Exactly one of user_id or ai_entity_id must be provided")

        # Build query based on participant type
        conditions = [
            ConversationParticipant.conversation_id == conversation_id,
            ConversationParticipant.left_at.is_(None),
        ]

        if user_id:
            conditions.append(ConversationParticipant.user_id == user_id)
        else:
            conditions.append(ConversationParticipant.ai_entity_id == ai_entity_id)

        participant_query = select(ConversationParticipant).where(and_(*conditions))
        result = await self.db.execute(participant_query)
        participant = result.scalar_one_or_none()

        if participant:
            from datetime import datetime

            participant.left_at = datetime.now()
            await self.db.commit()

            # Auto-archive if no active participants left
            active_count = await self.count_active_participants(conversation_id)
            if active_count == 0:
                conversation = await self.get_by_id(conversation_id)
                if conversation:
                    conversation.is_active = False
                    await self.db.commit()

            return True
        return False

    async def is_participant(self, conversation_id: int, user_id: int) -> bool:
        """Check if user is active participant in conversation."""
        participant_query = select(ConversationParticipant).where(
            and_(
                ConversationParticipant.conversation_id == conversation_id,
                ConversationParticipant.user_id == user_id,
                ConversationParticipant.left_at.is_(None),
            )
        )
        result = await self.db.execute(participant_query)
        participant = result.scalar_one_or_none()
        return participant is not None

    async def get_participants(self, conversation_id: int) -> list[ConversationParticipant]:
        """Get all active participants in conversation (polymorphic: User + AI)."""
        participants_query = (
            select(ConversationParticipant)
            .options(selectinload(ConversationParticipant.user), selectinload(ConversationParticipant.ai_entity))
            .where(
                and_(
                    ConversationParticipant.conversation_id == conversation_id,
                    ConversationParticipant.left_at.is_(None),
                )
            )
        )

        result = await self.db.execute(participants_query)
        return list(result.scalars().all())

    async def get_user_conversations(self, user_id: int) -> list[Conversation]:
        """Get all active conversations for a user."""
        conversations_query = (
            select(Conversation)
            .join(
                ConversationParticipant,
                and_(
                    ConversationParticipant.conversation_id == Conversation.id,
                    ConversationParticipant.user_id == user_id,
                    ConversationParticipant.left_at.is_(None),
                ),
            )
            .where(Conversation.is_active.is_(True))
        )

        result = await self.db.execute(conversations_query)
        return list(result.scalars().all())

    async def get_room_conversations(self, room_id: int) -> list[Conversation]:
        """Get all active conversations in a room."""
        query = select(Conversation).where(and_(Conversation.room_id == room_id, Conversation.is_active.is_(True)))
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_active_conversations_for_ai(self, ai_entity_id: int) -> list[Conversation]:
        """
        Get all active conversations where AI is still a participant.

        Returns conversations where AI has not left yet (left_at IS NULL).
        """
        conversations_query = (
            select(Conversation)
            .join(
                ConversationParticipant,
                and_(
                    ConversationParticipant.conversation_id == Conversation.id,
                    ConversationParticipant.ai_entity_id == ai_entity_id,
                    ConversationParticipant.left_at.is_(None),
                ),
            )
            .where(Conversation.is_active.is_(True))
        )

        result = await self.db.execute(conversations_query)
        return list(result.scalars().all())

    async def get_all(self, limit: int = 100, offset: int = 0) -> list[Conversation]:
        """Get all conversations with pagination."""
        query = select(Conversation).limit(limit).offset(offset)
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def delete(self, id: int) -> bool:
        """Soft delete conversation (set inactive)."""
        conversation = await self.get_by_id(id)
        if conversation:
            conversation.is_active = False
            await self.db.commit()
            return True
        return False

    async def exists(self, id: int) -> bool:
        """Check if conversation exists by ID (active conversations only)."""
        return await self._check_exists_where(and_(Conversation.id == id, Conversation.is_active.is_(True)))

    async def count_active_participants(self, conversation_id: int) -> int:
        """Count active participants in conversation."""
        from sqlalchemy import func

        query = (
            select(func.count())
            .select_from(ConversationParticipant)
            .where(
                and_(
                    ConversationParticipant.conversation_id == conversation_id,
                    ConversationParticipant.left_at.is_(None),
                )
            )
        )
        result = await self.db.execute(query)
        return result.scalar() or 0
