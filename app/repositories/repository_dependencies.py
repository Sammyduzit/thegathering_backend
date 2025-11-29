from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.repositories.ai_cooldown_repository import AICooldownRepository, IAICooldownRepository
from app.repositories.ai_entity_repository import AIEntityRepository, IAIEntityRepository
from app.repositories.ai_memory_repository import AIMemoryRepository, IAIMemoryRepository
from app.repositories.conversation_repository import (
    ConversationRepository,
    IConversationRepository,
)
from app.repositories.message_repository import IMessageRepository, MessageRepository
from app.repositories.message_translation_repository import (
    IMessageTranslationRepository,
    MessageTranslationRepository,
)
from app.repositories.room_repository import IRoomRepository, RoomRepository
from app.repositories.user_repository import IUserRepository, UserRepository


def get_user_repository(db: AsyncSession = Depends(get_db)) -> IUserRepository:
    """
    Create UserRepository instance with async database session.
    :param db: Async database session from get_db dependency
    :return: UserRepository instance
    """
    return UserRepository(db)


def get_room_repository(db: AsyncSession = Depends(get_db)) -> IRoomRepository:
    """
    Create RoomRepository instance with async database session.
    :param db: Async database session from get_db dependency
    :return: RoomRepository instance
    """
    return RoomRepository(db)


def get_message_repository(db: AsyncSession = Depends(get_db)) -> IMessageRepository:
    """
    Create MessageRepository instance with async database session.
    :param db: Async database session from get_db dependency
    :return: MessageRepository instance
    """
    return MessageRepository(db)


def get_conversation_repository(
    db: AsyncSession = Depends(get_db),
) -> IConversationRepository:
    """
    Create ConversationRepository instance with async database session.
    :param db: Async database session from get_db dependency
    :return: ConversationRepository instance
    """
    return ConversationRepository(db)


def get_message_translation_repository(
    db: AsyncSession = Depends(get_db),
) -> IMessageTranslationRepository:
    """
    Create MessageTranslationRepository instance with async database session.
    :param db: Async database session from get_db dependency
    :return: MessageTranslationRepository instance
    """
    return MessageTranslationRepository(db)


def get_ai_entity_repository(db: AsyncSession = Depends(get_db)) -> IAIEntityRepository:
    """
    Create AIEntityRepository instance with async database session.
    :param db: Async database session from get_db dependency
    :return: AIEntityRepository instance
    """
    return AIEntityRepository(db)


def get_ai_memory_repository(db: AsyncSession = Depends(get_db)) -> IAIMemoryRepository:
    """
    Create AIMemoryRepository instance with async database session.
    :param db: Async database session from get_db dependency
    :return: AIMemoryRepository instance
    """
    return AIMemoryRepository(db)


def get_ai_cooldown_repository(db: AsyncSession = Depends(get_db)) -> IAICooldownRepository:
    """
    Create AICooldownRepository instance with async database session.
    :param db: Async database session from get_db dependency
    :return: AICooldownRepository instance
    """
    return AICooldownRepository(db)
