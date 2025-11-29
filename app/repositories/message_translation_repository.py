from abc import abstractmethod

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.message_translation import MessageTranslation
from app.repositories.base_repository import BaseRepository


class IMessageTranslationRepository(BaseRepository[MessageTranslation]):
    """Abstract interface for MessageTranslation repository."""

    @abstractmethod
    async def create_translation(self, message_id: int, target_language: str, content: str) -> MessageTranslation:
        """Create a new message translation."""
        pass

    @abstractmethod
    async def get_by_message_and_language(self, message_id: int, target_language: str) -> MessageTranslation | None:
        """Get translation for specific message and language."""
        pass

    @abstractmethod
    async def get_by_message_id(self, message_id: int) -> list[MessageTranslation]:
        """Get all translations for a message."""
        pass

    @abstractmethod
    async def delete_by_message_id(self, message_id: int) -> int:
        """Delete all translations for a message."""
        pass

    @abstractmethod
    async def bulk_create_translations(self, translations: list[MessageTranslation]) -> list[MessageTranslation]:
        """Create multiple translations in one transaction."""
        pass

    @abstractmethod
    async def cleanup_old_translations(self, days_old: int) -> int:
        """Delete translations older than specified days."""
        pass


class MessageTranslationRepository(IMessageTranslationRepository):
    """SQLAlchemy implementation of MessageTranslation repository."""

    def __init__(self, db: AsyncSession):
        """
        Initialize with async database session.
        :param db: SQLAlchemy async database session
        """
        super().__init__(db)

    async def get_by_id(self, id: int) -> MessageTranslation | None:
        """Get message translation by ID."""
        query = select(MessageTranslation).where(MessageTranslation.id == id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def create_translation(self, message_id: int, target_language: str, content: str) -> MessageTranslation:
        """Create a new message translation."""
        new_translation = MessageTranslation(
            message_id=message_id,
            target_language=target_language.upper(),
            content=content,
        )

        self.db.add(new_translation)
        await self.db.commit()
        await self.db.refresh(new_translation)
        return new_translation

    async def get_by_message_and_language(self, message_id: int, target_language: str) -> MessageTranslation | None:
        """Get translation for specific message and language."""
        query = select(MessageTranslation).where(
            and_(
                MessageTranslation.message_id == message_id,
                MessageTranslation.target_language == target_language.upper(),
            )
        )
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_by_message_id(self, message_id: int) -> list[MessageTranslation]:
        """Get all translations for a message."""
        query = (
            select(MessageTranslation)
            .where(MessageTranslation.message_id == message_id)
            .order_by(MessageTranslation.target_language)
        )
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def delete_by_message_id(self, message_id: int) -> int:
        """Delete all translations for a message."""
        translations = await self.get_by_message_id(message_id)
        deleted_count = len(translations)

        for translation in translations:
            self.db.delete(translation)

        if deleted_count > 0:
            await self.db.commit()

        return deleted_count

    async def bulk_create_translations(self, translations: list[MessageTranslation]) -> list[MessageTranslation]:
        """Create multiple translations in one transaction."""
        if not translations:
            return []

        try:
            for translation in translations:
                self.db.add(translation)

            await self.db.commit()

            # Refresh all objects
            for translation in translations:
                await self.db.refresh(translation)

            return translations

        except Exception as e:
            await self.db.rollback()
            print(f"Failed to bulk create translations: {e}")
            return []

    async def get_all(self, limit: int = 100, offset: int = 0) -> list[MessageTranslation]:
        """Get all message translations with pagination."""
        query = select(MessageTranslation).limit(limit).offset(offset).order_by(MessageTranslation.created_at.desc())
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def delete(self, id: int) -> bool:
        """Delete message translation by ID."""
        translation = await self.get_by_id(id)
        if translation:
            self.db.delete(translation)
            await self.db.commit()
            return True
        return False

    async def exists(self, id: int) -> bool:
        """Check if message translation exists by ID."""
        return await self._check_exists_where(MessageTranslation.id == id)

    async def cleanup_old_translations(self, days_old: int) -> int:
        """
        Delete translations older than specified days.
        :param days_old: Remove translations older than this many days
        :return: Number of deleted translations
        """
        from datetime import datetime, timedelta

        cutoff_date = datetime.now() - timedelta(days=days_old)

        query = select(MessageTranslation).where(MessageTranslation.created_at < cutoff_date)
        result = await self.db.execute(query)
        old_translations = list(result.scalars().all())

        deleted_count = len(old_translations)

        for translation in old_translations:
            self.db.delete(translation)

        if deleted_count > 0:
            await self.db.commit()

        return deleted_count
