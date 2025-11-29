import structlog

from app.interfaces.translator import TranslationError, TranslatorInterface
from app.models.message_translation import MessageTranslation
from app.models.user import User
from app.repositories.message_repository import IMessageRepository
from app.repositories.message_translation_repository import (
    IMessageTranslationRepository,
)

logger = structlog.get_logger(__name__)


class TranslationService:
    """Service for text translation with dependency injection."""

    def __init__(
        self,
        translator: TranslatorInterface,
        message_repo: IMessageRepository,
        translation_repo: IMessageTranslationRepository,
    ):
        self.translator = translator
        self.message_repo = message_repo
        self.translation_repo = translation_repo

    @staticmethod
    def get_target_languages_from_users(users: list[User], current_user: User) -> list[str]:
        """
        Extract unique target languages from a list of users.

        Filters out:
        - Users without preferred_language
        - Users with same language as current_user
        - The current_user themselves

        :param users: List of User objects
        :param current_user: Current authenticated user
        :return: List of unique uppercase language codes
        """
        return list(
            {
                user.preferred_language.upper()
                for user in users
                if user.preferred_language
                and user.preferred_language != current_user.preferred_language
                and user.id != current_user.id
            }
        )

    async def translate_message_content(
        self,
        content: str,
        source_language: str | None = None,
        target_languages: list[str] | None = None,
    ) -> dict[str, str]:
        """
        Translate message content to multiple target languages.

        :param content: Original message content
        :param source_language: Source language (auto-detect if None)
        :param target_languages: List of target language codes
        :return: Dictionary mapping language codes to translated content
        """
        if not target_languages:
            logger.debug("No target languages specified - skipping translation")
            return {}

        if not content or not content.strip():
            logger.debug("Empty content - skipping translation")
            return {}

        try:
            return await self.translator.translate_to_multiple_languages(
                text=content, target_languages=target_languages, source_language=source_language
            )
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return {}

    async def create_message_translations(
        self, message_id: int, translations: dict[str, str]
    ) -> list[MessageTranslation]:
        """
        Store translations in database.
        :param message_id: ID of the original message
        :param translations: Dictionary mapping language codes to translated content
        :return: List of created MessageTranslation objects
        """
        if not translations:
            return []

        translation_objects = []

        for target_language, translated_content in translations.items():
            try:
                translation = MessageTranslation(
                    message_id=message_id,
                    target_language=target_language,
                    content=translated_content,
                )

                translation_objects.append(translation)

            except ValueError as e:
                logger.error(f"Failed to create translation for {target_language}: {e}")
                continue

        if translation_objects:
            created_translations = await self.translation_repo.bulk_create_translations(translation_objects)

            if created_translations:
                print(f"✅ Saved {len(created_translations)} translations to database")
            else:
                print("❌ Failed to save translations to database")

            return created_translations

        return []

    async def translate_and_store_message(
        self,
        message_id: int,
        content: str,
        source_language: str | None = None,
        target_languages: list[str] | None = None,
    ) -> int:
        """
        Complete translation workflow: translate content and store in database.
        :param message_id: ID of the message to translate
        :param content: Original message content
        :param source_language: Source language code (auto-detect if None)
        :param target_languages: Target language codes
        :return: Number of successful translations created
        """
        try:
            translations = await self.translate_message_content(
                content=content,
                source_language=source_language,
                target_languages=target_languages,
            )

            if not translations:
                print(f"No translations created for message {message_id}")
                return 0

            translation_objects = await self.create_message_translations(
                message_id=message_id, translations=translations
            )

            print(f"Created {len(translation_objects)} translations for message {message_id}")
            return len(translation_objects)

        except (TranslationError, ValueError, RuntimeError) as e:
            logger.error(f"Translation workflow failed for message {message_id}: {e}")
            return 0

    async def get_message_translation(self, message_id: int, target_language: str) -> str | None:
        """
        Retrieve specific translation via repository.
        :param message_id: ID of the original message
        :param target_language: Target language code
        :return: Translated content or None if not found
        """
        translation = await self.translation_repo.get_by_message_and_language(
            message_id=message_id, target_language=target_language.upper()
        )

        return translation.content if translation else None

    async def get_all_message_translations(self, message_id: int) -> dict[str, str]:
        """
        Get all translations for a message.
        :param message_id: ID of the original message
        :return: Dictionary mapping language codes to translated content
        """
        translations = await self.translation_repo.get_by_message_id(message_id)

        return {translation.target_language: translation.content for translation in translations}

    async def delete_message_translations(self, message_id: int) -> int:
        """
        Delete all translations for a message.
        :param message_id: ID of the message
        :return: Number of translations deleted
        """
        return await self.translation_repo.delete_by_message_id(message_id)
