import structlog
from sqlalchemy.exc import SQLAlchemyError

from app.core.background_tasks import background_task_retry
from app.interfaces.translator import TranslationError
from app.models.message import Message
from app.models.message_translation import MessageTranslation
from app.repositories.message_translation_repository import IMessageTranslationRepository
from app.services.domain.translation_service import TranslationService

logger = structlog.get_logger(__name__)


class BackgroundService:
    """Service for handling background tasks like translations and notifications."""

    def __init__(
        self,
        translation_service: TranslationService,
        message_translation_repo: IMessageTranslationRepository,
    ):
        self.translation_service = translation_service
        self.message_translation_repo = message_translation_repo

    @background_task_retry(max_retries=2, delay=2.0)
    async def process_message_translation_background(
        self, message: Message, target_languages: list[str], room_translation_enabled: bool = True
    ) -> dict[str, str]:
        """
        Process message translation in background for multiple languages.
        :param message: Message to translate
        :param target_languages: List of target language codes
        :param room_translation_enabled: Whether room has translation enabled
        :return: Dictionary of language -> translated content
        """
        if not room_translation_enabled:
            logger.info("translation_disabled_for_room", message_id=message.id)
            return {}

        logger.info(
            "background_translation_started",
            message_id=message.id,
            target_language_count=len(target_languages),
        )

        translations = {}

        for target_lang in target_languages:
            try:
                # Check if translation already exists
                existing_translation = await self.message_translation_repo.get_by_message_and_language(
                    message.id, target_lang
                )

                if existing_translation:
                    translations[target_lang] = existing_translation.content
                    logger.info(
                        "existing_translation_used",
                        message_id=message.id,
                        target_language=target_lang,
                    )
                    continue

                # Create new translation
                translation_result = await self.translation_service.translate_message_content(
                    content=message.content, target_languages=[target_lang], source_language="auto"
                )

                if target_lang in translation_result:
                    content = translation_result[target_lang]

                    # Store translation in database
                    new_translation = MessageTranslation(
                        message_id=message.id, content=content, target_language=target_lang
                    )
                    await self.message_translation_repo.create(new_translation)

                    translations[target_lang] = content
                    logger.info(
                        "message_translated",
                        message_id=message.id,
                        target_language=target_lang,
                    )

            except (TranslationError, SQLAlchemyError, ValueError) as e:
                logger.error(
                    "translation_failed",
                    message_id=message.id,
                    target_language=target_lang,
                    error=str(e),
                )
                continue

        logger.info(
            "background_translation_completed",
            message_id=message.id,
            translation_count=len(translations),
        )
        return translations

    @background_task_retry(max_retries=1, delay=1.0)
    async def cleanup_old_translations_background(self, days_old: int = 30) -> int:
        """
        Clean up old translations in background.
        :param days_old: Remove translations older than this many days
        :return: Number of cleaned up translations
        """
        logger.info("translation_cleanup_started", days_old=days_old)

        try:
            cleaned_count = await self.message_translation_repo.cleanup_old_translations(days_old)
            logger.info("translation_cleanup_completed", cleaned_count=cleaned_count)
            return cleaned_count
        except SQLAlchemyError as e:
            logger.error("translation_cleanup_failed", error=str(e))
            raise

    @background_task_retry(max_retries=1, delay=0.5)
    async def log_user_activity_background(
        self, user_id: int, activity_type: str, details: dict[str, any] = None
    ) -> None:
        """
        Log user activity in background.
        :param user_id: User ID
        :param activity_type: Type of activity (message_sent, room_joined, etc.)
        :param details: Additional activity details
        """
        logger.info("user_activity_logging", user_id=user_id, activity_type=activity_type)

        try:
            # TODO: Store activity in database or external analytics service
            activity_details = details or {}
            logger.info(
                "user_activity_logged",
                user_id=user_id,
                activity_type=activity_type,
                details=activity_details,
            )
        except (OSError, ValueError) as e:
            logger.error("user_activity_logging_failed", error=str(e))
            raise

    @background_task_retry(max_retries=2, delay=3.0)
    async def notify_room_users_background(
        self, room_id: int, message: str, exclude_user_ids: list[int] = None
    ) -> None:
        """
        Send notifications to room users in background.
        :param room_id: Room ID
        :param message: Notification message
        :param exclude_user_ids: User IDs to exclude from notification
        """
        exclude_user_ids = exclude_user_ids or []
        logger.info(
            "room_notification_sending",
            room_id=room_id,
            excluded_user_count=len(exclude_user_ids),
        )

        try:
            # TODO: Integrate with notification service (WebSocket, Push, Email)
            logger.info("room_notification_sent", room_id=room_id, message=message)
        except (OSError, ValueError) as e:
            logger.error("room_notification_failed", error=str(e))
            raise
