from typing import Any

import structlog

from app.core.background_tasks import background_task_retry

logger = structlog.get_logger(__name__)


class BackgroundService:
    """Service for handling background tasks like activity logging and notifications."""

    def __init__(self):
        """Initialize BackgroundService."""
        pass

    @background_task_retry(max_retries=1, delay=0.5)
    async def log_user_activity_background(
        self, user_id: int, activity_type: str, details: dict[str, Any] | None = None
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
