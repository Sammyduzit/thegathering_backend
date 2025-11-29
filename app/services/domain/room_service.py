import structlog
from sqlalchemy.exc import SQLAlchemyError

from app.core.constants import MAX_ROOM_MESSAGES, MESSAGE_CLEANUP_FREQUENCY
from app.core.exceptions import (
    DuplicateResourceException,
    InvalidOperationException,
    RoomNotFoundException,
    UserNotInRoomException,
)
from app.models.message import Message
from app.models.room import Room
from app.models.user import User, UserStatus
from app.repositories.ai_entity_repository import IAIEntityRepository
from app.repositories.conversation_repository import IConversationRepository
from app.repositories.message_repository import IMessageRepository
from app.repositories.message_translation_repository import IMessageTranslationRepository
from app.repositories.room_repository import IRoomRepository
from app.repositories.user_repository import IUserRepository
from app.services.domain.translation_service import TranslationService

logger = structlog.get_logger(__name__)


class RoomService:
    """Service for room business logic using Repository Pattern."""

    def __init__(
        self,
        room_repo: IRoomRepository,
        user_repo: IUserRepository,
        message_repo: IMessageRepository,
        conversation_repo: IConversationRepository,
        message_translation_repo: IMessageTranslationRepository,
        translation_service: TranslationService,
        ai_entity_repo: IAIEntityRepository,
    ):
        self.room_repo = room_repo
        self.user_repo = user_repo
        self.message_repo = message_repo
        self.conversation_repo = conversation_repo
        self.message_translation_repo = message_translation_repo
        self.translation_service = translation_service
        self.ai_entity_repo = ai_entity_repo

    async def get_all_rooms(self) -> list[Room]:
        """Get all active rooms."""
        return await self.room_repo.get_active_rooms()

    async def create_room(
        self,
        name: str,
        description: str | None,
        max_users: int | None,
        is_translation_enabled: bool = False,
    ) -> Room:
        """
        Create new room with validation.
        :param name: Room name
        :param description: Room description
        :param max_users: Maximum users allowed
        :return: Created room
        """
        if await self.room_repo.name_exists(name):
            raise DuplicateResourceException("Room", name)

        new_room = Room(
            name=name,
            description=description,
            max_users=max_users,
            is_translation_enabled=is_translation_enabled,
        )

        return await self.room_repo.create(new_room)

    async def update_room(
        self,
        room_id: int,
        name: str,
        description: str | None,
        max_users: int | None,
        is_translation_enabled: bool = False,
    ) -> Room:
        """
        Update room with validation.
        :param room_id: Room ID to update
        :param name: New room name
        :param description: New room description
        :param max_users: New max users
        :return: Updated room
        """
        room = await self._get_room_or_404(room_id)

        if name != room.name and await self.room_repo.name_exists(name, room_id):
            raise DuplicateResourceException("Room", name)

        room.name = name
        room.description = description
        room.max_users = max_users
        room.is_translation_enabled = is_translation_enabled

        return await self.room_repo.update(room)

    async def delete_room(self, room_id: int) -> dict:
        """
        Delete room with cleanup: soft delete room, kick users, archive conversations, hard delete room messages.

        Room messages are permanently deleted (hard delete).
        Conversation messages (private/group chats) are preserved - they remain accessible even when archived.

        :param room_id: Room ID to delete
        :return: Cleanup summary with statistics
        """
        room = await self._get_room_or_404(room_id)

        # Kick all users from room
        users_in_room = await self.room_repo.get_users_in_room(room_id)
        for user in users_in_room:
            user.current_room_id = None
            user.status = UserStatus.AWAY
            await self.user_repo.update(user)

        # Archive all conversations in room
        conversations = await self.conversation_repo.get_room_conversations(room_id)
        for conversation in conversations:
            conversation.is_active = False
            await self.conversation_repo.update(conversation)

        # Hard delete all room messages (conversation messages are preserved)
        deleted_messages = await self.message_repo.delete_room_messages(room_id)

        # Soft delete room
        await self.room_repo.soft_delete(room_id)
        room.is_active = False

        return {
            "message": f"Room '{room.name}' has been deleted",
            "room_id": room_id,
            "users_removed": len(users_in_room),
            "conversations_archived": len(conversations),
            "messages_deleted": deleted_messages,
        }

    async def get_room_by_id(self, room_id: int) -> Room:
        """Get room by ID with validation."""
        return await self._get_room_or_404(room_id)

    async def get_room_count(self) -> dict:
        """Get count of active rooms."""
        active_rooms = await self.room_repo.get_active_rooms()
        room_count = len(active_rooms)

        return {
            "active_rooms": room_count,
            "message": f"Found {room_count} active rooms",
        }

    async def join_room(self, current_user: User, room_id: int) -> dict:
        """
        User joins room with validation.
        :param current_user: User joining room
        :param room_id: Room ID to join
        :return: Join confirmation
        """
        room = await self._get_room_or_404(room_id)

        current_user_count = await self.room_repo.get_user_count(room_id)
        if room.max_users and current_user_count >= room.max_users:
            raise InvalidOperationException(f"Room '{room.name}' is full (max {room.max_users} users)")

        current_user.current_room_id = room_id
        current_user.status = UserStatus.AVAILABLE
        await self.user_repo.update(current_user)

        final_user_count = await self.room_repo.get_user_count(room_id)

        return {
            "message": f"Successfully joined room '{room.name}'",
            "room_id": room_id,
            "room_name": room.name,
            "user_count": final_user_count,
        }

    async def leave_room(self, current_user: User, room_id: int) -> dict:
        """
        User leaves room with validation.
        :param current_user: User leaving room
        :param room_id: Room ID to leave
        :return: Leave confirmation
        """
        room = await self._get_room_or_404(room_id)

        if current_user.current_room_id != room_id:
            raise InvalidOperationException(f"User is not in room '{room.name}'")

        current_user.current_room_id = None
        current_user.status = UserStatus.AWAY
        await self.user_repo.update(current_user)

        return {
            "message": f"Left room '{room.name}'",
            "room_id": room_id,
            "room_name": room.name,
        }

    async def get_room_participants(self, room_id: int) -> dict:
        """
        Get all participants (humans + AI) in room with validation.
        Returns unified list of participants consistent with conversation participants structure.
        :param room_id: Room ID
        :return: Room participants data including both humans and AI
        """
        room = await self._get_room_or_404(room_id)

        # Get human users
        users = await self.room_repo.get_users_in_room(room_id)

        # Build participant list with human users
        participants = [
            {
                "id": user.id,
                "username": user.username,
                "avatar_url": user.avatar_url,
                "status": user.status.value,
                "is_ai": False,
                "last_active": user.last_active,
                "preferred_language": user.preferred_language,
            }
            for user in users
        ]

        # Get AI entity if present in room
        ai_entity = await self.ai_entity_repo.get_ai_in_room(room_id)
        if ai_entity:
            participants.append(
                {
                    "id": ai_entity.id,
                    "username": ai_entity.username,
                    "avatar_url": None,  # AI entities don't have avatars
                    "status": ai_entity.status.value,
                    "is_ai": True,
                    "last_active": None,  # AI entities don't have last_active
                }
            )

        return {
            "room_id": room_id,
            "room_name": room.name,
            "total_participants": len(participants),
            "participants": participants,
        }

    async def update_user_status(self, current_user: User, new_status: UserStatus) -> dict:
        """
        Update user status.
        :param current_user: User to update
        :param new_status: New status
        :return: Status update confirmation
        """
        current_user.status = new_status
        await self.user_repo.update(current_user)

        return {
            "message": f"Status updated to '{new_status.value}'",
            "new_status": new_status.value,
            "user": current_user.username,
        }

    async def send_room_message(
        self,
        current_user: User,
        room_id: int,
        content: str,
        in_reply_to_message_id: int | None = None,
    ) -> Message:
        """
        Send message to room with validation.
        :param current_user: User sending message
        :param room_id: Target room ID
        :param content: Message content
        :param in_reply_to_message_id: Optional message to reply to
        :return: Created message
        """
        # Validate access
        room = await self._validate_user_in_room(current_user, room_id)

        # Increment weekly message counter
        current_user.weekly_message_count += 1
        await self.user_repo.update(current_user)

        # Create message
        message = await self.message_repo.create_room_message(
            room_id=room_id,
            content=content,
            sender_user_id=current_user.id,
            in_reply_to_message_id=in_reply_to_message_id,
        )

        # Trigger translation if needed
        await self._trigger_room_translation_if_needed(room, message, current_user, content)

        # Periodic cleanup
        await self._cleanup_old_messages_if_needed(room_id, message.id)

        return message

    async def _validate_user_in_room(self, current_user: User, room_id: int):
        """Validate that user is currently in the room."""
        room = await self._get_room_or_404(room_id)
        if current_user.current_room_id != room_id:
            raise UserNotInRoomException(f"User must be in room '{room.name}' to send messages")
        return room

    async def _trigger_room_translation_if_needed(
        self, room, message: Message, current_user: User, content: str
    ) -> None:
        """Trigger translation for room message if translation is enabled."""
        if not room.is_translation_enabled:
            return

        # Get target languages from room users
        target_languages = await self._get_room_translation_target_languages(room.id, current_user)

        if target_languages:
            source_lang = current_user.preferred_language.upper() if current_user.preferred_language else None
            await self.translation_service.translate_and_store_message(
                message_id=message.id,
                content=content,
                source_language=source_lang,
                target_languages=target_languages,
            )

    async def _get_room_translation_target_languages(self, room_id: int, current_user: User) -> list[str]:
        """Get list of unique target languages for translation from room users."""
        room_users = await self.room_repo.get_users_in_room(room_id)
        return TranslationService.get_target_languages_from_users(room_users, current_user)

    async def _cleanup_old_messages_if_needed(self, room_id: int, message_id: int) -> None:
        """Periodically cleanup old room messages to maintain limit."""
        try:
            if message_id % MESSAGE_CLEANUP_FREQUENCY == 0:
                await self.message_repo.cleanup_old_room_messages(room_id, MAX_ROOM_MESSAGES)
        except SQLAlchemyError as e:
            logger.warning(f"Cleanup failed, but message sent successfully: {e}")

    async def get_room_messages(
        self, current_user: User, room_id: int, page: int = 1, page_size: int = 50
    ) -> tuple[list[Message], int]:
        """
        Get room messages with validation and pagination.
        :param current_user: User requesting messages
        :param room_id: Room ID
        :param page: Page number
        :param page_size: Messages per page
        :return: Tuple of (messages, total_count)
        """
        await self._get_room_or_404(room_id)

        if current_user.current_room_id != room_id:
            raise UserNotInRoomException("User must join the room before viewing messages")

        messages, total_count = await self.message_repo.get_room_messages(
            room_id=room_id,
            page=page,
            page_size=page_size,
        )

        # Apply translations if user has preferred language
        if current_user.preferred_language:
            messages = await self._apply_translations_to_messages(messages, current_user.preferred_language)

        return messages, total_count

    async def _apply_translations_to_messages(self, messages: list[Message], user_language: str) -> list[Message]:
        """Apply translations to messages based on user's preferred language."""
        if not messages:
            return messages

        # Get all message IDs for batch translation lookup
        message_ids = [msg.id for msg in messages]

        # Batch query all translations for efficiency (avoid N+1 queries)
        translations = {}
        for message_id in message_ids:
            translation = await self.message_translation_repo.get_by_message_and_language(
                message_id, user_language.upper()
            )
            if translation:
                translations[message_id] = translation.content

        # Apply translations to messages
        for message in messages:
            if message.id in translations:
                message.content = translations[message.id]

        return messages

    async def _get_room_or_404(self, room_id: int) -> Room:
        """Get room by ID or raise NotFoundException."""
        room = await self.room_repo.get_by_id(room_id)
        if not room:
            raise RoomNotFoundException(room_id)
        return room
