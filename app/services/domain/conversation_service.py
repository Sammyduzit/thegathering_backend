import structlog
from arq import ArqRedis

from app.core.config import settings
from app.core.exceptions import (
    ConversationNotFoundException,
    ConversationValidationException,
    NotConversationParticipantException,
    UserNotFoundException,
    UserNotInRoomException,
)
from app.core.utils import enqueue_arq_job_safe
from app.models.conversation import Conversation, ConversationType
from app.models.message import Message
from app.models.user import User
from app.repositories.ai_entity_repository import IAIEntityRepository
from app.repositories.conversation_repository import IConversationRepository
from app.repositories.message_repository import IMessageRepository
from app.repositories.room_repository import IRoomRepository
from app.repositories.user_repository import IUserRepository
from app.schemas.chat_schemas import (
    ConversationPermissions,
    MessageResponse,
    ParticipantAddResponse,
    ParticipantInfo,
    ParticipantRemoveResponse,
)
from app.services.domain.translation_service import TranslationService

logger = structlog.get_logger(__name__)


class ConversationService:
    """Service for conversation business logic using Repository Pattern."""

    def __init__(
        self,
        conversation_repo: IConversationRepository,
        message_repo: IMessageRepository,
        user_repo: IUserRepository,
        room_repo: IRoomRepository,
        translation_service: TranslationService,
        ai_entity_repo: IAIEntityRepository,
        arq_pool: ArqRedis | None = None,
    ):
        self.conversation_repo = conversation_repo
        self.message_repo = message_repo
        self.user_repo = user_repo
        self.room_repo = room_repo
        self.translation_service = translation_service
        self.ai_entity_repo = ai_entity_repo
        self.arq_pool = arq_pool

    @staticmethod
    def _get_message_preview(message: Message | None, max_length: int = 50) -> str | None:
        """
        Generate message preview with truncation.

        :param message: Message object or None
        :param max_length: Maximum length before truncation
        :return: Truncated message preview or None if no message
        """
        if not message:
            return None

        if len(message.content) > max_length:
            return message.content[:max_length] + "..."

        return message.content

    async def create_conversation(
        self,
        current_user: User,
        participant_usernames: list[str],
        conversation_type: ConversationType,
    ) -> Conversation:
        """
        Create private or group conversation with validation.
        Supports both human and AI participants.
        :param current_user: User creating the conversation
        :param participant_usernames: List of participant usernames (human or AI)
        :param conversation_type: ConversationType enum (PRIVATE or GROUP)
        :return: Created conversation
        """
        if not current_user.current_room_id:
            raise UserNotInRoomException("User must be in a room to create conversations")

        if conversation_type == ConversationType.PRIVATE and len(participant_usernames) != 1:
            raise ConversationValidationException("Private conversations require exactly 1 other participant")

        if conversation_type == ConversationType.GROUP and len(participant_usernames) < 1:
            raise ConversationValidationException("Group conversations require at least 1 other participant")

        # Validate and separate human/AI participants
        human_participants, ai_participants = await self._validate_participants(
            participant_usernames, current_user.current_room_id
        )

        # Extract IDs
        user_ids = [current_user.id] + [user.id for user in human_participants]
        ai_ids = [ai.id for ai in ai_participants]

        # Create conversation with all participants at once
        if conversation_type == ConversationType.PRIVATE:
            conversation = await self.conversation_repo.create_private_conversation(
                room_id=current_user.current_room_id,
                user_ids=user_ids,
                ai_ids=ai_ids,
            )
        else:
            conversation = await self.conversation_repo.create_group_conversation(
                room_id=current_user.current_room_id,
                user_ids=user_ids,
                ai_ids=ai_ids,
            )

        return conversation

    async def send_message(
        self,
        current_user: User,
        conversation_id: int,
        content: str,
        in_reply_to_message_id: int | None = None,
    ) -> Message:
        """
        Send message to conversation with validation.
        :param current_user: User sending the message
        :param conversation_id: Target conversation ID
        :param content: Message content
        :param in_reply_to_message_id: Optional message to reply to
        :return: Created message
        """
        # Validate access
        conversation = await self._validate_sender_access(current_user.id, conversation_id)

        # Increment weekly message counter
        current_user.weekly_message_count += 1
        await self.user_repo.update(current_user)

        # Create message
        message = await self.message_repo.create_conversation_message(
            conversation_id=conversation_id,
            content=content,
            sender_user_id=current_user.id,
            in_reply_to_message_id=in_reply_to_message_id,
        )

        # Trigger translation if needed
        await self._trigger_translation_if_needed(conversation, message, current_user, content)

        return message

    async def _validate_sender_access(self, user_id: int, conversation_id: int) -> Conversation:
        """Validate that user has access to send messages in conversation."""
        conversation = await self.conversation_repo.get_by_id(conversation_id)
        if not conversation:
            raise ConversationNotFoundException(conversation_id)

        if not await self.conversation_repo.is_participant(conversation_id, user_id):
            raise NotConversationParticipantException()

        return conversation

    async def _trigger_translation_if_needed(
        self, conversation: Conversation, message: Message, current_user: User, content: str
    ) -> None:
        """Trigger translation for message if room has translation enabled."""
        # Load room async to check translation settings (avoid lazy loading)
        room = None
        if conversation.room_id:
            room = await self.room_repo.get_by_id(conversation.room_id)

        if not room or not room.is_translation_enabled:
            return

        # Get target languages from participants
        target_languages = await self._get_translation_target_languages(conversation.id, current_user)

        if target_languages:
            source_lang = current_user.preferred_language.upper() if current_user.preferred_language else None
            await self.translation_service.translate_and_store_message(
                message_id=message.id,
                content=content,
                source_language=source_lang,
                target_languages=target_languages,
            )

    async def _get_translation_target_languages(self, conversation_id: int, current_user: User) -> list[str]:
        """Get list of unique target languages for translation from conversation participants."""
        participants = await self.conversation_repo.get_participants(conversation_id)
        # Extract User objects from participants (filter out AI participants)
        users = [p.user for p in participants if p.user_id and p.user]
        return TranslationService.get_target_languages_from_users(users, current_user)

    async def get_messages(
        self,
        current_user: User,
        conversation_id: int,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[Message], int]:
        """
        Get conversation messages with validation and pagination.
        :param current_user: User requesting messages
        :param conversation_id: Conversation ID
        :param page: Page number
        :param page_size: Messages per page
        :return: Tuple of (messages, total_count)
        """
        await self._validate_conversation_access(current_user.id, conversation_id)

        messages, total_count = await self.message_repo.get_conversation_messages(
            conversation_id=conversation_id,
            page=page,
            page_size=page_size,
        )

        return messages, total_count

    async def get_user_conversations(self, user_id: int) -> list[dict]:
        """
        Get all active conversations for user with formatted response.
        Includes room name, participant info, and latest message preview.
        :param user_id: User ID
        :return: List of formatted conversation data
        """
        conversations = await self.conversation_repo.get_user_conversations(user_id)

        conversation_list = []
        for conv in conversations:
            # Get participants
            participants = await self.conversation_repo.get_participants(conv.id)
            participant_names = [
                p.participant_name for p in participants if p.user_id != user_id or p.ai_entity_id is not None
            ]

            # Get room name
            room = await self.room_repo.get_by_id(conv.room_id)
            room_name = room.name if room else None

            # Get latest message for preview
            latest_message_obj = await self.message_repo.get_latest_conversation_message(conv.id)
            latest_message_at = latest_message_obj.sent_at if latest_message_obj else None
            latest_message_preview = self._get_message_preview(latest_message_obj)

            conversation_list.append(
                {
                    "id": conv.id,
                    "type": conv.conversation_type.value,
                    "room_id": conv.room_id,
                    "room_name": room_name,
                    "participants": participant_names,
                    "participant_count": len(participants),
                    "created_at": conv.created_at,
                    "latest_message_at": latest_message_at,
                    "latest_message_preview": latest_message_preview,
                }
            )

        return conversation_list

    async def get_participants(self, current_user: User, conversation_id: int) -> list[ParticipantInfo]:
        """
        Get conversation participants with validation.
        :param current_user: User requesting participants
        :param conversation_id: Conversation ID
        :return: List of formatted participant data
        """
        await self._validate_conversation_access(current_user.id, conversation_id)

        participants = await self.conversation_repo.get_participants(conversation_id)

        return await self._format_participant_details(participants)

    async def _validate_participants(self, usernames: list[str], room_id: int) -> tuple[list[User], list]:
        """
        Validate and separate human and AI participants.

        Logic:
        1. Try to find each username as human user first
        2. If not found, try to find as AI entity
        3. Validate that humans are in the same room
        4. Validate that AI entities are in the same room

        :param usernames: List of usernames (human or AI)
        :param room_id: Room ID for validation
        :return: Tuple of (human_users, ai_entities)
        """
        human_participants = []
        ai_participants = []

        for username in usernames:
            # Try human first
            user = await self.user_repo.get_by_username(username)
            if user:
                if not user.is_active:
                    raise ConversationValidationException(f"User '{username}' is not active")
                if user.current_room_id != room_id:
                    raise ConversationValidationException(f"User '{username}' is not in the same room")
                human_participants.append(user)
                continue

            # Try AI
            ai_entity = await self.ai_entity_repo.get_by_username(username)
            if ai_entity:
                if ai_entity.current_room_id != room_id:
                    raise ConversationValidationException(f"AI '{username}' is not in the same room")
                ai_participants.append(ai_entity)
                continue

            # Neither found
            raise UserNotFoundException(f"Participant '{username}' not found")

        return human_participants, ai_participants

    async def _validate_conversation_access(self, user_id: int, conversation_id: int) -> Conversation:
        """
        Validate conversation exists and user has access.
        :param user_id: User ID
        :param conversation_id: Conversation ID
        :return: Conversation object
        """
        conversation = await self.conversation_repo.get_by_id(conversation_id)
        if not conversation:
            raise ConversationNotFoundException(conversation_id)

        if not await self.conversation_repo.is_participant(conversation_id, user_id):
            raise NotConversationParticipantException()

        return conversation

    async def add_participant(
        self,
        conversation_id: int,
        username: str,
        current_user: User,
    ) -> ParticipantAddResponse:
        """
        Add participant to conversation (human or AI).

        Logic:
        1. Check if current_user is a participant.
        2. Try to find a Human user with the username.
        3. If not found, try to find an AI entity with the name.
        4. Add to the conversation.

        :param conversation_id: Conversation ID
        :param username: Username of human or name of AI entity
        :param current_user: User adding the participant
        :return: Success response with participant info
        """
        if not await self.conversation_repo.is_participant(conversation_id, current_user.id):
            raise NotConversationParticipantException()

        # Try Human first
        human_user = await self.user_repo.get_by_username(username)
        if human_user:
            await self.conversation_repo.add_participant(conversation_id, user_id=human_user.id)
            participant_count = await self.conversation_repo.count_active_participants(conversation_id)
            return ParticipantAddResponse(
                message=f"User '{username}' added to conversation",
                conversation_id=conversation_id,
                username=username,
                participant_count=participant_count,
            )

        # Try AI
        ai_entity = await self.ai_entity_repo.get_by_username(username)
        if ai_entity:
            await self.conversation_repo.add_participant(conversation_id, ai_entity_id=ai_entity.id)
            participant_count = await self.conversation_repo.count_active_participants(conversation_id)
            return ParticipantAddResponse(
                message=f"Participant '{username}' added to conversation",
                conversation_id=conversation_id,
                username=username,
                participant_count=participant_count,
            )

        raise UserNotFoundException(f"Participant '{username}' not found")

    async def remove_participant(
        self,
        conversation_id: int,
        username: str,
        current_user: User,
    ) -> ParticipantRemoveResponse:
        """
        Remove participant from conversation.

        Logic:
        - Users can remove themselves (self-leave)
        - Only admins can remove other users or AI entities

        :param conversation_id: Conversation ID
        :param username: Username of human or name of AI entity
        :param current_user: User performing the removal
        :return: Success response with removal info
        """
        from app.core.exceptions import ForbiddenException

        conversation = await self.conversation_repo.get_by_id(conversation_id)
        if not conversation:
            raise ConversationNotFoundException(conversation_id)

        # Check if user is removing themselves
        is_self_remove = username == current_user.username

        # Try Human first
        human_user = await self.user_repo.get_by_username(username)
        if human_user:
            # If not self-remove, require admin
            if not is_self_remove and not current_user.is_admin:
                raise ForbiddenException("Only admins can remove other participants")

            removed = await self.conversation_repo.remove_participant(conversation_id, user_id=human_user.id)
            if not removed:
                raise UserNotFoundException(f"User '{username}' is not a participant in this conversation")
            participant_count = await self.conversation_repo.count_active_participants(conversation_id)
            return ParticipantRemoveResponse(
                message=f"User '{username}' removed from conversation",
                conversation_id=conversation_id,
                username=username,
                participant_count=participant_count,
            )

        # Try AI (only admins can remove AI)
        ai_entity = await self.ai_entity_repo.get_by_username(username)
        if ai_entity:
            if not current_user.is_admin:
                raise ForbiddenException("Only admins can remove AI participants")

            removed = await self.conversation_repo.remove_participant(conversation_id, ai_entity_id=ai_entity.id)
            if not removed:
                raise UserNotFoundException(f"AI '{username}' is not a participant in this conversation")

            # Enqueue long-term memory creation if AI features enabled
            await self._enqueue_long_term_memory_for_ai(
                conversation_id=conversation_id,
                ai_entity_id=ai_entity.id,
            )

            participant_count = await self.conversation_repo.count_active_participants(conversation_id)
            return ParticipantRemoveResponse(
                message=f"Participant '{username}' removed from conversation",
                conversation_id=conversation_id,
                username=username,
                participant_count=participant_count,
            )

        raise UserNotFoundException(f"Participant '{username}' not found")

    async def _enqueue_long_term_memory_for_ai(
        self,
        conversation_id: int,
        ai_entity_id: int,
    ) -> None:
        """
        Enqueue background task to create long-term memory when AI leaves conversation.

        This method is called whenever an AI entity is removed from a conversation
        (via goodbye, admin removal, etc.) to preserve the conversation in the AI's memory.

        The task will automatically fetch all participants from the conversation.

        :param conversation_id: The conversation the AI is leaving
        :param ai_entity_id: The AI entity's ID
        """
        if not settings.ai_features_enabled or not self.arq_pool:
            logger.debug(
                "long_term_memory_skipped",
                reason="ai_features_disabled_or_no_arq_pool",
                conversation_id=conversation_id,
                ai_entity_id=ai_entity_id,
            )
            return

        await enqueue_arq_job_safe(
            self.arq_pool,
            "create_long_term_memory_task",
            {
                "ai_entity_id": ai_entity_id,
                "conversation_id": conversation_id,
            },
            ai_entity_id,
            conversation_id,
        )

    async def get_conversation_detail(self, current_user: User, conversation_id: int) -> dict:
        """
        Get detailed conversation information including participants, permissions, and message metadata.
        :param current_user: User requesting conversation details
        :param conversation_id: Conversation ID
        :return: Detailed conversation data formatted for frontend
        """
        # Validate access
        await self._validate_conversation_access(current_user.id, conversation_id)

        # Get conversation data
        conversation = await self.conversation_repo.get_by_id(conversation_id)
        if not conversation:
            raise ConversationNotFoundException(f"Conversation {conversation_id} not found")

        participants = await self.conversation_repo.get_participants(conversation_id)

        # Format response components
        participant_details = await self._format_participant_details(participants)
        room_name = await self._get_room_name(conversation.room_id)
        message_metadata = await self._get_message_metadata(conversation_id)
        permissions = self._calculate_permissions(current_user, participants)

        return {
            "id": conversation.id,
            "type": conversation.conversation_type.value,
            "room_id": conversation.room_id,
            "room_name": room_name,
            "is_active": conversation.is_active,
            "created_at": conversation.created_at,
            "participants": participant_details,
            "participant_count": len(participants),
            **message_metadata,
            "permissions": permissions,
        }

    async def _format_participant_details(self, participants) -> list[ParticipantInfo]:
        """Format participant list with full details."""
        return [
            ParticipantInfo(
                id=p.user_id if p.user_id else p.ai_entity_id,
                username=(
                    p.participant_name if p.user_id
                    else p.ai_entity.username if p.ai_entity
                    else "Unknown"
                ),
                avatar_url=p.user.avatar_url if p.user_id else None,
                status=p.user.status if p.user_id else p.ai_entity.status if p.ai_entity else "offline",
                is_ai=p.is_ai,
            )
            for p in participants
        ]

    async def _get_room_name(self, room_id: int) -> str | None:
        """Get room name by ID."""
        room = await self.room_repo.get_by_id(room_id)
        return room.name if room else None

    async def _get_message_metadata(self, conversation_id: int) -> dict:
        """Get message count and latest message."""
        message_count = await self.message_repo.count_conversation_messages(conversation_id)
        latest_message_obj = await self.message_repo.get_latest_conversation_message(conversation_id)

        latest_message = None
        if latest_message_obj:
            latest_message = MessageResponse.model_validate(latest_message_obj)

        return {
            "message_count": message_count,
            "latest_message": latest_message,
        }

    def _calculate_permissions(self, current_user: User, participants) -> ConversationPermissions:
        """Calculate user permissions for conversation."""
        is_participant = any(p.user_id == current_user.id for p in participants)
        return ConversationPermissions(
            can_post=is_participant,
            can_manage_participants=current_user.is_admin or is_participant,
            can_leave=is_participant,
        )

    async def update_conversation(
        self,
        current_user: User,
        conversation_id: int,
        is_active: bool,
    ) -> Conversation:
        """
        Update conversation metadata (currently supports archiving/unarchiving).

        Only participants can archive their own view of the conversation.

        :param current_user: User updating the conversation
        :param conversation_id: Conversation ID
        :param is_active: Whether conversation is active (false = archived)
        :return: Updated conversation
        :raises ConversationNotFoundException: If conversation not found
        :raises NotConversationParticipantException: If user is not a participant
        """
        # Get conversation
        conversation = await self.conversation_repo.get_by_id(conversation_id)
        if not conversation:
            raise ConversationNotFoundException(f"Conversation {conversation_id} not found")

        # Verify participant
        is_participant = await self.conversation_repo.is_participant(conversation_id, current_user.id)
        if not is_participant:
            raise NotConversationParticipantException(
                f"User {current_user.username} is not a participant in conversation {conversation_id}"
            )

        # Update is_active field
        conversation.is_active = is_active
        updated = await self.conversation_repo.update(conversation)

        return updated

    async def delete_conversation(
        self,
        current_user: User,
        conversation_id: int,
    ) -> None:
        """
        Soft-delete conversation by setting is_active to False (archive).

        Only participants can archive conversations.
        This is an alias for update_conversation with is_active=False.

        :param current_user: User deleting the conversation
        :param conversation_id: Conversation ID
        :raises ConversationNotFoundException: If conversation not found
        :raises NotConversationParticipantException: If user is not a participant
        """
        await self.update_conversation(current_user, conversation_id, is_active=False)
