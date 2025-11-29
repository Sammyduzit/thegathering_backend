from typing import TYPE_CHECKING

import structlog
from sqlalchemy.exc import IntegrityError

from app.core.exceptions import (
    AIEntityNotFoundException,
    AIEntityOfflineException,
    ConversationNotFoundException,
    DuplicateResourceException,
    InvalidOperationException,
    RoomNotFoundException,
)
from app.interfaces.ai_provider import IAIProvider
from app.models.ai_entity import AIEntity, AIEntityStatus, AIResponseStrategy
from app.models.message import Message
from app.repositories.ai_cooldown_repository import IAICooldownRepository
from app.repositories.ai_entity_repository import IAIEntityRepository
from app.repositories.conversation_repository import IConversationRepository
from app.repositories.message_repository import IMessageRepository
from app.repositories.room_repository import IRoomRepository

if TYPE_CHECKING:
    from app.models.conversation import Conversation
    from app.services.domain.conversation_service import ConversationService

logger = structlog.get_logger(__name__)


class AIEntityService:
    """Service for AI entity business logic using Repository Pattern."""

    def __init__(
        self,
        ai_entity_repo: IAIEntityRepository,
        conversation_repo: IConversationRepository,
        cooldown_repo: IAICooldownRepository,
        room_repo: IRoomRepository,
        message_repo: IMessageRepository,
        conversation_service: "ConversationService",
        ai_provider: IAIProvider | None = None,
    ):
        self.ai_entity_repo = ai_entity_repo
        self.conversation_repo = conversation_repo
        self.cooldown_repo = cooldown_repo
        self.room_repo = room_repo
        self.message_repo = message_repo
        self.conversation_service = conversation_service
        self.ai_provider = ai_provider

    async def get_all_entities(self) -> list[AIEntity]:
        """Get all AI entities."""
        return await self.ai_entity_repo.get_all()

    async def get_available_entities(self) -> list[AIEntity]:
        """Get all available AI entities (online and not deleted)."""
        return await self.ai_entity_repo.get_available_entities()

    async def get_entity_by_id(self, entity_id: int) -> AIEntity:
        """Get AI entity by ID with validation."""
        entity = await self.ai_entity_repo.get_by_id(entity_id)
        if not entity:
            raise AIEntityNotFoundException(entity_id)
        return entity

    async def create_entity(
        self,
        username: str,
        system_prompt: str,
        model_name: str,
        description: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        room_response_strategy: AIResponseStrategy | None = None,
        conversation_response_strategy: AIResponseStrategy | None = None,
        response_probability: float | None = None,
        cooldown_seconds: int | None = None,
        config: dict | None = None,
    ) -> AIEntity:
        """Create new AI entity with validation."""
        if await self.ai_entity_repo.username_exists(username):
            raise DuplicateResourceException("AI entity", username)

        new_entity = AIEntity(
            username=username,
            description=description,
            system_prompt=system_prompt,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            room_response_strategy=room_response_strategy,
            conversation_response_strategy=conversation_response_strategy,
            response_probability=response_probability,
            cooldown_seconds=cooldown_seconds,
            config=config,
            status=AIEntityStatus.OFFLINE,
        )

        return await self.ai_entity_repo.create(new_entity)

    async def update_entity(
        self,
        entity_id: int,
        username: str | None = None,
        description: str | None = None,
        system_prompt: str | None = None,
        model_name: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        room_response_strategy: AIResponseStrategy | None = None,
        conversation_response_strategy: AIResponseStrategy | None = None,
        response_probability: float | None = None,
        cooldown_seconds: int | None = None,
        config: dict | None = None,
        status: AIEntityStatus | None = None,
        current_room_id: int | None = ...,  # ... as sentinel: not provided
    ) -> AIEntity:
        """Update AI entity with validation and room assignment.

        :param status: If set to OFFLINE, AI will automatically leave current room
        :param current_room_id: Room assignment (None = leave room, int = assign to room, ... = no change)
        :return: Updated AI entity
        """
        entity = await self.get_entity_by_id(entity_id)

        # Handle status change (auto-leave room if set to OFFLINE)
        if status is not None and status != entity.status:
            await self._handle_status_update(entity, status)

        # Handle room assignment if explicitly provided
        if current_room_id is not ...:
            await self._handle_room_update(entity, current_room_id)

        # Update other fields
        self._update_entity_fields(
            entity,
            username=username,
            description=description,
            system_prompt=system_prompt,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            room_response_strategy=room_response_strategy,
            conversation_response_strategy=conversation_response_strategy,
            response_probability=response_probability,
            cooldown_seconds=cooldown_seconds,
            config=config,
        )

        # Commit changes with optimistic locking for room assignment
        try:
            return await self.ai_entity_repo.update(entity)
        except IntegrityError as e:
            # Race condition: Another AI was assigned to the room concurrently
            if "idx_unique_ai_per_room" in str(e.orig):
                raise InvalidOperationException(
                    "Room already has an AI entity assigned (concurrent assignment detected)"
                )
            # Re-raise if it's a different integrity constraint
            raise

    async def delete_entity(self, entity_id: int) -> dict:
        """Soft delete AI entity (set to OFFLINE)."""
        entity = await self.get_entity_by_id(entity_id)

        await self.ai_entity_repo.delete(entity_id)

        return {
            "message": f"AI entity '{entity.username}' has been deleted",
            "entity_id": entity_id,
        }

    async def get_available_in_room(self, room_id: int) -> list[AIEntity]:
        """Get AI entities available in a room (active and not in conversation)."""
        return await self.ai_entity_repo.get_available_in_room(room_id)

    async def invite_to_conversation(self, conversation_id: int, ai_entity_id: int) -> dict:
        """Invite AI entity to a conversation."""
        # Validate AI entity exists and is active
        entity = await self.get_entity_by_id(ai_entity_id)
        if entity.status != AIEntityStatus.ONLINE:
            raise AIEntityOfflineException(entity.username)

        # Validate conversation exists
        conversation = await self.conversation_repo.get_by_id(conversation_id)
        if not conversation:
            raise ConversationNotFoundException(conversation_id)

        # Check if AI is already in this conversation
        existing_ai = await self.ai_entity_repo.get_ai_in_conversation(conversation_id)
        if existing_ai:
            raise InvalidOperationException(f"AI entity '{existing_ai.username}' is already in this conversation")

        # Add AI to conversation
        try:
            await self.conversation_repo.add_participant(conversation_id, ai_entity_id=ai_entity_id)
        except ValueError as e:
            raise InvalidOperationException(str(e))

        return {
            "message": f"AI entity '{entity.username}' invited to conversation",
            "conversation_id": conversation_id,
            "ai_entity_id": ai_entity_id,
        }

    async def remove_from_conversation(self, conversation_id: int, ai_entity_id: int) -> dict:
        """Remove AI entity from a conversation."""
        # Validate AI entity exists
        entity = await self.get_entity_by_id(ai_entity_id)

        # Validate conversation exists
        conversation = await self.conversation_repo.get_by_id(conversation_id)
        if not conversation:
            raise ConversationNotFoundException(conversation_id)

        # Remove AI from conversation and enqueue long-term memory creation
        await self.conversation_repo.remove_participant(conversation_id, ai_entity_id=ai_entity_id)
        await self.conversation_service._enqueue_long_term_memory_for_ai(
            conversation_id=conversation_id,
            ai_entity_id=ai_entity_id,
        )

        return {
            "message": f"AI entity '{entity.username}' removed from conversation",
            "conversation_id": conversation_id,
            "ai_entity_id": ai_entity_id,
        }

    async def update_cooldown(
        self,
        ai_entity_id: int,
        room_id: int | None = None,
        conversation_id: int | None = None,
    ) -> None:
        """Update AI entity cooldown for rate limiting."""
        await self.cooldown_repo.upsert_cooldown(
            ai_entity_id=ai_entity_id,
            room_id=room_id,
            conversation_id=conversation_id,
        )

    async def _assign_to_room(self, entity: AIEntity, new_room_id: int) -> None:
        """Assign AI entity to a room with validation.

        :param entity: AI entity to assign
        :param new_room_id: Target room ID
        :raises RoomNotFoundException: If room doesn't exist
        :raises InvalidOperationException: If room already has AI or AI is offline
        """
        # Validate new room exists
        new_room = await self.room_repo.get_by_id(new_room_id)
        if not new_room:
            raise RoomNotFoundException(new_room_id)

        # Check if new room already has AI
        if new_room.has_ai:
            raise InvalidOperationException(f"Room '{new_room.name}' already has an AI entity")

        # Check AI is ONLINE before joining
        if entity.status != AIEntityStatus.ONLINE:
            raise InvalidOperationException(f"AI entity '{entity.username}' must be ONLINE to join a room")

        # Remove from old room if present
        if entity.current_room_id:
            old_room = await self.room_repo.get_by_id(entity.current_room_id)
            if old_room:
                old_room.has_ai = False

        # Assign to new room
        new_room.has_ai = True
        entity.current_room_id = new_room_id

        logger.info(
            "ai_assigned_to_room",
            ai_entity_id=entity.id,
            ai_name=entity.username,
            room_id=new_room_id,
            room_name=new_room.name,
        )

    async def _remove_from_room(self, entity: AIEntity) -> None:
        """Remove AI entity from current room.

        :param entity: AI entity to remove from room
        """
        if not entity.current_room_id:
            return  # Already not in a room

        # Get current room and update has_ai flag
        room = await self.room_repo.get_by_id(entity.current_room_id)
        if room:
            room.has_ai = False

        logger.info(
            "ai_removed_from_room",
            ai_entity_id=entity.id,
            ai_name=entity.username,
            room_id=entity.current_room_id,
            room_name=room.name if room else "unknown",
        )

        # Clear room assignment
        entity.current_room_id = None

    async def _handle_status_update(self, entity: AIEntity, new_status: AIEntityStatus) -> None:
        """
        Handle status change with auto-leave logic.

        :param entity: AI entity to update
        :param new_status: New status to set

        Side Effects:
            - Removes entity from room if status is OFFLINE and entity is in a room
            - Updates entity.status
        """
        if new_status == AIEntityStatus.OFFLINE and entity.current_room_id:
            await self._remove_from_room(entity)
        entity.status = new_status

    async def _handle_room_update(self, entity: AIEntity, new_room_id: int | None) -> None:
        """
        Handle room assignment or removal.

        :param entity: AI entity to update
        :param new_room_id: New room ID (None = leave room, int = assign to room)
        :raises RoomNotFoundException: If room doesn't exist
        :raises InvalidOperationException: If room already has AI or AI is offline
        """
        if new_room_id is None:
            if entity.current_room_id:
                await self._remove_from_room(entity)
        else:
            await self._assign_to_room(entity, new_room_id)

    def _update_entity_fields(
        self,
        entity: AIEntity,
        username: str | None = None,
        description: str | None = None,
        system_prompt: str | None = None,
        model_name: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        room_response_strategy: AIResponseStrategy | None = None,
        conversation_response_strategy: AIResponseStrategy | None = None,
        response_probability: float | None = None,
        cooldown_seconds: int | None = None,
        config: dict | None = None,
    ) -> None:
        """
        Update entity fields if provided.

        :param entity: AI entity to update
        :param username: New username (if provided)
        :param description: New description (if provided)
        :param system_prompt: New system prompt (if provided)
        :param model_name: New model name (if provided)
        :param temperature: New temperature (if provided)
        :param max_tokens: New max tokens (if provided)
        :param room_response_strategy: New room response strategy (if provided)
        :param conversation_response_strategy: New conversation response strategy (if provided)
        :param response_probability: New response probability (if provided)
        :param cooldown_seconds: New cooldown seconds (if provided)
        :param config: New config (if provided)

        Side Effects:
            Updates entity fields in-place
        """
        if username is not None:
            entity.username = username
        if description is not None:
            entity.description = description
        if system_prompt is not None:
            entity.system_prompt = system_prompt
        if model_name is not None:
            entity.model_name = model_name
        if temperature is not None:
            entity.temperature = temperature
        if max_tokens is not None:
            entity.max_tokens = max_tokens
        if room_response_strategy is not None:
            entity.room_response_strategy = room_response_strategy
        if conversation_response_strategy is not None:
            entity.conversation_response_strategy = conversation_response_strategy
        if response_probability is not None:
            entity.response_probability = response_probability
        if cooldown_seconds is not None:
            entity.cooldown_seconds = cooldown_seconds
        if config is not None:
            entity.config = config

    async def _generate_farewell_message(
        self,
        ai_entity: AIEntity,
        room_id: int | None = None,
        conversation_id: int | None = None,
    ) -> str:
        """
        Generate contextual farewell message for AI entity.

        Uses recent message context to create a natural 1-2 sentence goodbye.

        :param ai_entity: AI entity saying goodbye
        :param room_id: Room ID if saying goodbye in room
        :param conversation_id: Conversation ID if saying goodbye in conversation
        :return: Generated farewell message (1-2 sentences)
        """
        # Get recent messages for context (last 10)
        if not self.ai_provider:
            raise InvalidOperationException("AI provider is not configured for farewell generation")

        messages = await self.message_repo.get_recent_messages(
            room_id=room_id,
            conversation_id=conversation_id,
            limit=10,
        )

        # Build context from recent messages
        context_lines = []
        for msg in messages[::-1]:  # Reverse to chronological order
            # Handle polymorphic sender (User or AI)
            if msg.sender_user_id:
                sender = msg.sender_user.username
            elif msg.sender_ai_id:
                sender = msg.sender_ai.username
            else:
                sender = "System"  # Fallback for system messages
            context_lines.append(f"{sender}: {msg.content}")

        context = "\n".join(context_lines) if context_lines else "No previous messages"

        # Build farewell system prompt
        farewell_prompt = f"""You are {ai_entity.username}, an AI assistant.

You are leaving this conversation. Generate a brief, natural farewell message (1-2 sentences max).

Be warm, friendly, and contextual based on the recent conversation.

Recent conversation context:
{context}

Generate ONLY the farewell message, nothing else."""

        # Call LLM with high temperature for natural variation
        farewell_message = await self.ai_provider.generate_response(
            messages=[],  # Empty messages, using only system prompt
            system_prompt=farewell_prompt,
            temperature=0.8,  # Higher temperature for natural variation
            max_tokens=100,
        )

        logger.info(
            "farewell_message_generated",
            ai_entity_id=ai_entity.id,
            room_id=room_id,
            conversation_id=conversation_id,
            message_length=len(farewell_message),
        )

        return farewell_message

    async def _say_goodbye_to_conversation(self, entity: "AIEntity", conversation: "Conversation") -> dict:
        """
        Handle goodbye process for a single conversation.

        :param entity: AI entity saying goodbye
        :param conversation: Conversation to leave
        :return: Summary dict with conversation goodbye details
        """
        # Generate and post farewell message
        farewell = await self._generate_farewell_message(
            ai_entity=entity,
            conversation_id=conversation.id,
        )

        # Create message in conversation
        farewell_message = Message(
            conversation_id=conversation.id,
            sender_ai_id=entity.id,
            content=farewell,
        )
        await self.message_repo.create(farewell_message)

        # Leave conversation and enqueue long-term memory creation
        await self.conversation_repo.remove_participant(conversation.id, ai_entity_id=entity.id)
        await self.conversation_service._enqueue_long_term_memory_for_ai(
            conversation_id=conversation.id,
            ai_entity_id=entity.id,
        )

        logger.info(
            "ai_said_goodbye_in_conversation",
            ai_entity_id=entity.id,
            conversation_id=conversation.id,
            message_id=farewell_message.id,
        )

        return {
            "conversation_id": conversation.id,
            "message_id": farewell_message.id,
            "farewell": farewell[:100] + "..." if len(farewell) > 100 else farewell,
        }

    async def _say_goodbye_to_room(self, entity: "AIEntity") -> dict:
        """
        Handle goodbye process for room assignment.

        :param entity: AI entity saying goodbye (must have current_room_id set)
        :return: Summary dict with room goodbye details
        """
        room_id = entity.current_room_id

        # Generate and post farewell message in public room
        farewell = await self._generate_farewell_message(ai_entity=entity, room_id=room_id)

        # Create message in room
        farewell_message = Message(
            room_id=room_id,
            sender_ai_id=entity.id,
            content=farewell,
        )
        await self.message_repo.create(farewell_message)

        # Leave room (clears current_room_id and room.has_ai)
        await self._remove_from_room(entity)

        logger.info(
            "ai_said_goodbye_in_room",
            ai_entity_id=entity.id,
            room_id=room_id,
            message_id=farewell_message.id,
        )

        return {
            "room_id": room_id,
            "message_id": farewell_message.id,
            "farewell": farewell[:100] + "..." if len(farewell) > 100 else farewell,
        }

    async def _disable_ai_responses(self, entity: "AIEntity") -> None:
        """
        Disable AI responses by setting both strategies to NO_RESPONSE.

        :param entity: AI entity to disable
        """
        entity.room_response_strategy = AIResponseStrategy.NO_RESPONSE
        entity.conversation_response_strategy = AIResponseStrategy.NO_RESPONSE
        await self.ai_entity_repo.update(entity)

    async def initiate_graceful_goodbye(self, entity_id: int) -> dict:
        """
        Initiate graceful goodbye for AI entity.

        Orchestrates the goodbye process:
        1. Say goodbye to all active conversations
        2. Say goodbye to room (if assigned)
        3. Disable AI responses

        :param entity_id: AI entity ID to say goodbye
        :return: Dict with summary of goodbye actions
        """
        entity = await self.get_entity_by_id(entity_id)

        summary = {
            "ai_entity_id": entity_id,
            "ai_name": entity.username,
            "room_goodbye": None,
            "conversation_goodbyes": [],
            "strategies_updated": False,
        }

        # Step 1: Handle conversation goodbyes
        active_conversations = await self.conversation_repo.get_active_conversations_for_ai(entity_id)
        for conversation in active_conversations:
            conversation_summary = await self._say_goodbye_to_conversation(entity, conversation)
            summary["conversation_goodbyes"].append(conversation_summary)

        # Step 2: Handle room goodbye (if AI is directly assigned to room)
        if entity.current_room_id:
            summary["room_goodbye"] = await self._say_goodbye_to_room(entity)

        # Step 3: Disable AI responses
        await self._disable_ai_responses(entity)
        summary["strategies_updated"] = True

        logger.info(
            "graceful_goodbye_completed",
            ai_entity_id=entity_id,
            room_goodbyes=1 if summary["room_goodbye"] else 0,
            conversation_goodbyes=len(summary["conversation_goodbyes"]),
        )

        return summary
