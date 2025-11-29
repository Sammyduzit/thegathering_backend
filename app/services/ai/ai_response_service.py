"""
AI Response Service - Orchestrates AI message generation.

This service coordinates:
1. Context building (message history + memories)
2. LLM response generation
3. Message persistence
"""

import re

import structlog

from app.interfaces.ai_provider import AIProviderError, IAIProvider
from app.models.ai_entity import AIEntity
from app.models.message import Message
from app.repositories.ai_cooldown_repository import IAICooldownRepository
from app.repositories.message_repository import IMessageRepository
from app.services.ai.ai_context_service import AIContextService
from app.services.ai.response_strategies import ConversationResponseStrategyEvaluator, RoomResponseStrategyEvaluator

logger = structlog.get_logger(__name__)


class AIResponseService:
    """Service for generating and managing AI responses."""

    def __init__(
        self,
        ai_provider: IAIProvider,
        context_service: AIContextService,
        message_repo: IMessageRepository,
        cooldown_repo: IAICooldownRepository,
    ):
        self.ai_provider = ai_provider
        self.context_service = context_service
        self.message_repo = message_repo
        self.cooldown_repo = cooldown_repo

        # Strategy evaluators
        self.room_strategy_evaluator = RoomResponseStrategyEvaluator()
        self.conversation_strategy_evaluator = ConversationResponseStrategyEvaluator()

    async def generate_conversation_response(
        self,
        conversation_id: int,
        ai_entity: AIEntity,
        user_id: int | None = None,
        include_memories: bool = True,
        in_reply_to_message_id: int | None = None,
    ) -> Message:
        """
        Generate AI response for a conversation message.

        :param conversation_id: Conversation ID to respond in
        :param ai_entity: AI entity that should respond
        :param user_id: User ID for personalized memory retrieval (optional)
        :param include_memories: Whether to include AI memories in context
        :param in_reply_to_message_id: Optional message reference for threading
        :return: Created message with AI response
        :raises AIProviderError: If LLM generation fails
        """
        try:
            # Build context (message history + memories)
            messages, memory_context = await self.context_service.build_full_context(
                conversation_id=conversation_id,
                room_id=None,
                ai_entity=ai_entity,
                user_id=user_id,
                include_memories=include_memories,
            )

            # Enhance system prompt with memories if available
            system_prompt = ai_entity.system_prompt
            if memory_context:
                system_prompt = f"{system_prompt}\n\n{memory_context}"

            # Add anti-parroting instruction (prevents "Name:" prefix copying)
            system_prompt = f"""{system_prompt}

IMPORTANT: You respond directly as part of the conversation.
NEVER begin your responses with your name '{ai_entity.username}:' or similar prefix formats.
Respond naturally and directly."""

            # Generate response from LLM
            response_content = await self.ai_provider.generate_response(
                messages=messages,
                system_prompt=system_prompt,
                temperature=ai_entity.temperature,
                max_tokens=ai_entity.max_tokens,
            )

            # Post-processing: Remove any name prefixes (safety net)
            response_content = self._clean_parroting(response_content, ai_entity.username)

            # Save AI response as message
            message = await self.message_repo.create_conversation_message(
                conversation_id=conversation_id,
                content=response_content,
                sender_ai_id=ai_entity.id,
                in_reply_to_message_id=in_reply_to_message_id,
            )

            logger.info(
                f"AI '{ai_entity.username}' generated response in conversation {conversation_id}: "
                f"{len(response_content)} chars"
            )

            return message

        except Exception as e:
            logger.error(f"Failed to generate AI response for conversation {conversation_id}: {e}")
            raise AIProviderError(f"AI response generation failed: {str(e)}", original_error=e)

    async def generate_room_response(
        self,
        room_id: int,
        ai_entity: AIEntity,
        user_id: int | None = None,
        include_memories: bool = True,
        in_reply_to_message_id: int | None = None,
    ) -> Message:
        """
        Generate AI response for a room message.

        :param room_id: Room ID to respond in
        :param ai_entity: AI entity that should respond
        :param user_id: User ID for personalized memory (optional, rooms use global memories)
        :param include_memories: Whether to include AI memories in context
        :param in_reply_to_message_id: Optional message reference for threading
        :return: Created message with AI response
        :raises AIProviderError: If LLM generation fails
        """
        try:
            # Build context (message history + memories)
            # Note: For rooms, user_id is optional (global memories only for now)
            messages, memory_context = await self.context_service.build_full_context(
                conversation_id=None,
                room_id=room_id,
                ai_entity=ai_entity,
                user_id=user_id or 0,  # Placeholder for rooms (TODO: room memory system)
                include_memories=include_memories,
            )

            # Enhance system prompt with memories if available
            system_prompt = ai_entity.system_prompt
            if memory_context:
                system_prompt = f"{system_prompt}\n\n{memory_context}"

            # Add anti-parroting instruction (prevents "Name:" prefix copying)
            system_prompt = f"""{system_prompt}

IMPORTANT: You respond directly as part of the conversation.
NEVER begin your responses with your name '{ai_entity.username}:' or similar prefix formats.
Respond naturally and directly."""

            # Generate response from LLM
            response_content = await self.ai_provider.generate_response(
                messages=messages,
                system_prompt=system_prompt,
                temperature=ai_entity.temperature,
                max_tokens=ai_entity.max_tokens,
            )

            # Post-processing: Remove any name prefixes (safety net)
            response_content = self._clean_parroting(response_content, ai_entity.username)

            # Save AI response as message
            message = await self.message_repo.create_room_message(
                room_id=room_id,
                content=response_content,
                sender_ai_id=ai_entity.id,
                in_reply_to_message_id=in_reply_to_message_id,
            )

            logger.info(
                f"AI '{ai_entity.username}' generated response in room {room_id}: {len(response_content)} chars"
            )

            return message

        except Exception as e:
            logger.error(f"Failed to generate AI response for room {room_id}: {e}")
            raise AIProviderError(f"AI response generation failed: {str(e)}", original_error=e)

    async def should_ai_respond(
        self,
        ai_entity: AIEntity,
        latest_message: Message,
        conversation_id: int | None = None,
        room_id: int | None = None,
    ) -> bool:
        """
        Determine if AI should respond to a message based on configured strategies.

        Checks performed in order:
        1. Own message check (never respond to self)
        2. Cooldown check (rate limiting)
        3. Strategy check (MENTION_ONLY, PROBABILISTIC, etc.)

        :param ai_entity: AI entity to check
        :param latest_message: Latest message in the conversation
        :param conversation_id: Conversation ID (for private/group chats)
        :param room_id: Room ID (for public rooms)
        :return: True if AI should respond, False otherwise
        """
        # 1. Don't respond to own messages
        if latest_message.sender_ai_id == ai_entity.id:
            return False

        # 2. Check cooldown (rate limiting)
        if ai_entity.cooldown_seconds is not None:
            is_on_cooldown = await self.cooldown_repo.is_on_cooldown(
                ai_entity_id=ai_entity.id,
                cooldown_seconds=ai_entity.cooldown_seconds,
                room_id=room_id,
                conversation_id=conversation_id,
            )
            if is_on_cooldown:
                logger.info(
                    "ai_response_skipped_cooldown",
                    ai_entity_id=ai_entity.id,
                    ai_name=ai_entity.username,
                    cooldown_seconds=ai_entity.cooldown_seconds,
                    room_id=room_id,
                    conversation_id=conversation_id,
                )
                return False

        # 3. Delegate to appropriate strategy handler
        if room_id:
            return self._should_respond_in_room(ai_entity, latest_message, room_id)
        elif conversation_id:
            return self._should_respond_in_conversation(ai_entity, latest_message, conversation_id)
        else:
            logger.warning("should_ai_respond called without room_id or conversation_id")
            return False

    async def check_provider_availability(self) -> bool:
        """
        Check if AI provider is available and configured.

        :return: True if provider is available, False otherwise
        """
        if not self.ai_provider:
            logger.warning("AI provider not configured")
            return False

        try:
            return await self.ai_provider.check_availability()
        except Exception as e:
            logger.error(f"AI provider availability check failed: {e}")
            return False

    def _should_respond_in_room(self, ai_entity: AIEntity, message: Message, room_id: int) -> bool:
        """
        Determine if AI should respond in a room based on room response strategy.

        Delegates to RoomResponseStrategyEvaluator for strategy pattern implementation.

        :param ai_entity: AI entity to check
        :param message: Latest message
        :param room_id: Room ID
        :return: True if AI should respond, False otherwise
        """
        return self.room_strategy_evaluator.should_respond(ai_entity, message, room_id)

    def _should_respond_in_conversation(self, ai_entity: AIEntity, message: Message, conversation_id: int) -> bool:
        """
        Determine if AI should respond in a conversation based on conversation response strategy.

        Delegates to ConversationResponseStrategyEvaluator for strategy pattern implementation.

        :param ai_entity: AI entity to check
        :param message: Latest message
        :param conversation_id: Conversation ID
        :return: True if AI should respond, False otherwise
        """
        return self.conversation_strategy_evaluator.should_respond(ai_entity, message, conversation_id)

    def _clean_parroting(self, text: str, ai_username: str) -> str:
        """
        Remove name prefixes from AI responses (LLM parroting prevention).

        This post-processing step prevents the AI from copying the "Name: message"
        format pattern from the conversation context into its responses.

        Community-proven approach from OpenAI forums for eliminating unwanted
        prefixes like "AI:", "You:", or "{ai_username}:" at the start of responses.

        :param text: Raw AI response text
        :param ai_username: AI entity's username (to remove specific patterns)
        :return: Cleaned text without name prefixes

        Examples:
            >>> _clean_parroting("Sokrates: Hello there!", "Sokrates")
            "Hello there!"
            >>> _clean_parroting("You: How are you?", "Sokrates")
            "How are you?"
            >>> _clean_parroting("Just a normal message", "Sokrates")
            "Just a normal message"
        """
        # Patterns to remove (order matters - most specific first)
        patterns = [
            rf"^{re.escape(ai_username)}:\s*",  # "Sokrates: "
            r"^You:\s*",  # "You: "
            r"^AI:\s*",  # "AI: "
            r"^Assistant:\s*",  # "Assistant: "
            r"^\[.*?\]:\s*",  # "[Name]: " or "[Assistant]: "
            r"^[A-Za-z_][A-Za-z0-9_]*:\s*",  # Generic "username: " (last resort)
        ]

        cleaned_text = text
        for pattern in patterns:
            cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.IGNORECASE)
            # Only remove first match to avoid over-cleaning
            if cleaned_text != text:
                logger.debug(
                    "parroting_cleaned",
                    ai_username=ai_username,
                    pattern=pattern,
                    original_length=len(text),
                    cleaned_length=len(cleaned_text),
                )
                break

        return cleaned_text.strip()
