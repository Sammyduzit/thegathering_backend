"""AI Response Strategy Pattern Implementation.

Separates response decision logic for rooms and conversations.
"""

import random
from abc import ABC, abstractmethod

import structlog

from app.models.ai_entity import AIEntity, AIResponseStrategy
from app.models.message import Message

logger = structlog.get_logger(__name__)


class ResponseStrategyEvaluator(ABC):
    """Base class for evaluating if AI should respond based on strategy."""

    @abstractmethod
    def should_respond(self, ai_entity: AIEntity, message: Message, context_id: int) -> bool:
        """
        Evaluate if AI should respond to message.

        :param ai_entity: AI entity to evaluate
        :param message: Message to respond to
        :param context_id: Room or Conversation ID
        :return: True if AI should respond, False otherwise
        """
        pass

    def _is_ai_mentioned(self, ai_entity: AIEntity, message: Message) -> bool:
        """Check if AI is mentioned in message."""
        content = message.content.lower()
        return ai_entity.username.lower() in content

    def _is_question(self, message: Message) -> bool:
        """Check if message appears to be a question."""
        content = message.content.lower()
        question_indicators = ["?", "what", "how", "why", "when", "where", "who", "can you", "could you"]
        return any(indicator in content for indicator in question_indicators)


class RoomResponseStrategyEvaluator(ResponseStrategyEvaluator):
    """Evaluates room response strategies."""

    def should_respond(self, ai_entity: AIEntity, message: Message, room_id: int) -> bool:
        """
        Determine if AI should respond in a room based on room response strategy.

        :param ai_entity: AI entity to check
        :param message: Latest message
        :param room_id: Room ID
        :return: True if AI should respond, False otherwise
        """
        strategy = ai_entity.room_response_strategy

        # NO_RESPONSE: Never respond (after graceful goodbye)
        if strategy == AIResponseStrategy.NO_RESPONSE:
            return False

        ai_mentioned = self._is_ai_mentioned(ai_entity, message)

        # ROOM_MENTION_ONLY: Only respond when mentioned
        if strategy == AIResponseStrategy.ROOM_MENTION_ONLY:
            if ai_mentioned:
                logger.info("ai_room_mention_response", ai_username=ai_entity.username, room_id=room_id)
                return True
            return False

        # ROOM_PROBABILISTIC: Respond based on probability (higher chance if mentioned)
        if strategy == AIResponseStrategy.ROOM_PROBABILISTIC:
            probability = 1.0 if ai_mentioned else ai_entity.response_probability
            should_respond = random.random() < probability

            if should_respond:
                logger.info(
                    "ai_room_probabilistic_response",
                    ai_username=ai_entity.username,
                    room_id=room_id,
                    probability=probability,
                )
            return should_respond

        # ROOM_ACTIVE: Respond to most messages (filter very short ones)
        if strategy == AIResponseStrategy.ROOM_ACTIVE:
            # Always respond if mentioned
            if ai_mentioned:
                return True

            # Filter very short messages (like "ok", "lol")
            if len(message.content.strip()) < 3:
                return False

            logger.info("ai_room_active_response", ai_username=ai_entity.username, room_id=room_id)
            return True

        # Unknown strategy
        logger.warning("unknown_room_strategy", strategy=strategy, ai_username=ai_entity.username)
        return False


class ConversationResponseStrategyEvaluator(ResponseStrategyEvaluator):
    """Evaluates conversation response strategies."""

    def should_respond(self, ai_entity: AIEntity, message: Message, conversation_id: int) -> bool:
        """
        Determine if AI should respond in a conversation based on conversation response strategy.

        :param ai_entity: AI entity to check
        :param message: Latest message
        :param conversation_id: Conversation ID
        :return: True if AI should respond, False otherwise
        """
        strategy = ai_entity.conversation_response_strategy

        # NO_RESPONSE: Never respond (after graceful goodbye)
        if strategy == AIResponseStrategy.NO_RESPONSE:
            return False

        ai_mentioned = self._is_ai_mentioned(ai_entity, message)

        # CONV_EVERY_MESSAGE: Respond to every message
        if strategy == AIResponseStrategy.CONV_EVERY_MESSAGE:
            logger.info(
                "ai_conversation_every_message", ai_username=ai_entity.username, conversation_id=conversation_id
            )
            return True

        # CONV_ON_QUESTIONS: Only respond to questions
        if strategy == AIResponseStrategy.CONV_ON_QUESTIONS:
            is_question = self._is_question(message)
            if is_question:
                logger.info(
                    "ai_conversation_question_response", ai_username=ai_entity.username, conversation_id=conversation_id
                )
                return True
            return False

        # CONV_SMART: Respond to questions OR mentions
        if strategy == AIResponseStrategy.CONV_SMART:
            is_question = self._is_question(message)
            if ai_mentioned or is_question:
                logger.info(
                    "ai_conversation_smart_response",
                    ai_username=ai_entity.username,
                    conversation_id=conversation_id,
                    mentioned=ai_mentioned,
                    question=is_question,
                )
                return True
            return False

        # Unknown strategy
        logger.warning("unknown_conversation_strategy", strategy=strategy, ai_username=ai_entity.username)
        return False
