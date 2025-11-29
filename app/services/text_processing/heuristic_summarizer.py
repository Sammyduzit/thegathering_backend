"""
Heuristic Memory Summarizer implementation.

Provides simple rule-based conversation summarization without requiring LLM calls.
Fast and cost-effective for initial implementation.
"""

import structlog

from app.interfaces.memory_summarizer import IMemorySummarizer, MemorySummarizationError
from app.models.ai_entity import AIEntity
from app.models.message import Message

logger = structlog.get_logger(__name__)


class HeuristicMemorySummarizer(IMemorySummarizer):
    """Heuristic-based memory summarizer implementation."""

    async def summarize(
        self,
        messages: list[Message],
        ai_entity: AIEntity | None = None,
    ) -> str:
        """
        Generate summary from conversation messages using heuristic rules.

        Strategy:
        1. Extract participants from messages
        2. Determine topic from first user message (first 100 chars)
        3. Build simple template: "{AI} talked with {participants} about {topic}"

        :param messages: List of messages to summarize (chronological order)
        :param ai_entity: AI entity object (for contextualized summaries using username)
        :return: Generated summary text (1-2 sentences), e.g., 'Assistant Alpha discussed FastAPI setup and database configuration with Alice'
        :raises MemorySummarizationError: If summarization fails
        """
        try:
            if not messages:
                return "Empty conversation"

            # Extract participants (unique sender names)
            participants = self._extract_participants(messages)

            # Determine topic from first user message
            topic = self._determine_topic(messages)

            # Build summary
            if ai_entity:
                # Filter out AI from participants list (using username for consistency)
                human_participants = [p for p in participants if p != ai_entity.username]
                if human_participants:
                    participants_str = ", ".join(human_participants)
                    summary = f"{ai_entity.username} talked with {participants_str} about {topic}"
                else:
                    summary = f"{ai_entity.username} discussed {topic}"
            else:
                participants_str = ", ".join(participants)
                summary = f"Conversation with {participants_str} about {topic}"

            logger.debug(
                "generated_heuristic_summary",
                message_count=len(messages),
                summary=summary,
                participants=participants,
            )

            return summary

        except Exception as e:
            logger.error("heuristic_summarization_failed", error=str(e))
            raise MemorySummarizationError(f"Failed to generate summary: {str(e)}", original_error=e)

    def _extract_participants(self, messages: list[Message]) -> list[str]:
        """
        Extract unique participant names from messages.

        :param messages: List of messages
        :return: List of unique participant names
        """
        participants = set()

        for msg in messages:
            if msg.sender_user_id and hasattr(msg, "sender_user") and msg.sender_user:
                participants.add(msg.sender_user.username)
            elif msg.sender_ai_id and hasattr(msg, "sender_ai") and msg.sender_ai:
                participants.add(msg.sender_ai.username)

        return sorted(participants)

    def _determine_topic(self, messages: list[Message]) -> str:
        """
        Determine conversation topic from messages.

        Strategy: Use first 100 chars of first user message as topic indicator.

        :param messages: List of messages
        :return: Topic string (max 100 chars)
        """
        # Find first user (non-AI) message
        for msg in messages:
            if msg.sender_user_id and msg.content:
                # First 100 chars, clean whitespace
                topic = " ".join(msg.content[:100].split())
                if len(msg.content) > 100:
                    topic += "..."
                return topic

        # Fallback: any message
        if messages and messages[0].content:
            topic = " ".join(messages[0].content[:100].split())
            if len(messages[0].content) > 100:
                topic += "..."
            return topic

        # Final fallback
        return "general conversation"
