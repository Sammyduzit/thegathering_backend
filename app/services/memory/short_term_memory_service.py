from app.interfaces.keyword_extractor import IKeywordExtractor
from app.models.ai_memory import AIMemory
from app.models.message import Message
from app.repositories.ai_memory_repository import AIMemoryRepository


class ShortTermMemoryService:
    """Service for creating short-term conversation memories."""

    def __init__(
        self,
        memory_repo: AIMemoryRepository,
        keyword_extractor: IKeywordExtractor,
    ):
        self.memory_repo = memory_repo
        self.keyword_extractor = keyword_extractor

    async def create_short_term_memory(
        self,
        entity_id: int,
        user_ids: list[int],
        conversation_id: int,
        messages: list[Message],
    ) -> AIMemory:
        """
        Create short-term memory from recent conversation messages.

        - Takes last 20 messages
        - Filters only user messages (no system, no AI)
        - Extracts keywords (YAKE)
        - Creates simple summary
        - NO embedding (fast!)

        :param entity_id: AI entity ID
        :param user_ids: List of user IDs (participants in conversation)
        :param conversation_id: Conversation ID
        :param messages: List of recent messages
        :return: Created AIMemory instance
        """
        # Get last 20 messages
        recent = messages[-20:] if len(messages) > 20 else messages

        # Filter: Only user messages (not system, not AI)
        user_messages = [m for m in recent if m.sender_user_id is not None and m.message_type != "system"]

        if not user_messages:
            # No user messages, create minimal memory
            memory = AIMemory(
                entity_id=entity_id,
                user_ids=user_ids,
                conversation_id=conversation_id,
                summary="No recent user messages",
                memory_content={"message_count": 0},
                keywords=[],
                importance_score=0.5,
                embedding=None,  # No embedding for short-term
                memory_metadata={"type": "short_term"},
            )
            return await self.memory_repo.create(memory)

        # Combine user message content for keyword extraction
        combined_text = " ".join([m.content for m in user_messages])

        # Extract keywords
        keywords = await self._extract_keywords(combined_text)

        # Create simple summary (first 200 chars of first user message)
        first_message = user_messages[0].content
        summary = first_message[:200] + "..." if len(first_message) > 200 else first_message

        # Create memory
        memory = AIMemory(
            entity_id=entity_id,
            user_ids=user_ids,
            conversation_id=conversation_id,
            summary=summary,
            memory_content={
                "message_count": len(user_messages),
                "last_messages": [
                    {"sender": m.sender_user_id, "content": m.content}
                    for m in user_messages[-5:]  # Store last 5
                ],
            },
            keywords=keywords,
            importance_score=1.0,
            embedding=None,  # No embedding for short-term
            memory_metadata={"type": "short_term"},
        )

        return await self.memory_repo.create(memory)

    async def _extract_keywords(self, text: str) -> list[str]:
        """Extract keywords from text using keyword extractor."""
        if not text or not text.strip():
            return []

        try:
            return await self.keyword_extractor.extract_keywords(text, max_keywords=10)
        except Exception:
            return []
