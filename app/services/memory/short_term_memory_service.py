from app.interfaces.keyword_extractor import IKeywordExtractor
from app.models.ai_memory import AIMemory
from app.models.message import Message
from app.repositories.ai_memory_repository import AIMemoryRepository
from app.services.memory.base_memory_service import BaseMemoryService


class ShortTermMemoryService(BaseMemoryService):
    """Service for creating chunked short-term conversation memories.

    NEW ARCHITECTURE: Creates individual memory chunks from message segments.
    Each chunk contains SHORT_TERM_CHUNK_SIZE messages (default: 24).
    """

    def __init__(
        self,
        memory_repo: AIMemoryRepository,
        keyword_extractor: IKeywordExtractor,
    ):
        super().__init__(keyword_extractor)
        self.memory_repo = memory_repo

    async def create_short_term_chunk(
        self,
        entity_id: int,
        user_ids: list[int],
        conversation_id: int,
        chunk_messages: list[Message],
        chunk_index: int,
        start_idx: int,
        end_idx: int,
    ) -> AIMemory:
        """
        Create a single short-term memory chunk.

        This method creates ONE chunk from the provided messages.
        Orchestration of which chunks to create happens in the caller (background task).

        :param entity_id: AI entity ID
        :param user_ids: List of user IDs (participants)
        :param conversation_id: Conversation ID
        :param chunk_messages: Messages for this specific chunk (already filtered)
        :param chunk_index: Index of this chunk (0, 1, 2, ...)
        :param start_idx: Starting message index in full conversation
        :param end_idx: Ending message index in full conversation
        :return: Created AIMemory instance for this chunk
        """
        # Combine all chunk message content for keyword extraction
        combined_text = " ".join([m.content for m in chunk_messages])

        # Extract keywords (from THIS chunk only)
        keywords = await self._extract_keywords(combined_text)

        # Create simple label for chunk
        summary = f"Chunk {chunk_index} (Msgs {start_idx}–{end_idx})"

        # Create memory
        memory = AIMemory(
            entity_id=entity_id,
            user_ids=user_ids,
            conversation_id=conversation_id,
            summary=summary,
            memory_content={
                "message_count": len(chunk_messages),
                "messages": [
                    {
                        "sender_user_id": m.sender_user_id,
                        "sender_ai_id": m.sender_ai_id,
                        "content": m.content,
                    }
                    for m in chunk_messages  # Store ALL messages in chunk!
                ],
            },
            keywords=keywords,
            importance_score=1.0,
            embedding=None,  # No embedding for short-term
            memory_metadata={
                "type": "short_term",
                "chunk_index": chunk_index,
                "message_range": f"{start_idx}-{end_idx}",
            },
        )

        return await self.memory_repo.create(memory)
