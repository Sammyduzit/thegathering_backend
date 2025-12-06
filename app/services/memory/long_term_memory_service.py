from app.interfaces.embedding_service import EmbeddingServiceError, IEmbeddingService
from app.interfaces.keyword_extractor import IKeywordExtractor
from app.models.ai_memory import AIMemory
from app.repositories.ai_memory_repository import AIMemoryRepository
from app.repositories.message_repository import MessageRepository
from app.services.memory.base_memory_service import BaseMemoryService
from app.services.text_processing.text_chunking_service import TextChunkingService


class LongTermMemoryService(BaseMemoryService):
    """Service for creating long-term conversation archives."""

    def __init__(
        self,
        memory_repo: AIMemoryRepository,
        message_repo: MessageRepository,
        embedding_service: IEmbeddingService,
        chunking_service: TextChunkingService,
        keyword_extractor: IKeywordExtractor,
    ):
        """
        Initialize long-term memory service.

        :param memory_repo: AI memory repository
        :param message_repo: Message repository
        :param embedding_service: Embedding service (Google/OpenAI)
        :param chunking_service: Text chunking service
        :param keyword_extractor: Keyword extractor (YAKE by default)
        """
        super().__init__(keyword_extractor)
        self.memory_repo = memory_repo
        self.message_repo = message_repo
        self.embedding_service = embedding_service
        self.chunking_service = chunking_service

    async def create_long_term_archive(
        self,
        entity_id: int,
        user_ids: list[int],
        conversation_id: int,
    ) -> list[AIMemory]:
        """
        Create long-term memory archive from entire conversation.

        - Fetches ALL messages from conversation
        - Chunks text into manageable pieces
        - Extracts keywords per chunk
        - Generates embeddings per chunk (batch)
        - Creates AIMemory per chunk

        :param entity_id: AI entity ID
        :param user_ids: List of user IDs (participants in conversation)
        :param conversation_id: Conversation ID
        :return: List of created AIMemory instances (one per chunk)
        :raises Exception: If embedding generation fails (fail fast)
        """
        # Fetch all messages from conversation
        messages, _ = await self.message_repo.get_conversation_messages(
            conversation_id=conversation_id,
            page=1,
            page_size=10000,  # High limit to get all
        )

        if not messages:
            return []

        # Sort messages in chronological order (oldest first) for proper archiving
        # Note: get_conversation_messages returns newest first (DESC), so we reverse
        messages_chronological = sorted(messages, key=lambda m: m.sent_at)

        # Combine all message content with usernames
        combined_text = "\n\n".join([f"{m.sender_username}: {m.content}" for m in messages_chronological])

        # Chunk text
        chunks = self.chunking_service.chunk_text(combined_text)

        if not chunks:
            return []

        # Extract keywords per chunk (async)
        chunk_keywords = []
        for chunk in chunks:
            keywords = await self._extract_keywords(chunk)
            chunk_keywords.append(keywords)

        # Generate embeddings per chunk (batch)
        try:
            embeddings = await self.embedding_service.embed_batch(chunks)
        except Exception as e:
            # Fail fast: Embedding error = Memory creation fails
            raise EmbeddingServiceError(f"Long-term memory creation failed: {e}", original_error=e)

        # Create AIMemory per chunk
        memories = []
        for i, (chunk, keywords, embedding) in enumerate(zip(chunks, chunk_keywords, embeddings)):
            summary = self._truncate_summary(chunk, max_length=200)

            memory = AIMemory(
                entity_id=entity_id,
                user_ids=user_ids,
                conversation_id=conversation_id,
                summary=summary,
                memory_content={"full_text": chunk},
                keywords=keywords,
                importance_score=1.0,
                embedding=embedding,
                memory_metadata={
                    "type": "long_term",
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                },
            )

            created = await self.memory_repo.create(memory)
            memories.append(created)

        return memories
