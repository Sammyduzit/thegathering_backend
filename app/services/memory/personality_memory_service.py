from app.interfaces.embedding_service import EmbeddingServiceError, IEmbeddingService
from app.interfaces.keyword_extractor import IKeywordExtractor
from app.models.ai_memory import AIMemory
from app.repositories.ai_memory_repository import AIMemoryRepository
from app.services.memory.base_memory_service import BaseMemoryService
from app.services.text_processing.text_chunking_service import TextChunkingService


class PersonalityMemoryService(BaseMemoryService):
    """Service for uploading personality knowledge base (books, docs, etc)."""

    def __init__(
        self,
        memory_repo: AIMemoryRepository,
        embedding_service: IEmbeddingService,
        chunking_service: TextChunkingService,
        keyword_extractor: IKeywordExtractor,
    ):
        super().__init__(keyword_extractor)
        self.memory_repo = memory_repo
        self.embedding_service = embedding_service
        self.chunking_service = chunking_service

    async def upload_personality(
        self,
        entity_id: int,
        text: str,
        category: str,
        metadata: dict,
    ) -> list[AIMemory]:
        """
        Upload personality knowledge from text (books, documents, etc).

        - Global memory (user_ids = [], conversation_id = NULL)
        - Chunks text into manageable pieces
        - Extracts keywords per chunk
        - Generates embeddings per chunk (batch)
        - Creates AIMemory per chunk

        :param entity_id: AI entity ID
        :param text: Text content to upload
        :param category: Category (e.g., "books", "docs")
        :param metadata: Additional metadata (e.g., book_title, chapter)
        :return: List of created AIMemory instances (one per chunk)
        :raises Exception: If embedding generation fails (fail fast)
        """
        if not text or not text.strip():
            return []

        # Chunk text
        chunks = self.chunking_service.chunk_text(text)

        if not chunks:
            return []

        # Extract keywords per chunk (sequential for-loop like LongTermMemoryService)
        chunk_keywords = []
        for chunk in chunks:
            keywords = await self._extract_keywords(chunk)
            chunk_keywords.append(keywords)

        # Generate embeddings per chunk (batch)
        try:
            embeddings = await self.embedding_service.embed_batch(chunks)
        except Exception as e:
            # Fail fast: Embedding error = Personality upload fails
            raise EmbeddingServiceError(f"Personality upload failed: {e}", original_error=e)

        # Create AIMemory per chunk
        memories = []
        for i, (chunk, keywords, embedding) in enumerate(zip(chunks, chunk_keywords, embeddings)):
            summary = self._truncate_summary(chunk, max_length=200)

            memory = AIMemory(
                entity_id=entity_id,
                user_ids=[],  # Global (not user-specific)
                conversation_id=None,  # Not conversation-bound
                summary=summary,
                memory_content={"full_text": chunk},
                keywords=keywords,
                importance_score=1.0,
                embedding=embedding,
                memory_metadata={
                    "type": "personality",
                    "category": category,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    **metadata,  # Include additional metadata
                },
            )

            created = await self.memory_repo.create(memory)
            memories.append(created)

        return memories
