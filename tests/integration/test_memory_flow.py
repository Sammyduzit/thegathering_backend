"""Integration tests for memory creation flow."""

import pytest

from app.core.config import settings
from app.models.ai_entity import AIEntity, AIEntityStatus, AIResponseStrategy
from app.repositories.ai_memory_repository import AIMemoryRepository
from app.services.memory.long_term_memory_service import LongTermMemoryService
from app.services.memory.short_term_memory_service import ShortTermMemoryService
from app.services.text_processing.text_chunking_service import TextChunkingService
from tests.fixtures import ConversationFactory, MessageFactory, RoomFactory, UserFactory


class FakeEmbeddingService:
    """Simple embedding stub returning deterministic vectors."""

    async def embed_text(self, text: str) -> list[float]:
        return [float(len(text))] * settings.embedding_dimensions

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] * settings.embedding_dimensions for text in texts]


class FakeKeywordExtractor:
    """Async keyword extractor stub."""

    async def extract_keywords(self, text: str, max_keywords: int = 10, language: str = "en") -> list[str]:
        return ["topic"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_short_to_long_term_memory_flow(db_session, message_repo):
    """Verify that short-term and long-term services persist memories end-to-end."""
    # Arrange: create conversation context with human messages
    room = await RoomFactory.create(db_session)
    conversation = await ConversationFactory.create_private_conversation(db_session, room=room)
    user = await UserFactory.create(db_session)

    ai_entity = AIEntity(
        username="memory_ai",
        description="Test AI",
        system_prompt="You summarize conversations.",
        model_name="gpt-4o-mini",
        temperature=0.7,
        max_tokens=512,
        status=AIEntityStatus.ONLINE,
        room_response_strategy=AIResponseStrategy.ROOM_MENTION_ONLY,
        conversation_response_strategy=AIResponseStrategy.CONV_ON_QUESTIONS,
    )
    db_session.add(ai_entity)
    await db_session.commit()
    await db_session.refresh(ai_entity)

    messages = []
    for i in range(3):
        msg = await MessageFactory.create_conversation_message(
            db_session,
            sender=user,
            conversation=conversation,
            content=f"Message {i} about memories",
        )
        messages.append(msg)

    memory_repo = AIMemoryRepository(db_session)
    short_term_service = ShortTermMemoryService(
        memory_repo=memory_repo,
        keyword_extractor=FakeKeywordExtractor(),
    )
    long_term_service = LongTermMemoryService(
        memory_repo=memory_repo,
        message_repo=message_repo,
        embedding_service=FakeEmbeddingService(),
        chunking_service=TextChunkingService(chunk_size=50, chunk_overlap=0),
        keyword_extractor=FakeKeywordExtractor(),
    )

    entity_id = ai_entity.id
    user_ids = [user.id]

    # Act: create short-term memory from recent messages
    short_memory = await short_term_service.create_short_term_memory(
        entity_id=entity_id,
        user_ids=user_ids,
        conversation_id=conversation.id,
        messages=messages,
    )

    # Act: archive conversation into long-term memories
    long_memories = await long_term_service.create_long_term_archive(
        entity_id=entity_id,
        user_ids=user_ids,
        conversation_id=conversation.id,
    )

    # Assert short-term persisted with metadata
    assert short_memory.memory_metadata["type"] == "short_term"
    assert short_memory.memory_content["message_count"] == 3

    # Assert long-term memories created and stored
    assert len(long_memories) >= 1
    assert all(memory.memory_metadata["type"] == "long_term" for memory in long_memories)
    assert all(memory.embedding is not None for memory in long_memories)

    # Verify repository can fetch both types by entity
    stored = await memory_repo.get_entity_memories(entity_id=entity_id, limit=10)
    memory_types = {memory.memory_metadata.get("type") for memory in stored if memory.memory_metadata}
    assert {"short_term", "long_term"}.issubset(memory_types)
