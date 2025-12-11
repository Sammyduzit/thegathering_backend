"""Integration test for STM → LTM fact extraction flow (with mocked LLM)."""

import pytest

from app.core.config import settings
from app.models.ai_entity import AIEntity, AIEntityStatus, AIResponseStrategy
from app.repositories.ai_memory_repository import AIMemoryRepository
from app.services.memory.long_term_memory_service import LongTermMemoryService
from app.services.memory.short_term_memory_service import ShortTermMemoryService
from tests.fixtures.factories import ConversationFactory, MessageFactory, RoomFactory, UserFactory


class StubAIProvider:
    """Stub LLM provider returning a deterministic fact JSON."""

    async def generate_response(self, messages, system_prompt=None, temperature=None, max_tokens=None, **kwargs) -> str:
        return (
            '{"facts": ['
            '{"text": "User liebt Rust", "importance": 0.9, "participants": ["user1"], "theme": "Rust"}'
            ']}'  # noqa: E501
        )


class StubEmbeddingService:
    """Embedder stub returning a fixed-length vector."""

    async def embed_text(self, text: str) -> list[float]:
        return [0.1] * settings.embedding_dimensions


class StubKeywordExtractor:
    """Keyword extractor stub."""

    async def extract_keywords(self, text: str, max_keywords: int = 10, language: str = "en") -> list[str]:
        return ["Rust"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stm_to_ltm_flow(db_session):
    """End-to-end: STM chunk → LTM facts (mocked LLM) → STM deletion."""
    # Arrange: create conversation context
    room = await RoomFactory.create(db_session)
    conversation = await ConversationFactory.create_private_conversation(db_session, room=room)
    user = await UserFactory.create(db_session, username="user1")

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

    # Create a few conversation messages
    messages = []
    for i in range(3):
        msg = await MessageFactory.create_conversation_message(
            db_session,
            sender=user,
            conversation=conversation,
            content=f"Message {i} about Rust",
        )
        messages.append(msg)

    memory_repo = AIMemoryRepository(db_session)

    # Create STM chunk
    stm_service = ShortTermMemoryService(
        memory_repo=memory_repo,
        keyword_extractor=StubKeywordExtractor(),
    )
    stm_chunk = await stm_service.create_short_term_chunk(
        entity_id=ai_entity.id,
        user_ids=[user.id],
        conversation_id=conversation.id,
        chunk_messages=messages,
        chunk_index=0,
        start_idx=0,
        end_idx=len(messages) - 1,
    )

    # LTM extraction with stubbed provider/embedding/keywords
    ltm_service = LongTermMemoryService(
        memory_repo=memory_repo,
        embedding_service=StubEmbeddingService(),
        keyword_extractor=StubKeywordExtractor(),
        ai_provider=StubAIProvider(),
    )

    ltm_memories = await ltm_service.create_long_term_from_chunks(
        entity_id=ai_entity.id,
        user_ids=[user.id],
        conversation_id=conversation.id,
        stm_chunks=[stm_chunk],
    )

    assert len(ltm_memories) == 1
    created = ltm_memories[0]
    assert created.memory_metadata["type"] == "long_term"
    assert created.memory_metadata.get("fact_hash")
    assert created.summary == "Rust"
    assert created.memory_content["fact"]["text"] == "User liebt Rust"

    # Delete STM chunks (simulating task cleanup)
    deleted_count = await memory_repo.delete_short_term_chunks(
        conversation_id=conversation.id,
        entity_id=ai_entity.id,
    )
    assert deleted_count == 1
