"""Integration tests for pgvector similarity via AIMemoryRepository."""

import pytest

from app.core.config import settings
from app.models.ai_entity import AIEntity, AIEntityStatus, AIResponseStrategy
from app.models.ai_memory import AIMemory
from app.repositories.ai_memory_repository import AIMemoryRepository
from tests.fixtures import ConversationFactory, RoomFactory, UserFactory


def _basis_vector(dim_index: int) -> list[float]:
    """Return vector with 1.0 at dim_index and 0 elsewhere."""
    vec = [0.0] * settings.embedding_dimensions
    vec[dim_index % settings.embedding_dimensions] = 1.0
    return vec


async def _create_ai_entity(db_session) -> AIEntity:
    entity = AIEntity(
        username="vector_test_ai",
        description="Used for pgvector tests",
        system_prompt="You help verify embeddings.",
        model_name="gpt-4o-mini",
        temperature=0.7,
        max_tokens=256,
        status=AIEntityStatus.ONLINE,
        room_response_strategy=AIResponseStrategy.ROOM_MENTION_ONLY,
        conversation_response_strategy=AIResponseStrategy.CONV_ON_QUESTIONS,
    )
    db_session.add(entity)
    await db_session.commit()
    await db_session.refresh(entity)
    return entity


@pytest.mark.integration
@pytest.mark.asyncio
async def test_vector_search_orders_by_similarity(db_session):
    """Closest embedding should be returned first."""
    entity = await _create_ai_entity(db_session)
    user = await UserFactory.create(db_session)
    repo = AIMemoryRepository(db_session)

    mem_close = AIMemory(
        entity_id=entity.id,
        user_ids=[user.id],
        conversation_id=None,
        summary="Close memory",
        memory_content={},
        keywords=["close"],
        importance_score=1.0,
        embedding=_basis_vector(0),
        memory_metadata={"type": "personality"},
    )
    mem_far = AIMemory(
        entity_id=entity.id,
        user_ids=[user.id],
        conversation_id=None,
        summary="Far memory",
        memory_content={},
        keywords=["far"],
        importance_score=1.0,
        embedding=_basis_vector(1),
        memory_metadata={"type": "personality"},
    )
    mem_close = await repo.create(mem_close)
    mem_far = await repo.create(mem_far)

    results = await repo.vector_search(entity_id=entity.id, embedding=_basis_vector(0), limit=2)

    assert [m.id for m in results] == [mem_close.id, mem_far.id]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_vector_search_respects_filters(db_session):
    """vector_search should honor user/conversation/type filters."""
    entity = await _create_ai_entity(db_session)
    room = await RoomFactory.create(db_session)
    conversation1 = await ConversationFactory.create_private_conversation(db_session, room=room)
    conversation2 = await ConversationFactory.create_private_conversation(db_session, room=room)
    user1 = await UserFactory.create(db_session)
    user2 = await UserFactory.create(db_session)
    repo = AIMemoryRepository(db_session)

    mem_user1 = AIMemory(
        entity_id=entity.id,
        user_ids=[user1.id],
        conversation_id=conversation1.id,
        summary="User1 convo memory",
        memory_content={},
        keywords=["user1"],
        importance_score=1.0,
        embedding=_basis_vector(0),
        memory_metadata={"type": "short_term"},
    )
    mem_user2 = AIMemory(
        entity_id=entity.id,
        user_ids=[user2.id],
        conversation_id=conversation2.id,
        summary="User2 convo memory",
        memory_content={},
        keywords=["user2"],
        importance_score=1.0,
        embedding=_basis_vector(1),
        memory_metadata={"type": "long_term"},
    )
    mem_user1 = await repo.create(mem_user1)
    await repo.create(mem_user2)

    results = await repo.vector_search(
        entity_id=entity.id,
        embedding=_basis_vector(0),
        conversation_id=conversation1.id,
        limit=5,
    )

    assert len(results) == 1
    assert results[0].id == mem_user1.id
