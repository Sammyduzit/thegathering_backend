"""Integration tests for ARQ worker tasks."""

import pytest
from arq import Retry

from app.interfaces.ai_provider import AIProviderError
from app.repositories.ai_memory_repository import AIMemoryRepository
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.message_repository import MessageRepository
from app.workers.tasks import check_and_generate_ai_response, create_long_term_memory_task


class FakeDbManager:
    """Minimal db_manager compatible with ARQ tasks."""

    def __init__(self, session):
        self._session = session

    async def get_session(self):
        yield self._session


class FakeProvider:
    """Stub AI provider that returns a fixed response."""

    def __init__(self, content="AI reply"):
        self.content = content

    async def generate_response(self, **kwargs):
        return self.content

    async def check_availability(self):
        return True


class FakeKeywordExtractor:
    async def extract_keywords(self, text, max_keywords=10, language="en"):
        return []


class FakeEmbeddingService:
    def __init__(self, raise_error: bool = False):
        self.raise_error = raise_error

    async def embed_batch(self, texts):
        if self.raise_error:
            raise ValueError("embedding failed")
        # Return minimal vector-like lists
        return [[0.0] * 3 for _ in texts]


class FakeLongEmbeddingService(FakeEmbeddingService):
    async def embed_batch(self, texts):
        if self.raise_error:
            raise ValueError("embedding failed")
        return [[0.0] * 1536 for _ in texts]


def _monkeypatch_ai_stack(monkeypatch, *, provider=None, embedding_service=None, keyword_extractor=None):
    """Monkeypatch AI dependencies used inside tasks."""
    if provider:
        monkeypatch.setattr("app.workers.tasks.OpenAIProvider", lambda *a, **k: provider)
    if embedding_service is not None:
        monkeypatch.setattr("app.workers.tasks.get_embedding_service", lambda: embedding_service)
        monkeypatch.setattr("app.workers.tasks.get_memory_retriever", lambda **kwargs: None)
        monkeypatch.setattr("app.workers.tasks.create_embedding_service", lambda: embedding_service)
    if keyword_extractor is not None:
        monkeypatch.setattr("app.workers.tasks.create_keyword_extractor", lambda: keyword_extractor)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_check_and_generate_ai_response_happy_path(
    db_session,
    user_factory,
    conversation_factory,
    message_factory,
    ai_entity_factory,
    monkeypatch,
):
    user = await user_factory.create(db_session)
    conversation = await conversation_factory.create_private_conversation(db_session)
    conv_repo = ConversationRepository(db_session)
    await conv_repo.add_participant(conversation_id=conversation.id, user_id=user.id)
    ai_entity = await ai_entity_factory.create_online(db_session)
    await conv_repo.add_participant(conversation_id=conversation.id, ai_entity_id=ai_entity.id)
    message = await message_factory.create_conversation_message(
        db_session, sender=user, conversation=conversation, content="Hello there?"
    )

    _monkeypatch_ai_stack(
        monkeypatch,
        provider=FakeProvider("Sure, here's a response."),
        embedding_service=FakeEmbeddingService(),
        keyword_extractor=FakeKeywordExtractor(),
    )

    ctx = {"db_manager": FakeDbManager(db_session), "job_try": 1}

    result = await check_and_generate_ai_response(
        ctx,
        message_id=message.id,
        conversation_id=conversation.id,
        ai_entity_id=ai_entity.id,
    )

    assert result["ai_message_id"] > 0
    repo = MessageRepository(db_session)
    ai_message = await repo.get_by_id(result["ai_message_id"])
    assert ai_message is not None
    assert ai_message.sender_ai_id == ai_entity.id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_check_and_generate_ai_response_provider_error_raises_retry(
    db_session,
    user_factory,
    conversation_factory,
    message_factory,
    ai_entity_factory,
    monkeypatch,
):
    class ErrorProvider:
        async def generate_response(self, **kwargs):
            raise AIProviderError("boom")

    user = await user_factory.create(db_session)
    conversation = await conversation_factory.create_private_conversation(db_session)
    conv_repo = ConversationRepository(db_session)
    await conv_repo.add_participant(conversation_id=conversation.id, user_id=user.id)
    ai_entity = await ai_entity_factory.create_online(db_session)
    await conv_repo.add_participant(conversation_id=conversation.id, ai_entity_id=ai_entity.id)
    message = await message_factory.create_conversation_message(
        db_session, sender=user, conversation=conversation, content="Will you answer?"
    )

    _monkeypatch_ai_stack(
        monkeypatch,
        provider=ErrorProvider(),
        embedding_service=FakeEmbeddingService(),
        keyword_extractor=FakeKeywordExtractor(),
    )

    ctx = {"db_manager": FakeDbManager(db_session), "job_try": 1}

    with pytest.raises(Retry):
        await check_and_generate_ai_response(
            ctx,
            message_id=message.id,
            conversation_id=conversation.id,
            ai_entity_id=ai_entity.id,
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_long_term_memory_task_happy_path(
    db_session,
    user_factory,
    conversation_factory,
    message_factory,
    ai_entity_factory,
    monkeypatch,
):
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload
    from app.models.message import Message
    from app.services.memory.short_term_memory_service import ShortTermMemoryService

    user = await user_factory.create(db_session)
    conversation = await conversation_factory.create_private_conversation(db_session)
    conv_repo = ConversationRepository(db_session)
    await conv_repo.add_participant(conversation_id=conversation.id, user_id=user.id)
    ai_entity = await ai_entity_factory.create_online(db_session)

    # Create messages
    messages = []
    for i in range(2):
        msg = await message_factory.create_conversation_message(
            db_session, sender=user, conversation=conversation, content=f"Message {i} for memory."
        )
        messages.append(msg)

    # Reload messages with eager loading
    message_ids = [m.id for m in messages]
    query = (
        select(Message)
        .where(Message.id.in_(message_ids))
        .options(selectinload(Message.sender_user), selectinload(Message.sender_ai))
        .order_by(Message.created_at)
    )
    result_query = await db_session.execute(query)
    messages = list(result_query.scalars().all())

    # Create STM chunk
    memory_repo = AIMemoryRepository(db_session)
    stm_service = ShortTermMemoryService(
        memory_repo=memory_repo,
        keyword_extractor=FakeKeywordExtractor(),
    )
    await stm_service.create_short_term_chunk(
        entity_id=ai_entity.id,
        user_ids=[user.id],
        conversation_id=conversation.id,
        chunk_messages=messages,
        chunk_index=0,
        start_idx=0,
        end_idx=len(messages) - 1,
    )

    _monkeypatch_ai_stack(
        monkeypatch,
        embedding_service=FakeLongEmbeddingService(),
        keyword_extractor=FakeKeywordExtractor(),
    )

    ctx = {"db_manager": FakeDbManager(db_session), "job_try": 1}

    result = await create_long_term_memory_task(
        ctx,
        ai_entity_id=ai_entity.id,
        conversation_id=conversation.id,
    )

    assert result["memory_count"] > 0
    assert result["stm_chunks_processed"] > 0
    memories = await memory_repo.get_entity_memories(ai_entity.id, limit=10)
    assert memories
    assert user.id in memories[0].user_ids


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_long_term_memory_task_skips_without_users(
    db_session,
    conversation_factory,
    ai_entity_factory,
    monkeypatch,
):
    conversation = await conversation_factory.create_private_conversation(db_session)
    ai_entity = await ai_entity_factory.create_online(db_session)

    _monkeypatch_ai_stack(
        monkeypatch,
        embedding_service=FakeLongEmbeddingService(),
        keyword_extractor=FakeKeywordExtractor(),
    )

    ctx = {"db_manager": FakeDbManager(db_session), "job_try": 1}

    result = await create_long_term_memory_task(
        ctx,
        ai_entity_id=ai_entity.id,
        conversation_id=conversation.id,
    )

    assert result == {"memory_count": 0, "stm_chunks_processed": 0, "stm_chunks_deleted": 0}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_long_term_memory_task_retries_on_embedding_error(
    db_session,
    user_factory,
    conversation_factory,
    message_factory,
    ai_entity_factory,
    monkeypatch,
):
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload
    from app.models.message import Message
    from app.services.memory.short_term_memory_service import ShortTermMemoryService

    user = await user_factory.create(db_session)
    conversation = await conversation_factory.create_private_conversation(db_session)
    conv_repo = ConversationRepository(db_session)
    await conv_repo.add_participant(conversation_id=conversation.id, user_id=user.id)
    ai_entity = await ai_entity_factory.create_online(db_session)

    # Create message
    msg = await message_factory.create_conversation_message(
        db_session, sender=user, conversation=conversation, content="Message to embed."
    )

    # Reload message with eager loading
    query = (
        select(Message)
        .where(Message.id == msg.id)
        .options(selectinload(Message.sender_user), selectinload(Message.sender_ai))
    )
    result_query = await db_session.execute(query)
    message = result_query.scalar_one()

    # Create STM chunk
    memory_repo = AIMemoryRepository(db_session)
    stm_service = ShortTermMemoryService(
        memory_repo=memory_repo,
        keyword_extractor=FakeKeywordExtractor(),
    )
    await stm_service.create_short_term_chunk(
        entity_id=ai_entity.id,
        user_ids=[user.id],
        conversation_id=conversation.id,
        chunk_messages=[message],
        chunk_index=0,
        start_idx=0,
        end_idx=0,
    )

    _monkeypatch_ai_stack(
        monkeypatch,
        embedding_service=FakeLongEmbeddingService(raise_error=True),
        keyword_extractor=FakeKeywordExtractor(),
    )

    ctx = {"db_manager": FakeDbManager(db_session), "job_try": 1}

    with pytest.raises(Retry):
        await create_long_term_memory_task(
            ctx,
            ai_entity_id=ai_entity.id,
            conversation_id=conversation.id,
        )
