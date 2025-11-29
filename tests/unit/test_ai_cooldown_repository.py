"""Tests for AI Cooldown Repository."""

from datetime import datetime, timezone

import pytest

from app.models.ai_cooldown import AICooldown
from app.repositories.ai_cooldown_repository import AICooldownRepository


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_cooldown_returns_none_if_not_exists(async_db_session, created_ai_entity, created_room):
    """Test get_cooldown returns None if no cooldown exists."""
    repo = AICooldownRepository(async_db_session)

    cooldown = await repo.get_cooldown(
        ai_entity_id=created_ai_entity.id,
        room_id=created_room.id,
    )

    assert cooldown is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_upsert_cooldown_creates_new_record(async_db_session, created_ai_entity, created_room):
    """Test upsert_cooldown creates a new cooldown record."""
    repo = AICooldownRepository(async_db_session)

    cooldown = await repo.upsert_cooldown(
        ai_entity_id=created_ai_entity.id,
        room_id=created_room.id,
    )

    assert cooldown is not None
    assert cooldown.ai_entity_id == created_ai_entity.id
    assert cooldown.room_id == created_room.id
    assert cooldown.conversation_id is None
    assert isinstance(cooldown.last_response_at, datetime)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_upsert_cooldown_updates_existing_record(async_db_session, created_ai_entity, created_conversation):
    """
    Test upsert_cooldown updates existing cooldown timestamp.

    SQLite allows multiple NULLs in UNIQUE constraints, so UPSERT semantics can't be
    validated there; skip and rely on integration tests (PostgreSQL) for that case.
    """
    if async_db_session.bind.dialect.name == "sqlite":
        pytest.skip("SQLite UNIQUE with NULLs does not enforce conflict; validated in integration (PostgreSQL).")
    repo = AICooldownRepository(async_db_session)

    # First upsert
    first_cooldown = await repo.upsert_cooldown(
        ai_entity_id=created_ai_entity.id,
        conversation_id=created_conversation.id,
    )
    first_timestamp = first_cooldown.last_response_at

    # Second upsert (should update timestamp)
    second_cooldown = await repo.upsert_cooldown(
        ai_entity_id=created_ai_entity.id,
        conversation_id=created_conversation.id,
    )

    assert second_cooldown.id == first_cooldown.id
    assert second_cooldown.last_response_at > first_timestamp


@pytest.mark.unit
@pytest.mark.asyncio
async def test_upsert_cooldown_conversation_context(async_db_session, created_ai_entity, created_conversation):
    """Test upsert_cooldown works with conversation context."""
    repo = AICooldownRepository(async_db_session)

    cooldown = await repo.upsert_cooldown(
        ai_entity_id=created_ai_entity.id,
        conversation_id=created_conversation.id,
    )

    assert cooldown.ai_entity_id == created_ai_entity.id
    assert cooldown.room_id is None
    assert cooldown.conversation_id == created_conversation.id


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_cooldown_retrieves_correct_context(async_db_session, created_ai_entity, created_room):
    """Test get_cooldown retrieves the correct cooldown by context."""
    repo = AICooldownRepository(async_db_session)

    # Create cooldown
    await repo.upsert_cooldown(
        ai_entity_id=created_ai_entity.id,
        room_id=created_room.id,
    )

    # Retrieve cooldown
    cooldown = await repo.get_cooldown(
        ai_entity_id=created_ai_entity.id,
        room_id=created_room.id,
    )

    assert cooldown is not None
    assert cooldown.ai_entity_id == created_ai_entity.id
    assert cooldown.room_id == created_room.id
