"""
Unit tests for AIEntityRepository.

Tests focus on CRUD operations and query methods using SQLite in-memory database.
"""

import pytest

from app.models.ai_entity import AIEntity, AIEntityStatus
from app.repositories.ai_entity_repository import AIEntityRepository


@pytest.mark.unit
class TestAIEntityRepository:
    """Unit tests for AIEntityRepository CRUD operations."""

    async def test_create_entity_success(self, db_session):
        """Test successful AI entity creation."""
        repo = AIEntityRepository(db_session)
        entity = AIEntity(
            username="assistant",
            system_prompt="You are a helpful assistant",
            model_name="gpt-4",
        )

        created_entity = await repo.create(entity)

        assert created_entity.id is not None
        assert created_entity.username == "assistant"
        assert created_entity.status == AIEntityStatus.OFFLINE

    async def test_get_by_id_success(self, db_session):
        """Test successful entity retrieval by ID."""
        repo = AIEntityRepository(db_session)
        entity = AIEntity(username="helper", system_prompt="Help users", model_name="gpt-4")
        created = await repo.create(entity)

        found_entity = await repo.get_by_id(created.id)

        assert found_entity is not None
        assert found_entity.id == created.id
        assert found_entity.username == "helper"

    async def test_get_by_id_not_found(self, db_session):
        """Test entity retrieval when ID does not exist."""
        repo = AIEntityRepository(db_session)

        found_entity = await repo.get_by_id(99999)

        assert found_entity is None

    async def test_get_by_username_success(self, db_session):
        """Test successful entity retrieval by username."""
        repo = AIEntityRepository(db_session)
        entity = AIEntity(username="coder", system_prompt="Help with code", model_name="gpt-4")
        await repo.create(entity)

        found_entity = await repo.get_by_username("coder")

        assert found_entity is not None
        assert found_entity.username == "coder"

    async def test_get_by_username_not_found(self, db_session):
        """Test entity retrieval when username does not exist."""
        repo = AIEntityRepository(db_session)

        found_entity = await repo.get_by_username("nonexistent")

        assert found_entity is None

    async def test_get_available_entities(self, db_session):
        """Test retrieval of available entities (online and not deleted)."""
        repo = AIEntityRepository(db_session)

        online_entity = AIEntity(
            username="online",
            system_prompt="Online",
            model_name="gpt-4",
            status=AIEntityStatus.ONLINE,
        )
        offline_entity = AIEntity(
            username="offline",
            system_prompt="Offline",
            model_name="gpt-4",
            status=AIEntityStatus.OFFLINE,
        )

        await repo.create(online_entity)
        await repo.create(offline_entity)

        available_entities = await repo.get_available_entities()

        assert len(available_entities) == 1
        assert available_entities[0].username == "online"

    async def test_username_exists(self, db_session):
        """Test username existence check."""
        repo = AIEntityRepository(db_session)
        entity = AIEntity(username="unique", system_prompt="Unique", model_name="gpt-4")
        await repo.create(entity)

        exists = await repo.username_exists("unique")
        not_exists = await repo.username_exists("other")

        assert exists is True
        assert not_exists is False

    async def test_username_exists_with_exclude(self, db_session):
        """Test username existence check excluding specific ID."""
        repo = AIEntityRepository(db_session)
        entity = AIEntity(username="test", system_prompt="Test", model_name="gpt-4")
        created = await repo.create(entity)

        exists = await repo.username_exists("test", exclude_id=created.id)

        assert exists is False

    async def test_update_entity(self, db_session):
        """Test entity update."""
        repo = AIEntityRepository(db_session)
        entity = AIEntity(username="updatable", system_prompt="Original", model_name="gpt-4")
        created = await repo.create(entity)

        created.username = "updated"
        updated = await repo.update(created)

        assert updated.username == "updated"

    async def test_soft_delete_entity(self, db_session):
        """Test soft delete sets entity is_active=False and status=OFFLINE."""
        repo = AIEntityRepository(db_session)
        entity = AIEntity(username="deletable", system_prompt="Delete", model_name="gpt-4")
        created = await repo.create(entity)

        deleted = await repo.delete(created.id)

        # After soft delete, get_by_id returns None (filters is_active=False)
        found_entity = await repo.get_by_id(created.id)

        assert deleted is True
        assert found_entity is None  # Soft deleted entities are not returned

    async def test_delete_nonexistent_entity(self, db_session):
        """Test delete returns False for nonexistent entity."""
        repo = AIEntityRepository(db_session)

        deleted = await repo.delete(99999)

        assert deleted is False

    async def test_exists_check(self, db_session):
        """Test entity existence check."""
        repo = AIEntityRepository(db_session)
        entity = AIEntity(username="exists", system_prompt="Exists", model_name="gpt-4")
        created = await repo.create(entity)

        exists = await repo.exists(created.id)
        not_exists = await repo.exists(99999)

        assert exists is True
        assert not_exists is False

    async def test_get_all_with_pagination(self, db_session):
        """Test get all entities with limit and offset."""
        repo = AIEntityRepository(db_session)

        for i in range(5):
            entity = AIEntity(
                username=f"entity{i}",
                system_prompt="Test",
                model_name="gpt-4",
            )
            await repo.create(entity)

        all_entities = await repo.get_all(limit=2, offset=1)

        assert len(all_entities) == 2
