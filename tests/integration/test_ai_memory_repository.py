"""
Integration tests for AIMemoryRepository.

Tests focus on CRUD operations and memory retrieval using PostgreSQL with pgvector.
"""

import pytest

from app.models.ai_entity import AIEntity
from app.models.ai_memory import AIMemory
from app.repositories.ai_entity_repository import AIEntityRepository
from app.repositories.ai_memory_repository import AIMemoryRepository


@pytest.mark.integration
class TestAIMemoryRepository:
    """Integration tests for AIMemoryRepository CRUD operations with PostgreSQL."""

    async def test_create_memory_success(self, db_session):
        """Test successful AI memory creation."""
        entity_repo = AIEntityRepository(db_session)
        memory_repo = AIMemoryRepository(db_session)

        entity = AIEntity(username="assistant", system_prompt="Test", model_name="gpt-4")
        created_entity = await entity_repo.create(entity)

        memory = AIMemory(
            entity_id=created_entity.id,
            summary="Discussed Python async patterns with user alice",
            memory_content={
                "participants": ["alice"],
                "room_name": "General",
                "key_facts": ["User prefers async/await"],
                "topics": ["python", "async"],
            },
            keywords=["python", "async"],
        )

        created_memory = await memory_repo.create(memory)

        assert created_memory.id is not None
        assert created_memory.entity_id == created_entity.id
        assert created_memory.summary == "Discussed Python async patterns with user alice"
        assert "alice" in created_memory.memory_content["participants"]

    async def test_get_by_id_success(self, db_session):
        """Test successful memory retrieval by ID."""
        entity_repo = AIEntityRepository(db_session)
        memory_repo = AIMemoryRepository(db_session)

        entity = await entity_repo.create(AIEntity(username="bot", system_prompt="Test", model_name="gpt-4"))
        memory = AIMemory(
            entity_id=entity.id,
            summary="Test memory",
            memory_content={"test": "data"},
            keywords=["test"],
        )
        created = await memory_repo.create(memory)

        found_memory = await memory_repo.get_by_id(created.id)

        assert found_memory is not None
        assert found_memory.id == created.id

    async def test_get_entity_memories(self, db_session):
        """Test retrieval of memories for specific entity."""
        entity_repo = AIEntityRepository(db_session)
        memory_repo = AIMemoryRepository(db_session)

        entity1 = await entity_repo.create(AIEntity(username="bot1", system_prompt="Test", model_name="gpt-4"))
        entity2 = await entity_repo.create(AIEntity(username="bot2", system_prompt="Test", model_name="gpt-4"))

        memory1 = AIMemory(
            entity_id=entity1.id, summary="Entity 1 memory", memory_content={"data": "1"}, keywords=["test"]
        )
        memory2 = AIMemory(
            entity_id=entity2.id, summary="Entity 2 memory", memory_content={"data": "2"}, keywords=["test"]
        )

        await memory_repo.create(memory1)
        await memory_repo.create(memory2)

        entity1_memories = await memory_repo.get_entity_memories(entity1.id)

        assert len(entity1_memories) == 1
        assert entity1_memories[0].summary == "Entity 1 memory"

    async def test_get_entity_memories_ordered_by_importance(self, db_session):
        """Test memories are ordered by importance score."""
        entity_repo = AIEntityRepository(db_session)
        memory_repo = AIMemoryRepository(db_session)

        entity = await entity_repo.create(AIEntity(username="bot", system_prompt="Test", model_name="gpt-4"))

        low_importance = AIMemory(
            entity_id=entity.id,
            summary="Low importance",
            memory_content={"data": "low"},
            keywords=["test"],
            importance_score=0.3,
        )
        high_importance = AIMemory(
            entity_id=entity.id,
            summary="High importance",
            memory_content={"data": "high"},
            keywords=["test"],
            importance_score=0.9,
        )

        await memory_repo.create(low_importance)
        await memory_repo.create(high_importance)

        memories = await memory_repo.get_entity_memories(entity.id)

        assert len(memories) == 2
        assert memories[0].summary == "High importance"
        assert memories[1].summary == "Low importance"

    async def test_search_by_keywords(self, db_session):
        """Test keyword-based memory search."""
        entity_repo = AIEntityRepository(db_session)
        memory_repo = AIMemoryRepository(db_session)

        entity = await entity_repo.create(AIEntity(username="bot", system_prompt="Test", model_name="gpt-4"))

        python_memory = AIMemory(
            entity_id=entity.id,
            summary="Python discussion",
            memory_content={"topic": "python"},
            keywords=["python", "async"],
        )
        javascript_memory = AIMemory(
            entity_id=entity.id,
            summary="JavaScript discussion",
            memory_content={"topic": "javascript"},
            keywords=["javascript", "promises"],
        )

        await memory_repo.create(python_memory)
        await memory_repo.create(javascript_memory)

        python_results = await memory_repo.search_by_keywords(entity.id, ["python"])

        assert len(python_results) == 1
        assert python_results[0].summary == "Python discussion"

    async def test_update_memory(self, db_session):
        """Test memory update."""
        entity_repo = AIEntityRepository(db_session)
        memory_repo = AIMemoryRepository(db_session)

        entity = await entity_repo.create(AIEntity(username="bot", system_prompt="Test", model_name="gpt-4"))
        memory = AIMemory(
            entity_id=entity.id,
            summary="Original summary",
            memory_content={"data": "original"},
            keywords=["original"],
        )
        created = await memory_repo.create(memory)

        created.summary = "Updated summary"
        updated = await memory_repo.update(created)

        assert updated.summary == "Updated summary"

    async def test_delete_memory(self, db_session):
        """Test memory deletion."""
        entity_repo = AIEntityRepository(db_session)
        memory_repo = AIMemoryRepository(db_session)

        entity = await entity_repo.create(AIEntity(username="bot", system_prompt="Test", model_name="gpt-4"))
        memory = AIMemory(
            entity_id=entity.id, summary="Delete me", memory_content={"data": "delete"}, keywords=["delete"]
        )
        created = await memory_repo.create(memory)

        deleted = await memory_repo.delete(created.id)
        found_memory = await memory_repo.get_by_id(created.id)

        assert deleted is True
        assert found_memory is None

    async def test_exists_check(self, db_session):
        """Test memory existence check."""
        entity_repo = AIEntityRepository(db_session)
        memory_repo = AIMemoryRepository(db_session)

        entity = await entity_repo.create(AIEntity(username="bot", system_prompt="Test", model_name="gpt-4"))
        memory = AIMemory(entity_id=entity.id, summary="Exists", memory_content={"data": "exists"}, keywords=["exists"])
        created = await memory_repo.create(memory)

        exists = await memory_repo.exists(created.id)
        not_exists = await memory_repo.exists(99999)

        assert exists is True
        assert not_exists is False
