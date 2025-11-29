"""
E2E tests for AI models with PostgreSQL.

Tests verify:
- AIEntity CRUD with real database
- AIMemory CRUD and retrieval
- Foreign key constraints
- JSONB storage and retrieval
"""

import pytest
from sqlalchemy import select

from app.models.ai_entity import AIEntity, AIEntityStatus
from app.models.ai_memory import AIMemory
from app.repositories.ai_entity_repository import AIEntityRepository
from app.repositories.ai_memory_repository import AIMemoryRepository


@pytest.mark.e2e
class TestAIEntityE2E:
    """E2E tests for AIEntity model with PostgreSQL."""

    async def test_create_ai_entity_with_postgres(self, db_session):
        """Test AI entity creation with real PostgreSQL database."""
        repo = AIEntityRepository(db_session)

        entity = AIEntity(
            username="assistant",
            description="Helpful AI assistant",
            system_prompt="You are a helpful assistant",
            model_name="gpt-4",
            temperature=0.7,
            max_tokens=1024,
        )

        created = await repo.create(entity)

        assert created.id is not None
        assert created.username == "assistant"
        assert created.model_name == "gpt-4"
        assert created.status == AIEntityStatus.OFFLINE

    async def test_ai_entity_unique_username_constraint(self, db_session):
        """Test unique constraint on AI entity username."""
        repo = AIEntityRepository(db_session)

        entity1 = AIEntity(username="duplicate", system_prompt="Test", model_name="gpt-4")
        await repo.create(entity1)

        entity2 = AIEntity(username="duplicate", system_prompt="Test", model_name="gpt-4")

        with pytest.raises(Exception):
            await repo.create(entity2)

    async def test_ai_entity_with_json_config(self, db_session):
        """Test AI entity with JSONB config storage."""
        repo = AIEntityRepository(db_session)

        entity = AIEntity(
            username="configurable",
            system_prompt="Test",
            model_name="gpt-4",
            config={"streaming": True, "top_p": 0.9, "frequency_penalty": 0.5},
        )

        created = await repo.create(entity)
        retrieved = await repo.get_by_id(created.id)

        assert retrieved.config["streaming"] is True
        assert retrieved.config["top_p"] == 0.9


@pytest.mark.e2e
class TestAIMemoryE2E:
    """E2E tests for AIMemory model with PostgreSQL."""

    async def test_create_ai_memory_with_postgres(self, db_session):
        """Test AI memory creation with real PostgreSQL database."""
        entity_repo = AIEntityRepository(db_session)
        memory_repo = AIMemoryRepository(db_session)

        entity = AIEntity(username="memory_bot", system_prompt="Test", model_name="gpt-4")
        created_entity = await entity_repo.create(entity)

        memory = AIMemory(
            entity_id=created_entity.id,
            summary="Discussed Python async patterns with alice in General room",
            memory_content={
                "participants": ["alice", "bob"],
                "room_name": "General",
                "key_facts": ["User prefers async/await", "Discussed performance optimization"],
                "topics": ["python", "async", "performance"],
            },
            keywords=["python", "async", "performance"],
            importance_score=0.8,
        )

        created_memory = await memory_repo.create(memory)

        assert created_memory.id is not None
        assert created_memory.entity_id == created_entity.id
        assert "alice" in created_memory.memory_content["participants"]
        assert created_memory.importance_score == 0.8

    async def test_ai_memory_foreign_key_cascade(self, db_session):
        """Test CASCADE delete when AI entity is deleted."""
        from sqlalchemy import select, text

        entity_repo = AIEntityRepository(db_session)
        memory_repo = AIMemoryRepository(db_session)

        entity = AIEntity(username="cascade_test", system_prompt="Test", model_name="gpt-4")
        created_entity = await entity_repo.create(entity)

        memory = AIMemory(
            entity_id=created_entity.id,
            summary="Test memory",
            memory_content={"test": "data"},
            keywords=["test"],
        )
        created_memory = await memory_repo.create(memory)

        memory_id = created_memory.id
        entity_id = created_entity.id

        # Expunge objects from session to avoid stale references
        db_session.expunge(created_entity)
        db_session.expunge(created_memory)

        # Hard delete entity via raw SQL to test CASCADE
        await db_session.execute(text(f"DELETE FROM ai_entities WHERE id = {entity_id}"))
        await db_session.commit()

        # Memory should be cascade deleted
        found_memory = await memory_repo.get_by_id(memory_id)
        assert found_memory is None

    async def test_ai_memory_jsonb_query(self, db_session):
        """Test JSONB storage and retrieval in PostgreSQL."""
        entity_repo = AIEntityRepository(db_session)
        memory_repo = AIMemoryRepository(db_session)

        entity = await entity_repo.create(AIEntity(username="json_test", system_prompt="Test", model_name="gpt-4"))

        complex_memory = AIMemory(
            entity_id=entity.id,
            summary="Complex memory test",
            memory_content={
                "participants": ["alice", "bob", "charlie"],
                "room_name": "Dev Room",
                "key_facts": ["Discussed microservices", "Mentioned Docker", "Talked about Kubernetes"],
                "topics": ["architecture", "devops", "containers"],
                "user_preferences": {"user_id": 123, "tone": "technical", "interests": ["backend"]},
                "context": "2025-10-02 in Dev Room",
            },
            keywords=["architecture", "devops", "docker", "kubernetes"],
        )

        created = await memory_repo.create(complex_memory)
        retrieved = await memory_repo.get_by_id(created.id)

        assert len(retrieved.memory_content["participants"]) == 3
        assert "alice" in retrieved.memory_content["participants"]
        assert retrieved.memory_content["room_name"] == "Dev Room"
        assert retrieved.memory_content["user_preferences"]["tone"] == "technical"

    async def test_get_entity_memories_ordered(self, db_session):
        """Test memory retrieval ordered by importance and recency."""
        entity_repo = AIEntityRepository(db_session)
        memory_repo = AIMemoryRepository(db_session)

        entity = await entity_repo.create(AIEntity(username="ordered_test", system_prompt="Test", model_name="gpt-4"))

        # Create memories with different importance scores
        for i, importance in enumerate([0.3, 0.9, 0.5, 0.7]):
            memory = AIMemory(
                entity_id=entity.id,
                summary=f"Memory {i} with importance {importance}",
                memory_content={"index": i},
                keywords=["test"],
                importance_score=importance,
            )
            await memory_repo.create(memory)

        memories = await memory_repo.get_entity_memories(entity.id, limit=10)

        # Should be ordered by importance descending
        assert len(memories) == 4
        assert memories[0].importance_score == 0.9
        assert memories[1].importance_score == 0.7
        assert memories[2].importance_score == 0.5
        assert memories[3].importance_score == 0.3

    async def test_keyword_search_with_gin_index(self, db_session):
        """Test keyword-based search (uses GIN index in PostgreSQL)."""
        entity_repo = AIEntityRepository(db_session)
        memory_repo = AIMemoryRepository(db_session)

        entity = await entity_repo.create(AIEntity(username="search_test", system_prompt="Test", model_name="gpt-4"))

        # Create memories with different keywords
        python_memory = AIMemory(
            entity_id=entity.id,
            summary="Python discussion",
            memory_content={"topic": "python"},
            keywords=["python", "async", "fastapi"],
        )
        js_memory = AIMemory(
            entity_id=entity.id,
            summary="JavaScript discussion",
            memory_content={"topic": "javascript"},
            keywords=["javascript", "react", "node"],
        )

        await memory_repo.create(python_memory)
        await memory_repo.create(js_memory)

        # Search for Python-related memories
        python_results = await memory_repo.search_by_keywords(entity.id, ["python"])
        js_results = await memory_repo.search_by_keywords(entity.id, ["react"])

        assert len(python_results) == 1
        assert python_results[0].summary == "Python discussion"

        assert len(js_results) == 1
        assert js_results[0].summary == "JavaScript discussion"
