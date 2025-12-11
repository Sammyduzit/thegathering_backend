"""Unit tests for AIMemoryRepository utilities."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.models.ai_memory import AIMemory
from app.repositories.ai_memory_repository import AIMemoryRepository


@pytest.mark.unit
class TestAIMemoryRepository:
    async def test_get_ltm_fact_returns_match(self):
        """get_ltm_fact should return the scalar result from the DB query."""
        fake_memory = AIMemory(id=1, entity_id=1, conversation_id=2, memory_metadata={"type": "long_term"})

        # Mock DB execute → returns object with scalar_one_or_none
        execute_result = SimpleNamespace(scalar_one_or_none=lambda: fake_memory)
        db = AsyncMock()
        db.execute.return_value = execute_result

        repo = AIMemoryRepository(db)

        result = await repo.get_ltm_fact(
            entity_id=1,
            conversation_id=2,
            chunk_index=0,
            fact_hash="deadbeef",
        )

        assert result is fake_memory
        db.execute.assert_called_once()

    async def test_get_ltm_fact_returns_none_when_missing(self):
        """If no fact matches, get_ltm_fact should return None."""
        execute_result = SimpleNamespace(scalar_one_or_none=lambda: None)
        db = AsyncMock()
        db.execute.return_value = execute_result

        repo = AIMemoryRepository(db)

        result = await repo.get_ltm_fact(
            entity_id=1,
            conversation_id=2,
            chunk_index=0,
            fact_hash="deadbeef",
        )

        assert result is None
        db.execute.assert_called_once()
