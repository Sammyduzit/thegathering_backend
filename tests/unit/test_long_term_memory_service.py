"""Unit tests for LongTermMemoryService (fact-based LTM)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.memory.long_term_memory_service import LongTermMemoryService


def _make_chunk(messages: list[dict], chunk_index: int = 0, message_range: str = "0-0"):
    return SimpleNamespace(
        memory_content={"messages": messages},
        memory_metadata={"chunk_index": chunk_index, "message_range": message_range},
    )


@pytest.mark.unit
class TestLongTermMemoryService:
    @pytest.fixture
    def deps(self):
        memory_repo = AsyncMock()
        embedding_service = AsyncMock()
        keyword_extractor = AsyncMock()
        keyword_extractor.extract_keywords = AsyncMock(return_value=["kw"])
        ai_provider = AsyncMock()
        service = LongTermMemoryService(
            memory_repo=memory_repo,
            embedding_service=embedding_service,
            keyword_extractor=keyword_extractor,
            ai_provider=ai_provider,
        )
        return {
            "service": service,
            "memory_repo": memory_repo,
            "embedding_service": embedding_service,
            "keyword_extractor": keyword_extractor,
            "ai_provider": ai_provider,
        }

    async def test_returns_empty_without_chunks(self, deps):
        result = await deps["service"].create_long_term_from_chunks(
            entity_id=1,
            user_ids=[1],
            conversation_id=123,
            stm_chunks=[],
        )
        assert result == []
        deps["ai_provider"].generate_response.assert_not_called()
        deps["memory_repo"].create.assert_not_called()

    async def test_skips_chunk_without_messages(self, deps):
        chunk = _make_chunk(messages=[], chunk_index=0)
        result = await deps["service"].create_long_term_from_chunks(
            entity_id=1,
            user_ids=[1],
            conversation_id=123,
            stm_chunks=[chunk],
        )
        assert result == []
        deps["ai_provider"].generate_response.assert_not_called()

    async def test_llm_success_creates_ltm_per_fact(self, deps):
        fact = {
            "text": "Alice ist Python-Expertin mit 10 Jahren Erfahrung",
            "importance": 0.9,
            "participants": ["Alice"],
            "theme": "Python Expertise",
        }
        deps["ai_provider"].generate_response.return_value = '{"facts": [%s]}' % (
            '{"text": "%s", "importance": 0.9, "participants": ["Alice"], "theme": "Python Expertise"}'
            % fact["text"]
        )
        deps["memory_repo"].get_ltm_fact.return_value = None
        deps["embedding_service"].embed_text.return_value = [0.1]

        chunk = _make_chunk(
            messages=[{"message_id": 1, "sender_name": "Alice", "content": "Alice ist Python-Expertin mit 10 Jahren Erfahrung."}],
            chunk_index=2,
            message_range="10-33",
        )

        # Mock create to return the memory it receives
        async def _create(memory):
            return memory

        deps["memory_repo"].create.side_effect = _create

        result = await deps["service"].create_long_term_from_chunks(
            entity_id=5,
            user_ids=[1],
            conversation_id=42,
            stm_chunks=[chunk],
        )

        assert len(result) == 1
        deps["ai_provider"].generate_response.assert_awaited_once()
        deps["embedding_service"].embed_text.assert_awaited_once()
        deps["memory_repo"].create.assert_awaited_once()
        created = result[0]
        assert created.summary == "Python Expertise"
        assert created.memory_metadata["fact_hash"]

    async def test_idempotence_skips_existing_fact(self, deps):
        deps["ai_provider"].generate_response.return_value = '{"facts": [{"text": "dup", "importance": 0.6, "participants": [], "theme": "T"}]}'
        deps["memory_repo"].get_ltm_fact.return_value = MagicMock()  # Simulate existing

        chunk = _make_chunk(
            messages=[{"sender_name": "Bob", "content": "Bob ist nett"}],
            chunk_index=1,
        )

        result = await deps["service"].create_long_term_from_chunks(
            entity_id=1,
            user_ids=[1],
            conversation_id=99,
            stm_chunks=[chunk],
        )

        assert result == []
        deps["memory_repo"].create.assert_not_called()

    async def test_heuristic_used_on_llm_failure(self, deps):
        deps["ai_provider"].generate_response.side_effect = Exception("llm down")
        deps["memory_repo"].get_ltm_fact.return_value = None
        deps["embedding_service"].embed_text.return_value = [0.2]

        chunk = _make_chunk(
            messages=[{"sender_name": "Alice", "content": "Alice ist sehr freundlich und hilft gerne anderen Menschen."}],
            chunk_index=0,
        )

        async def _create(memory):
            return memory

        deps["memory_repo"].create.side_effect = _create

        result = await deps["service"].create_long_term_from_chunks(
            entity_id=1,
            user_ids=[1],
            conversation_id=10,
            stm_chunks=[chunk],
        )

        assert len(result) >= 1
        deps["memory_repo"].create.assert_awaited()

    async def test_empty_facts_from_llm_triggers_heuristic(self, deps):
        """If LLM returns parsable JSON but empty facts, fallback heuristic should kick in."""
        deps["ai_provider"].generate_response.return_value = '{"facts": []}'
        deps["memory_repo"].get_ltm_fact.return_value = None
        deps["embedding_service"].embed_text.return_value = [0.3]

        chunk = _make_chunk(
            messages=[{"sender_name": "Alice", "content": "Alice arbeitet als Entwicklerin und lernt Rust."}],
            chunk_index=0,
        )

        async def _create(memory):
            return memory

        deps["memory_repo"].create.side_effect = _create

        result = await deps["service"].create_long_term_from_chunks(
            entity_id=1,
            user_ids=[1],
            conversation_id=11,
            stm_chunks=[chunk],
        )

        assert len(result) >= 1
        deps["ai_provider"].generate_response.assert_awaited_once()
        deps["memory_repo"].create.assert_awaited()

    def test_fact_hash_normalizes_whitespace_and_case(self, deps):
        """_get_fact_hash should be stable for case/whitespace variations."""
        service = deps["service"]
        h1 = service._get_fact_hash("Bob   ist   Experte")
        h2 = service._get_fact_hash(" bob ist experte ")
        assert h1 == h2
