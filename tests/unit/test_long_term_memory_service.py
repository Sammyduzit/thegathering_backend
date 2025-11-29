"""Unit tests for LongTermMemoryService."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.interfaces.embedding_service import EmbeddingServiceError
from app.services.memory.long_term_memory_service import LongTermMemoryService


@pytest.mark.unit
class TestLongTermMemoryService:
    """Covers happy-path and edge cases for long-term memory creation."""

    @pytest.fixture
    def deps(self):
        """Create mocked dependencies for the service."""
        memory_repo = AsyncMock()
        message_repo = AsyncMock()
        embedding_service = AsyncMock()
        chunking_service = MagicMock()
        keyword_extractor = AsyncMock()
        keyword_extractor.extract_keywords = AsyncMock(return_value=[])

        service = LongTermMemoryService(
            memory_repo=memory_repo,
            message_repo=message_repo,
            embedding_service=embedding_service,
            chunking_service=chunking_service,
            keyword_extractor=keyword_extractor,
        )

        return {
            "service": service,
            "memory_repo": memory_repo,
            "message_repo": message_repo,
            "embedding_service": embedding_service,
            "chunking_service": chunking_service,
            "keyword_extractor": keyword_extractor,
        }

    async def test_returns_empty_when_no_messages(self, deps):
        """If the conversation has no messages we should skip all downstream work."""
        deps["message_repo"].get_conversation_messages.return_value = ([], 0)

        result = await deps["service"].create_long_term_archive(
            entity_id=1,
            user_ids=[1, 2],
            conversation_id=123,
        )

        assert result == []
        deps["chunking_service"].chunk_text.assert_not_called()
        deps["embedding_service"].embed_batch.assert_not_called()
        deps["memory_repo"].create.assert_not_called()

    async def test_returns_empty_when_chunker_returns_nothing(self, deps):
        """If chunking yields nothing we return early without embeddings."""
        messages = [
            SimpleNamespace(
                sender_user_id=1,
                sender_username="user",
                content="Hello world",
            )
        ]
        deps["message_repo"].get_conversation_messages.return_value = (messages, 1)
        deps["chunking_service"].chunk_text.return_value = []

        result = await deps["service"].create_long_term_archive(
            entity_id=7,
            user_ids=[1],
            conversation_id=55,
        )

        assert result == []
        deps["chunking_service"].chunk_text.assert_called_once()
        deps["embedding_service"].embed_batch.assert_not_called()
        deps["memory_repo"].create.assert_not_called()

    async def test_happy_path_creates_memory_per_chunk(self, deps):
        """Ensure keywords, embeddings and persisted memories are wired correctly."""
        messages = [
            SimpleNamespace(
                sender_user_id=1,
                sender_username="human",
                content="User says hi",
            ),
            SimpleNamespace(
                sender_user_id=None,
                sender_username="AI assistant",
                content="AI replies hello",
            ),
        ]
        deps["message_repo"].get_conversation_messages.return_value = (messages, 2)

        combined_text = "human: User says hi\n\nAI assistant: AI replies hello"
        deps["chunking_service"].chunk_text.return_value = ["chunk-one", "chunk-two"]

        deps["keyword_extractor"].extract_keywords = AsyncMock(side_effect=[["kw1"], ["kw2"]])
        deps["embedding_service"].embed_batch.return_value = [[0.1], [0.2]]

        async def _create(memory):
            return memory

        deps["memory_repo"].create.side_effect = _create

        result = await deps["service"].create_long_term_archive(
            entity_id=9,
            user_ids=[1, 2],
            conversation_id=42,
        )

        deps["chunking_service"].chunk_text.assert_called_once_with(combined_text)
        deps["embedding_service"].embed_batch.assert_awaited_once_with(["chunk-one", "chunk-two"])
        assert deps["memory_repo"].create.await_count == 2

        assert len(result) == 2
        assert {memory.memory_metadata["chunk_index"] for memory in result} == {0, 1}
        assert result[0].keywords == ["kw1"]
        assert result[1].embedding == [0.2]

    async def test_raises_embedding_service_error_on_failure(self, deps):
        """Embedding failures should surface as EmbeddingServiceError."""
        messages = [
            SimpleNamespace(
                sender_user_id=1,
                sender_username="user",
                content="Hello",
            )
        ]
        deps["message_repo"].get_conversation_messages.return_value = (messages, 1)
        deps["chunking_service"].chunk_text.return_value = ["chunk"]
        deps["keyword_extractor"].extract_keywords = AsyncMock(return_value=["kw"])
        deps["embedding_service"].embed_batch.side_effect = Exception("embed boom")

        with pytest.raises(EmbeddingServiceError) as exc:
            await deps["service"].create_long_term_archive(
                entity_id=3,
                user_ids=[1],
                conversation_id=99,
            )

        assert "Long-term memory creation failed" in str(exc.value)
        deps["memory_repo"].create.assert_not_called()
