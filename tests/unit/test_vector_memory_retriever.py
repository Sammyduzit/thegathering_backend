"""Unit tests for VectorMemoryRetriever."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.ai_memory import AIMemory
from app.services.memory.vector_memory_retriever import VectorMemoryRetriever


class TestableVectorRetriever(VectorMemoryRetriever):
    """Concrete subclass for testing abstract base."""

    async def retrieve_candidates(
        self, entity_id: int, query: str | None = None, keywords: list[str] | None = None, limit: int = 20
    ):
        return []


@pytest.mark.unit
class TestVectorMemoryRetriever:
    """Covers hybrid search wiring, filtering and cross-layer fusion."""

    @pytest.fixture
    def retriever(self, monkeypatch):
        memory_repo = AsyncMock()
        embedding_service = AsyncMock()
        keyword_extractor = SimpleNamespace(
            extract_keywords=AsyncMock(return_value=["kw"]),
        )

        retriever = TestableVectorRetriever(
            memory_repo=memory_repo,
            embedding_service=embedding_service,
            keyword_extractor=keyword_extractor,
        )

        # deterministic settings overrides for tests
        from app.core.config import settings

        monkeypatch.setattr(settings, "vector_search_weight", 0.7)
        monkeypatch.setattr(settings, "keyword_search_weight", 0.3)
        monkeypatch.setattr(settings, "short_term_candidates", 3)
        monkeypatch.setattr(settings, "long_term_candidates", 3)
        monkeypatch.setattr(settings, "personality_candidates", 3)
        monkeypatch.setattr(settings, "total_memory_limit", 5)
        monkeypatch.setattr(settings, "guaranteed_short_term", 1)
        monkeypatch.setattr(settings, "guaranteed_long_term", 1)
        monkeypatch.setattr(settings, "guaranteed_personality", 0)
        monkeypatch.setattr(settings, "short_term_weight", 2.0)
        monkeypatch.setattr(settings, "long_term_weight", 1.0)
        monkeypatch.setattr(settings, "personality_weight", 0.5)

        return {
            "retriever": retriever,
            "memory_repo": memory_repo,
            "embedding_service": embedding_service,
        }

    @staticmethod
    def _memory(mid: int, mem_type: str) -> AIMemory:
        """Helper to create in-memory AIMemory objects."""
        memory = AIMemory(
            id=mid,
            entity_id=1,
            user_ids=[1],
            conversation_id=1 if mem_type != "personality" else None,
            summary=f"m{mid}",
            memory_content={},
            keywords=["kw"],
            importance_score=1.0,
        )
        memory.memory_metadata = {"type": mem_type}
        return memory

    async def test_hybrid_search_single_layer_invokes_keyword_search(self, retriever):
        r = retriever["retriever"]
        memory_repo = retriever["memory_repo"]
        embedding_service = retriever["embedding_service"]

        embedding_service.embed_text.return_value = [0.1]
        vector_results = [self._memory(1, "short_term")]
        memory_repo.vector_search.return_value = vector_results
        memory_repo.search_by_keywords.return_value = [self._memory(2, "short_term")]

        r._extract_keywords = AsyncMock(return_value=["topic"])
        r._filter_keyword_results = MagicMock(return_value=["filtered"])
        r._rrf_fusion = MagicMock(return_value=["final"])

        result = await r._hybrid_search_single_layer(
            entity_id=5,
            query="hello world",
            user_id=7,
            conversation_id=9,
            memory_type="short_term",
            limit=3,
        )

        assert result == ["final"]
        embedding_service.embed_text.assert_awaited_once_with("hello world")
        memory_repo.vector_search.assert_awaited_once()
        memory_repo.search_by_keywords.assert_awaited_once_with(entity_id=5, keywords=["topic"], limit=3)
        r._filter_keyword_results.assert_called_once_with(
            memory_repo.search_by_keywords.return_value,
            user_id=7,
            conversation_id=9,
            exclude_conversation_id=None,
            memory_type="short_term",
        )
        r._rrf_fusion.assert_called_once_with(
            vector_results=vector_results,
            keyword_results=["filtered"],
            vector_weight=pytest.approx(0.7),
            keyword_weight=pytest.approx(0.3),
            limit=3,
        )

    async def test_hybrid_search_skips_keyword_search_without_keywords(self, retriever):
        r = retriever["retriever"]
        memory_repo = retriever["memory_repo"]
        embedding_service = retriever["embedding_service"]

        embedding_service.embed_text.return_value = [0.1]
        memory_repo.vector_search.return_value = [self._memory(1, "short_term")]

        r._extract_keywords = AsyncMock(return_value=[])
        r._rrf_fusion = MagicMock(return_value=["final"])

        result = await r._hybrid_search_single_layer(
            entity_id=1,
            query=" ",
            user_id=None,
            memory_type="short_term",
            limit=2,
        )

        assert result == ["final"]
        memory_repo.search_by_keywords.assert_not_called()

    def test_cross_layer_rrf_fusion_respects_guaranteed_minimums(self, retriever):
        r = retriever["retriever"]
        short_term = [self._memory(1, "short_term"), self._memory(2, "short_term")]
        long_term = [self._memory(3, "long_term"), self._memory(4, "long_term")]
        personality = [self._memory(5, "personality")]

        result = r._cross_layer_rrf_fusion(
            short_term=short_term,
            long_term=long_term,
            personality=personality,
            total_limit=4,
        )

        # First element guaranteed from short_term, first from long_term,
        # rest filled via weighted pool (deterministic ordering by id fallback)
        assert result[0].id == 1
        assert result[1].id == 3
        assert len(result) == 4

    def test_rrf_fusion_combines_scores(self, retriever):
        r = retriever["retriever"]
        m1 = self._memory(1, "short_term")
        m2 = self._memory(2, "short_term")
        fused = r._rrf_fusion(
            vector_results=[m1, m2],
            keyword_results=[m2, m1],
            vector_weight=1.0,
            keyword_weight=1.0,
            limit=2,
        )
        # m1 has rank 0+1, m2 rank 1+0 -> m1 slightly higher (rank sum)
        assert fused[0].id == 1
        assert fused[1].id == 2

    def test_filter_keyword_results_applies_all_filters(self, retriever):
        r = retriever["retriever"]
        mem_ok = self._memory(1, "short_term")
        mem_ok.user_ids = [42]
        mem_ok.conversation_id = 100

        mem_wrong_user = self._memory(2, "short_term")
        mem_wrong_user.user_ids = [1]
        mem_wrong_user.conversation_id = 100

        mem_wrong_conv = self._memory(3, "short_term")
        mem_wrong_conv.user_ids = [42]
        mem_wrong_conv.conversation_id = 200

        filtered = r._filter_keyword_results(
            [mem_ok, mem_wrong_user, mem_wrong_conv],
            user_id=42,
            conversation_id=100,
            exclude_conversation_id=50,
            memory_type="short_term",
        )

        assert filtered == [mem_ok]
