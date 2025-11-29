from app.core.config import settings
from app.interfaces.embedding_service import IEmbeddingService
from app.interfaces.keyword_extractor import IKeywordExtractor
from app.interfaces.memory_retriever import IMemoryRetriever
from app.models.ai_memory import AIMemory
from app.repositories.ai_memory_repository import AIMemoryRepository


class VectorMemoryRetriever(IMemoryRetriever):
    """
    Hybrid memory retriever using vector search + keyword search with RRF fusion.

    Two-level RRF strategy:
    1. Within-layer: Combine vector + keyword results per layer
    2. Cross-layer: Weighted RRF across all layers with guaranteed minimums
    """

    def __init__(
        self,
        memory_repo: AIMemoryRepository,
        embedding_service: IEmbeddingService,
        keyword_extractor: IKeywordExtractor,
    ):
        self.memory_repo = memory_repo
        self.embedding_service = embedding_service
        self.keyword_extractor = keyword_extractor

    async def retrieve_candidates(
        self,
        entity_id: int,
        query: str | None = None,
        keywords: list[str] | None = None,
        limit: int = 20,
    ) -> list[AIMemory]:
        """
        Retrieve memory candidates for further filtering/ranking.

        Uses hybrid search (vector + keyword) to fetch candidates.
        """
        if not query:
            query = " ".join(keywords) if keywords else ""

        if not query:
            return []

        # Use hybrid search without layer filtering
        return await self._hybrid_search_single_layer(
            entity_id=entity_id,
            query=query,
            limit=limit,
        )

    async def retrieve_tiered(
        self,
        entity_id: int,
        user_id: int,
        conversation_id: int | None,
        query: str,
    ) -> list[AIMemory]:
        """
        Tiered retrieval with cross-layer RRF fusion.

        Returns mixed memories from all 3 layers, ordered by relevance + recency.
        """
        # Step 1: Retrieve candidates from all layers
        short_term = await self._hybrid_search_single_layer(
            entity_id=entity_id,
            query=query,
            user_id=user_id,
            conversation_id=conversation_id,
            memory_type="short_term",
            limit=settings.short_term_candidates,
        )

        long_term = await self._hybrid_search_single_layer(
            entity_id=entity_id,
            query=query,
            user_id=user_id,
            conversation_id=None,
            exclude_conversation_id=conversation_id,
            memory_type="long_term",
            limit=settings.long_term_candidates,
        )

        personality = await self._hybrid_search_single_layer(
            entity_id=entity_id,
            query=query,
            user_id=None,  # Global
            conversation_id=None,
            memory_type="personality",
            limit=settings.personality_candidates,
        )

        # Step 2: Cross-layer RRF fusion with guaranteed minimums
        final_memories = self._cross_layer_rrf_fusion(
            short_term=short_term,
            long_term=long_term,
            personality=personality,
            total_limit=settings.total_memory_limit,
        )

        return final_memories

    async def _hybrid_search_single_layer(
        self,
        entity_id: int,
        query: str,
        user_id: int | None = None,
        conversation_id: int | None = None,
        exclude_conversation_id: int | None = None,
        memory_type: str | None = None,
        limit: int = 20,
    ) -> list[AIMemory]:
        """
        Hybrid search (vector + keyword) within a single layer.

        Returns RRF-fused results from vector and keyword search.
        """
        # Extract keywords from query
        keywords = await self._extract_keywords(query)

        # Vector search
        embedding = await self.embedding_service.embed_text(query)
        vector_results = await self.memory_repo.vector_search(
            entity_id=entity_id,
            embedding=embedding,
            user_id=user_id,
            conversation_id=conversation_id,
            exclude_conversation_id=exclude_conversation_id,
            memory_type=memory_type,
            limit=limit,
        )

        # Keyword search (only if keywords exist)
        keyword_results = []
        if keywords:
            keyword_results = await self.memory_repo.search_by_keywords(
                entity_id=entity_id,
                keywords=keywords,
                limit=limit,
            )
            # Apply same filters as vector search
            keyword_results = self._filter_keyword_results(
                keyword_results,
                user_id=user_id,
                conversation_id=conversation_id,
                exclude_conversation_id=exclude_conversation_id,
                memory_type=memory_type,
            )

        # RRF fusion within layer
        fused = self._rrf_fusion(
            vector_results=vector_results,
            keyword_results=keyword_results,
            vector_weight=settings.vector_search_weight,
            keyword_weight=settings.keyword_search_weight,
            limit=limit,
        )

        return fused

    def _cross_layer_rrf_fusion(
        self,
        short_term: list[AIMemory],
        long_term: list[AIMemory],
        personality: list[AIMemory],
        total_limit: int,
    ) -> list[AIMemory]:
        """
        Cross-layer weighted RRF with guaranteed minimums.

        1. Extract guaranteed minimums from each layer
        2. Pool remaining candidates
        3. Apply weighted RRF to pool
        4. Combine guaranteed + pool
        """
        guaranteed = []
        pool = []

        # Extract guaranteed short-term
        if len(short_term) > 0 and settings.guaranteed_short_term > 0:
            guaranteed.extend(short_term[: settings.guaranteed_short_term])
            pool.extend(short_term[settings.guaranteed_short_term :])
        else:
            pool.extend(short_term)

        # Extract guaranteed long-term
        if len(long_term) > 0 and settings.guaranteed_long_term > 0:
            guaranteed.extend(long_term[: settings.guaranteed_long_term])
            pool.extend(long_term[settings.guaranteed_long_term :])
        else:
            pool.extend(long_term)

        # Extract guaranteed personality
        if len(personality) > 0 and settings.guaranteed_personality > 0:
            guaranteed.extend(personality[: settings.guaranteed_personality])
            pool.extend(personality[settings.guaranteed_personality :])
        else:
            pool.extend(personality)

        # Calculate remaining budget
        remaining_budget = total_limit - len(guaranteed)
        if remaining_budget <= 0:
            return guaranteed[:total_limit]

        # Weighted RRF over pool
        memory_scores: dict[int, float] = {}
        k = 60  # RRF constant

        # Assign weights per layer
        layer_weights = {
            "short_term": settings.short_term_weight,
            "long_term": settings.long_term_weight,
            "personality": settings.personality_weight,
        }

        # Score each memory in pool
        for memory in pool:
            if memory.id not in memory_scores:
                memory_scores[memory.id] = 0.0

            # Determine layer
            layer_type = memory.memory_metadata.get("type") if memory.memory_metadata else None
            weight = layer_weights.get(layer_type, 1.0)

            # Find rank in original layer
            rank = self._find_rank_in_layer(memory, short_term, long_term, personality)

            # Add weighted RRF score
            memory_scores[memory.id] += weight / (k + rank)

        # Sort pool by score
        pool_sorted = sorted(pool, key=lambda m: memory_scores.get(m.id, 0.0), reverse=True)

        # Combine guaranteed + top of pool
        final = guaranteed + pool_sorted[:remaining_budget]

        return final[:total_limit]

    def _find_rank_in_layer(
        self,
        memory: AIMemory,
        short_term: list[AIMemory],
        long_term: list[AIMemory],
        personality: list[AIMemory],
    ) -> int:
        """Find rank (position) of memory in its original layer."""
        layer_type = memory.memory_metadata.get("type") if memory.memory_metadata else None

        if layer_type == "short_term":
            try:
                return short_term.index(memory)
            except ValueError:
                return 999
        elif layer_type == "long_term":
            try:
                return long_term.index(memory)
            except ValueError:
                return 999
        elif layer_type == "personality":
            try:
                return personality.index(memory)
            except ValueError:
                return 999

        return 999

    def _rrf_fusion(
        self,
        vector_results: list[AIMemory],
        keyword_results: list[AIMemory],
        vector_weight: float,
        keyword_weight: float,
        limit: int,
    ) -> list[AIMemory]:
        """
        Reciprocal Rank Fusion (RRF) for combining vector + keyword results.

        Formula: Score = Σ(weight / (k + rank))
        """
        memory_scores: dict[int, float] = {}
        k = 60  # RRF constant

        # Score vector results
        for rank, memory in enumerate(vector_results):
            if memory.id not in memory_scores:
                memory_scores[memory.id] = 0.0
            memory_scores[memory.id] += vector_weight / (k + rank)

        # Score keyword results
        for rank, memory in enumerate(keyword_results):
            if memory.id not in memory_scores:
                memory_scores[memory.id] = 0.0
            memory_scores[memory.id] += keyword_weight / (k + rank)

        # Combine and deduplicate
        all_memories = {m.id: m for m in vector_results + keyword_results}

        # Sort by RRF score
        sorted_memories = sorted(
            all_memories.values(),
            key=lambda m: memory_scores.get(m.id, 0.0),
            reverse=True,
        )

        return sorted_memories[:limit]

    async def _extract_keywords(self, text: str) -> list[str]:
        """Extract keywords from text using keyword extractor."""
        if not text or not text.strip():
            return []

        try:
            return await self.keyword_extractor.extract_keywords(text, max_keywords=10)
        except Exception:
            return []

    def _filter_keyword_results(
        self,
        results: list[AIMemory],
        user_id: int | None = None,
        conversation_id: int | None = None,
        exclude_conversation_id: int | None = None,
        memory_type: str | None = None,
    ) -> list[AIMemory]:
        """Apply filters to keyword search results (matches vector search filters)."""
        filtered = results

        if user_id is not None:
            filtered = [m for m in filtered if user_id in m.user_ids]

        if conversation_id is not None:
            filtered = [m for m in filtered if m.conversation_id == conversation_id]

        if exclude_conversation_id is not None:
            filtered = [m for m in filtered if m.conversation_id != exclude_conversation_id]

        if memory_type is not None:
            filtered = [m for m in filtered if m.memory_metadata and m.memory_metadata.get("type") == memory_type]

        return filtered
