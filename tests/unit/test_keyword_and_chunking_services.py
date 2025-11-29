"""Unit tests for YakeKeywordExtractor and TextChunkingService utilities."""

import pytest

from app.services.text_processing.text_chunking_service import TextChunkingService
from app.services.text_processing.yake_extractor import YakeKeywordExtractor


@pytest.mark.unit
class TestYakeKeywordExtractor:
    """Ensure YAKE wrapper normalizes and filters keywords."""

    @pytest.mark.asyncio
    async def test_extract_keywords_filters_stopwords(self):
        extractor = YakeKeywordExtractor(language="de", max_ngram_size=3, top_n=5)
        text = "Die KI hilft Menschen, komplexe Probleme schneller zu lösen."

        keywords = await extractor.extract_keywords(text, max_keywords=5)

        assert all("die" not in kw for kw in keywords)
        assert any("probleme" in kw for kw in keywords)

    @pytest.mark.asyncio
    async def test_extract_keywords_handles_short_text(self):
        extractor = YakeKeywordExtractor()
        keywords = await extractor.extract_keywords("hi", max_keywords=5)
        assert keywords == []


@pytest.mark.unit
class TestTextChunkingService:
    """Validate chunk splitting mechanics."""

    def test_chunk_text_respects_chunk_size(self):
        service = TextChunkingService(chunk_size=20, chunk_overlap=0)
        text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
        chunks = service.chunk_text(text)
        assert all(len(chunk) <= 20 for chunk in chunks)

    def test_chunk_text_returns_empty_for_blank(self):
        service = TextChunkingService()
        assert service.chunk_text("   ") == []
