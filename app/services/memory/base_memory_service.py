"""Base class for memory services with shared utilities."""

import structlog

from app.interfaces.keyword_extractor import IKeywordExtractor

logger = structlog.get_logger(__name__)


class BaseMemoryService:
    """
    Base class for memory services providing shared utilities.

    Implements DRY principle for common operations:
    - Keyword extraction with error handling
    - Text truncation for summaries
    - Batch keyword processing
    """

    def __init__(self, keyword_extractor: IKeywordExtractor):
        self.keyword_extractor = keyword_extractor

    async def _extract_keywords(
        self,
        text: str,
        max_keywords: int = 10
    ) -> list[str]:
        """
        Extract keywords from text using keyword extractor.

        :param text: Text to extract keywords from
        :param max_keywords: Maximum number of keywords
        :return: List of extracted keywords (empty if extraction fails)
        """
        if not text or not text.strip():
            logger.debug("keyword_extraction_skipped", reason="empty_text")
            return []

        try:
            keywords = await self.keyword_extractor.extract_keywords(
                text, max_keywords=max_keywords
            )
            logger.debug(
                "keywords_extracted",
                count=len(keywords),
                keywords=keywords[:3] if keywords else []
            )
            return keywords
        except Exception as e:
            logger.warning(
                "keyword_extraction_failed",
                error=str(e),
                error_type=type(e).__name__,
                text_length=len(text)
            )
            return []

    async def _extract_keywords_batch(
        self,
        texts: list[str],
        max_keywords: int = 10
    ) -> list[list[str]]:
        """
        Extract keywords from multiple texts.

        :param texts: List of texts to process
        :param max_keywords: Maximum keywords per text
        :return: List of keyword lists (one per text)
        """
        keywords_batch = []
        for text in texts:
            keywords = await self._extract_keywords(text, max_keywords)
            keywords_batch.append(keywords)
        return keywords_batch

    @staticmethod
    def _truncate_summary(text: str, max_length: int = 200) -> str:
        """
        Truncate text for summary with ellipsis.

        :param text: Text to truncate
        :param max_length: Maximum length before truncation
        :return: Truncated text with '...' if longer than max_length
        """
        if len(text) > max_length:
            return text[:max_length] + "..."
        return text
