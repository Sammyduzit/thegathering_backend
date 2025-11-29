"""
Keyword Extractor interface for extracting keywords from text.

This interface abstracts keyword extraction functionality, allowing different
extractors (YAKE, LLM-based, etc.) to be used interchangeably through dependency injection.
"""

from abc import ABC, abstractmethod


class IKeywordExtractor(ABC):
    """Abstract interface for keyword extraction services."""

    @abstractmethod
    async def extract_keywords(
        self,
        text: str,
        max_keywords: int = 10,
        language: str = "en",
    ) -> list[str]:
        """
        Extract keywords from text.

        :param text: Text to extract keywords from
        :param max_keywords: Maximum number of keywords to extract
        :param language: Language code for text (e.g., 'en', 'de')
        :return: List of extracted keywords (lowercase, normalized), e.g., ['python', 'fastapi', 'sqlalchemy']
        :raises KeywordExtractionError: If extraction fails
        """
        pass


class KeywordExtractionError(Exception):
    """Exception raised when keyword extraction operations fail."""

    def __init__(self, message: str, original_error: Exception | None = None):
        super().__init__(message)
        self.original_error = original_error
