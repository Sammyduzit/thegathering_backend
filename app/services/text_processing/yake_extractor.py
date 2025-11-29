"""
YAKE (Yet Another Keyword Extractor) implementation.

Provides lightweight unsupervised keyword extraction using statistical features.
No training, external corpus, or dictionaries required.
"""

import structlog
import yake

from app.core.config import settings
from app.interfaces.keyword_extractor import IKeywordExtractor, KeywordExtractionError
from app.services.text_processing.stopwords_de import is_stopword

logger = structlog.get_logger(__name__)


class YakeKeywordExtractor(IKeywordExtractor):
    """YAKE-based keyword extractor implementation with improved German support."""

    def __init__(
        self,
        language: str | None = None,
        max_ngram_size: int | None = None,
        deduplication_threshold: float | None = None,
        window_size: int | None = None,
        top_n: int | None = None,
        min_length: int | None = None,
    ):
        """
        Initialize YAKE keyword extractor with config defaults.

        :param language: Language code (default: from settings.keyword_language)
        :param max_ngram_size: Maximum n-gram size (default: from settings.keyword_max_ngrams)
        :param deduplication_threshold: Similarity threshold (default: from settings.keyword_dedup_threshold)
        :param window_size: Context window size (default: from settings.keyword_window_size)
        :param top_n: Number of candidates to extract (default: from settings.keyword_top_n)
        :param min_length: Minimum keyword length (default: from settings.keyword_min_length)
        """
        # Use config defaults if not provided
        self.language = language or settings.keyword_language
        self.max_ngram_size = max_ngram_size if max_ngram_size is not None else settings.keyword_max_ngrams
        self.deduplication_threshold = (
            deduplication_threshold if deduplication_threshold is not None else settings.keyword_dedup_threshold
        )
        self.window_size = window_size if window_size is not None else settings.keyword_window_size
        self.top_n = top_n if top_n is not None else settings.keyword_top_n
        self.min_length = min_length if min_length is not None else settings.keyword_min_length

        # Initialize YAKE extractor
        self.extractor = yake.KeywordExtractor(
            lan=self.language,
            n=self.max_ngram_size,
            dedupLim=self.deduplication_threshold,
            dedupFunc="seqm",
            windowsSize=self.window_size,
            top=self.top_n,
        )

    async def extract_keywords(
        self,
        text: str,
        max_keywords: int = 10,
        language: str = "en",
    ) -> list[str]:
        """
        Extract keywords from text using YAKE algorithm.

        :param text: Text to extract keywords from
        :param max_keywords: Maximum number of keywords to extract
        :param language: Language code (ignored, uses instance language)
        :return: List of extracted keywords (lowercase, normalized), e.g., ['python', 'fastapi', 'sqlalchemy']
        :raises KeywordExtractionError: If extraction fails
        """
        try:
            # Handle empty or very short text
            if not text or len(text.strip()) < 3:
                logger.debug("Text too short for keyword extraction")
                return []

            # Extract keywords with YAKE
            # Returns list of tuples: [(keyword, score), ...]
            # Lower score = more relevant
            raw_keywords = self.extractor.extract_keywords(text)

            # Normalize and filter keywords
            keywords = self._normalize_keywords(raw_keywords, max_keywords)

            logger.debug(
                "extracted_keywords",
                keyword_count=len(keywords),
                text_length=len(text),
                keywords=keywords,
            )

            return keywords

        except Exception as e:
            logger.error("keyword_extraction_failed", error=str(e))
            raise KeywordExtractionError(f"Failed to extract keywords: {str(e)}", original_error=e)

    def _normalize_keywords(self, raw_keywords: list[tuple], max_keywords: int) -> list[str]:
        """
        Normalize and filter extracted keywords with improved quality.

        Filters applied:
        1. Length check (configurable minimum, default 2 for "AI", "KI")
        2. No pure numbers
        3. Score threshold (only keep reasonably scored keywords, score <= 0.5)
        4. Stopword filtering (German stopwords if language="de")
           - Single words: filtered if stopword
           - N-grams: filtered if ANY word is a stopword (e.g., "zeigen die grenzen")
        5. Deduplication

        :param raw_keywords: List of (keyword, score) tuples from YAKE (lower score = more relevant)
        :param max_keywords: Maximum number of keywords to return
        :return: Filtered and normalized keyword list
        """
        normalized = []

        # Determine max acceptable score (YAKE: lower is better, typically 0-1 range)
        # Only keep keywords with score <= 0.5 (reasonably good keywords)
        max_score = 0.5

        for keyword, score in raw_keywords:
            # Lowercase and strip whitespace
            kw = keyword.lower().strip()

            # Apply filters:
            # 1. Minimum length (allow "ai", "ki" with min_length=2)
            if len(kw) < self.min_length:
                continue

            # 2. No pure numbers
            if kw.isdigit():
                continue

            # 3. Score threshold (lower is better in YAKE)
            if score > max_score:
                continue

            # 4. Stopword filtering (only for German)
            if self.language == "de":
                # For single-word keywords, check if it's a stopword
                if " " not in kw and is_stopword(kw):
                    continue

                # For n-grams, reject if ANY word is a stopword
                # This filters: "zeigen die grenzen", "führt zu absurden", etc.
                if " " in kw:
                    words = kw.split()
                    if any(is_stopword(word) for word in words):
                        continue

            # 5. No duplicates
            if kw in normalized:
                continue

            normalized.append(kw)

            # Stop when we have enough
            if len(normalized) >= max_keywords:
                break

        return normalized
