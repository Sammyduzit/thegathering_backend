"""
Service interfaces for dependency injection.

This module defines abstract interfaces for all external dependencies
and services, enabling clean dependency injection and easy testing.
"""

from app.interfaces.ai_provider import AIProviderError, IAIProvider
from app.interfaces.keyword_extractor import IKeywordExtractor, KeywordExtractionError
from app.interfaces.memory_retriever import IMemoryRetriever, MemoryRetrievalError
from app.interfaces.memory_summarizer import IMemorySummarizer, MemorySummarizationError
from app.interfaces.translator import TranslationError, TranslatorInterface

__all__ = [
    "IAIProvider",
    "AIProviderError",
    "TranslatorInterface",
    "TranslationError",
    "IKeywordExtractor",
    "KeywordExtractionError",
    "IMemorySummarizer",
    "MemorySummarizationError",
    "IMemoryRetriever",
    "MemoryRetrievalError",
]
