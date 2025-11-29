"""Factory for creating keyword extractor instances based on configuration."""

from app.interfaces.keyword_extractor import IKeywordExtractor
from app.services.text_processing.yake_extractor import YakeKeywordExtractor


def create_keyword_extractor() -> IKeywordExtractor:
    """
    Create keyword extractor instance based on configuration.

    Currently only YAKE is supported. Future implementations could support:
    - LLM-based extraction (OpenAI, Claude)
    - spaCy/BERT-based extraction
    - Hybrid approaches

    :return: YAKE keyword extractor instance with settings defaults
    """
    # Future: Check settings.USE_LLM_KEYWORDS for LLM implementation
    return YakeKeywordExtractor()
