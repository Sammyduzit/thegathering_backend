from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.constants import (
    DEFAULT_COOKIE_SAMESITE,
    DEFAULT_COOKIE_SECURE,
    DEFAULT_CSRF_TOKEN_LENGTH,
    DEFAULT_REFRESH_TOKEN_EXPIRE_DAYS,
)


class Settings(BaseSettings):
    """Application settings"""

    database_url: str

    secret_key: str
    algorithm: str
    access_token_expire_minutes: int
    refresh_token_expire_days: int = DEFAULT_REFRESH_TOKEN_EXPIRE_DAYS

    app_name: str
    debug: bool

    deepl_api_key: str
    openai_api_key: str | None = None
    google_api_key: str | None = None

    redis_url: str
    ai_features_enabled: bool = True

    # RAG & Vector Search Configuration
    enable_vector_search: bool = True
    embedding_provider: Literal["google", "openai"] = "google"  # Which provider to use for embeddings
    embedding_model: str = "gemini-embedding-001"  # Default to Google Gemini
    embedding_dimensions: int = 1536
    vector_search_weight: float = 0.7
    keyword_search_weight: float = 0.3

    # Keyword Extraction Configuration (YAKE)
    keyword_language: str = "de"  # Language for keyword extraction (de=German, en=English)
    keyword_max_ngrams: int = 3  # Maximum n-gram size (1=unigrams, 2=bigrams, 3=trigrams)
    keyword_window_size: int = 3  # Context window size for co-occurrence analysis
    keyword_dedup_threshold: float = 0.9  # Deduplication threshold (0.0-1.0)
    keyword_min_length: int = 2  # Minimum keyword length (allows "AI", "KI")
    keyword_top_n: int = 20  # Number of candidate keywords to extract before filtering

    # Memory retrieval limits
    total_memory_limit: int = 7

    # Guaranteed minimums per layer (for cross-layer RRF)
    guaranteed_short_term: int = 1
    guaranteed_long_term: int = 0
    guaranteed_personality: int = 0

    # Layer weights for cross-layer RRF
    short_term_weight: float = 2.0
    long_term_weight: float = 1.0
    personality_weight: float = 1.0

    # Retrieval candidate limits per layer (over-fetch for RRF)
    short_term_candidates: int = 5
    long_term_candidates: int = 5
    personality_candidates: int = 5

    # TTL for short-term memories (in days)
    short_term_ttl_days: int = 7

    # Cookie Security Configuration
    cookie_domain: str | None = None
    cookie_secure: bool = DEFAULT_COOKIE_SECURE
    cookie_samesite: Literal["lax", "strict", "none"] = DEFAULT_COOKIE_SAMESITE

    # CSRF Configuration
    csrf_token_length: int = DEFAULT_CSRF_TOKEN_LENGTH

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore")

    @property
    def is_ai_available(self) -> bool:
        """Check if AI features can be enabled."""
        return self.ai_features_enabled and self.openai_api_key is not None


settings = Settings()
