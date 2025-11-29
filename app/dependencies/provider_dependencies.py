"""
Dependency injection for AI Providers.

Provides factory functions for creating AI provider instances with proper configuration.
"""

import structlog

from app.core.config import settings
from app.interfaces.ai_provider import IAIProvider
from app.providers.openai_provider import OpenAIProvider

logger = structlog.get_logger(__name__)


def get_ai_provider() -> IAIProvider | None:
    """
    Get configured AI provider instance.

    :return: Configured IAIProvider instance or None if not configured

    Note:
        Currently supports OpenAI. Returns None if OPENAI_API_KEY is not set.
        This allows the application to run without AI features if not configured.
    """
    if not settings.openai_api_key:
        logger.warning("OPENAI_API_KEY not configured - AI features will be unavailable")
        return None

    try:
        provider = OpenAIProvider(api_key=settings.openai_api_key)
        logger.info(f"AI Provider initialized: {provider.get_model_name()}")
        return provider
    except Exception as e:
        logger.error(f"Failed to initialize AI provider: {e}")
        return None
