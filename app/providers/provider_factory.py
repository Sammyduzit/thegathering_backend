"""Factory helpers for creating AI providers from config or entity settings."""

import structlog

from app.core.config import settings
from app.interfaces.ai_provider import IAIProvider
from app.models.ai_entity import AIEntity, AIModelProvider
from app.providers.google_provider import GoogleProvider
from app.providers.openai_provider import OpenAIProvider

logger = structlog.get_logger(__name__)


def create_ai_provider(provider: AIModelProvider, model_name: str | None = None) -> IAIProvider:
    """
    Create a concrete AI provider instance.

    :param provider: Provider enum (`openai` or `google`)
    :param model_name: Optional model name override
    :return: Configured AI provider instance
    :raises RuntimeError: If required API key is not configured
    """
    if provider == AIModelProvider.GOOGLE:
        if not settings.google_api_key:
            raise RuntimeError("AI provider 'google' requires GOOGLE_API_KEY")
        return (
            GoogleProvider(api_key=settings.google_api_key, model_name=model_name)
            if model_name
            else GoogleProvider(api_key=settings.google_api_key)
        )

    # Default + explicit OpenAI branch
    if not settings.openai_api_key:
        raise RuntimeError("AI provider 'openai' requires OPENAI_API_KEY")
    return (
        OpenAIProvider(api_key=settings.openai_api_key, model_name=model_name)
        if model_name
        else OpenAIProvider(api_key=settings.openai_api_key)
    )


def create_provider_for_entity(ai_entity: AIEntity) -> IAIProvider:
    """
    Create provider for an AI entity based on persisted provider + model.

    Falls back to OpenAI for legacy rows where provider might be NULL.
    """
    provider = ai_entity.provider or AIModelProvider.OPENAI
    if ai_entity.provider is None:
        logger.warning(
            "ai_entity_provider_missing_fallback",
            ai_entity_id=ai_entity.id,
            ai_username=ai_entity.username,
            fallback_provider=AIModelProvider.OPENAI.value,
        )

    return create_ai_provider(
        provider=provider,
        model_name=ai_entity.model_name,
    )

