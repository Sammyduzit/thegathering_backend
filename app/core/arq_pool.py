"""ARQ Redis Pool Manager for FastAPI."""

import structlog
from arq import create_pool
from arq.connections import ArqRedis, RedisSettings

from app.core.config import settings

logger = structlog.get_logger(__name__)

arq_pool: ArqRedis | None = None


async def create_arq_pool() -> None:
    """Create ARQ Redis pool on FastAPI startup."""
    global arq_pool

    if not settings.is_ai_available:
        logger.warning("ai_features_disabled", reason="AI features not enabled or OpenAI key missing")
        return

    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    arq_pool = await create_pool(redis_settings)
    logger.info("arq_pool_created", redis_url=settings.redis_url)


async def close_arq_pool() -> None:
    """Close ARQ Redis pool on FastAPI shutdown."""
    global arq_pool

    if arq_pool:
        await arq_pool.close()
        logger.info("arq_pool_closed")


def get_arq_pool() -> ArqRedis | None:
    """Dependency injection for ARQ pool."""
    return arq_pool
