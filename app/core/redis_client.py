"""Redis client for authentication and session management."""

import structlog
from redis.asyncio import Redis

from app.core.config import settings

logger = structlog.get_logger(__name__)

redis_client: Redis | None = None


async def create_redis_client() -> None:
    """
    Create Redis client on FastAPI startup.

    Separate from ARQ pool to avoid coupling auth logic with worker queue.
    """
    global redis_client

    logger.info("redis_client_initializing", redis_url=settings.redis_url)

    try:
        redis_client = Redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        # Test connection
        await redis_client.ping()
        logger.info("redis_client_created", redis_url=settings.redis_url)
    except Exception as e:
        logger.error("redis_client_creation_failed", error=str(e))
        redis_client = None


async def close_redis_client() -> None:
    """Close Redis client on FastAPI shutdown."""
    global redis_client

    if redis_client:
        await redis_client.aclose()
        logger.info("redis_client_closed")


def get_redis() -> Redis:
    """
    Dependency injection for Redis client.

    Used for auth token storage, CSRF tracking, and session management.

    :raises RuntimeError: If Redis client is not initialized
    :return: Redis client instance
    """
    if redis_client is None:
        raise RuntimeError("Redis client not initialized. Ensure create_redis_client() was called on startup.")

    return redis_client
