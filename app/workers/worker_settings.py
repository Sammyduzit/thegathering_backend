"""ARQ Worker Settings and Configuration."""

import structlog
from arq.connections import RedisSettings
from arq.cron import cron

from app.core.arq_db_manager import ARQDatabaseManager
from app.core.config import settings
from app.workers.tasks import (
    check_and_generate_ai_response,
    cleanup_old_short_term_memories_task,
    create_long_term_memory_task,
)

logger = structlog.get_logger(__name__)


async def startup(ctx: dict) -> None:
    """Initialize resources on worker startup."""
    db_manager = ARQDatabaseManager()
    await db_manager.connect()
    ctx["db_manager"] = db_manager
    logger.info("arq_worker_started", redis_url=settings.redis_url)


async def shutdown(ctx: dict) -> None:
    """Clean up resources on worker shutdown."""
    db_manager: ARQDatabaseManager = ctx.get("db_manager")
    if db_manager:
        await db_manager.disconnect()
    logger.info("arq_worker_stopped")


class WorkerSettings:
    """ARQ worker configuration."""

    functions = [
        check_and_generate_ai_response,
        create_long_term_memory_task,
    ]

    cron_jobs = [
        cron(cleanup_old_short_term_memories_task, hour=3, minute=0),  # Daily at 3 AM
    ]

    redis_settings = RedisSettings.from_dsn(settings.redis_url)

    on_startup = startup
    on_shutdown = shutdown

    max_jobs = 10
    job_timeout = 300
    keep_result = 3600

    max_tries = 3
    retry_jobs = True
