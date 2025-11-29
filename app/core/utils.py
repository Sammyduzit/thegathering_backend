"""Shared utility functions."""

from typing import Any

import structlog
from arq.connections import ArqRedis

logger = structlog.get_logger(__name__)


async def enqueue_arq_job_safe(
    arq_pool: ArqRedis,
    job_name: str,
    context_info: dict[str, Any],
    *args,
    **kwargs,
) -> bool:
    """
    Safely enqueue ARQ job with consistent error handling.

    :param arq_pool: ARQ Redis pool
    :param job_name: Name of the job to enqueue
    :param context_info: Context information for logging (e.g., message_id, conversation_id)
    :param args: Positional arguments for the job
    :param kwargs: Keyword arguments for the job
    :return: True if job was enqueued successfully, False otherwise
    """
    try:
        job = await arq_pool.enqueue_job(job_name, *args, **kwargs)
        logger.info(
            f"{job_name}_enqueued",
            job_id=job.job_id if job else None,
            **context_info,
        )
        return True
    except Exception as e:
        # Non-critical: log warning, don't fail the main operation
        logger.warning(
            f"{job_name}_enqueue_failed",
            error=str(e),
            **context_info,
        )
        return False


def calculate_pagination(total_count: int, page: int, page_size: int) -> tuple[int, bool]:
    """
    Calculate pagination metadata.

    :param total_count: Total number of items
    :param page: Current page number (1-indexed)
    :param page_size: Items per page
    :return: Tuple of (total_pages, has_more)
    """
    total_pages = (total_count + page_size - 1) // page_size if total_count > 0 else 0
    has_more = page < total_pages
    return total_pages, has_more
