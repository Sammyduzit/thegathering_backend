import asyncio
from collections.abc import Callable
from functools import wraps
from typing import Any

import structlog
from fastapi import BackgroundTasks

logger = structlog.get_logger(__name__)


class AsyncBackgroundTask:
    """Enhanced background task handler for async operations."""

    def __init__(self):
        self._running_tasks: set[asyncio.Task] = set()

    async def add_async_task(self, background_tasks: BackgroundTasks, func: Callable, *args, **kwargs) -> None:
        """
        Add an async function as background task.
        :param background_tasks: FastAPI BackgroundTasks instance
        :param func: Async function to execute
        :param args: Positional arguments for func
        :param kwargs: Keyword arguments for func
        """
        task = asyncio.create_task(self._execute_with_error_handling(func, *args, **kwargs))
        self._running_tasks.add(task)
        task.add_done_callback(self._running_tasks.discard)

        background_tasks.add_task(self._await_task, task)

    async def _execute_with_error_handling(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute async function with error handling and logging.
        :param func: Async function to execute
        :param args: Positional arguments
        :param kwargs: Keyword arguments
        :return: Function result or None on error
        """
        try:
            logger.info(f"Starting background task: {func.__name__}")
            result = await func(*args, **kwargs)
            logger.info(f"Background task completed: {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"Background task failed {func.__name__}: {e}")
            return None

    async def _await_task(self, task: asyncio.Task) -> None:
        """
        Await task completion for FastAPI background tasks.
        :param task: AsyncIO task to await
        """
        try:
            await task
        except Exception as e:
            logger.error(f"Background task error: {e}")

    @property
    def active_tasks_count(self) -> int:
        """Get count of currently running background tasks."""
        return len(self._running_tasks)


async_bg_task_manager = AsyncBackgroundTask()


def background_task_retry(max_retries: int = 3, delay: float = 1.0):
    """
    Decorator for background tasks with retry mechanism.
    :param max_retries: Maximum number of retry attempts
    :param delay: Delay between retries in seconds
    """

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Background task {func.__name__} failed (attempt {attempt + 1}): {e}. "
                            f"Retrying in {delay}s..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Background task {func.__name__} failed after {max_retries + 1} attempts: {e}")

            raise last_exception

        return wrapper

    return decorator
