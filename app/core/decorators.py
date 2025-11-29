"""Reusable decorators for cross-cutting concerns."""

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


def standard_retry(func):
    """
    Standard retry decorator for API calls with exponential backoff.

    Configuration:
    - Max attempts: 3
    - Initial wait: 2 seconds
    - Max wait: 10 seconds
    - Exponential multiplier: 1
    - Retries on: Any Exception
    - Re-raises: Yes (after max attempts)

    Usage:
        @standard_retry
        async def my_api_call():
            ...
    """
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )(func)
