"""Reusable retry decorator for synchronous and asynchronous operations."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)


def retry(
    *,
    max_attempts: int = 3,
    delay_seconds: float = 0.5,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Retry a function up to ``max_attempts`` total times.

    Works with both regular and ``async`` functions. Retries use exponential backoff;
    set ``exceptions`` to limit which errors should be retried.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if delay_seconds < 0:
        raise ValueError("delay_seconds cannot be negative")
    if backoff_factor < 1:
        raise ValueError("backoff_factor must be at least 1")
    if not exceptions or not all(
        isinstance(error_type, type) and issubclass(error_type, Exception)
        for error_type in exceptions
    ):
        raise ValueError("exceptions must contain one or more Exception types")

    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(function):

            @wraps(function)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                for attempt in range(1, max_attempts + 1):
                    try:
                        return await function(*args, **kwargs)
                    except exceptions as error:
                        if attempt == max_attempts:
                            raise
                        delay = delay_seconds * backoff_factor ** (attempt - 1)
                        logger.warning(
                            "%s raised %s on attempt %d/%d; retrying in %.2f seconds",
                            function.__qualname__,
                            type(error).__name__,
                            attempt,
                            max_attempts,
                            delay,
                        )
                        await asyncio.sleep(delay)
                raise RuntimeError("Retry loop ended unexpectedly")

            return async_wrapper

        @wraps(function)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(1, max_attempts + 1):
                try:
                    return function(*args, **kwargs)
                except exceptions as error:
                    if attempt == max_attempts:
                        raise
                    delay = delay_seconds * backoff_factor ** (attempt - 1)
                    logger.warning(
                        "%s raised %s on attempt %d/%d; retrying in %.2f seconds",
                        function.__qualname__,
                        type(error).__name__,
                        attempt,
                        max_attempts,
                        delay,
                    )
                    time.sleep(delay)
            raise RuntimeError("Retry loop ended unexpectedly")

        return sync_wrapper

    return decorate
