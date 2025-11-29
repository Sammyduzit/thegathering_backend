"""Cookie management utilities for authentication."""

import secrets

from fastapi import Response

from app.core.config import settings
from app.core.constants import SECONDS_PER_DAY, SECONDS_PER_MINUTE


def set_auth_cookies(
    response: Response,
    access_token: str,
    refresh_token: str,
    csrf_token: str,
) -> None:
    """
    Set all authentication cookies with proper security attributes.

    Sets three cookies:
    - tg_access: Short-lived access token (HttpOnly)
    - tg_refresh: Long-lived refresh token (HttpOnly)
    - tg_csrf: CSRF token (readable by JavaScript for header injection)

    :param response: FastAPI Response object
    :param access_token: JWT access token
    :param refresh_token: JWT refresh token
    :param csrf_token: CSRF token for double-submit pattern
    """
    cookie_kwargs = {
        "secure": settings.cookie_secure,
        "samesite": settings.cookie_samesite,
        "path": "/",
    }

    # Only set domain if explicitly configured (None = automatic browser origin)
    if settings.cookie_domain:
        cookie_kwargs["domain"] = settings.cookie_domain

    # Access token (HttpOnly, short-lived)
    response.set_cookie(
        key="tg_access",
        value=access_token,
        httponly=True,
        max_age=settings.access_token_expire_minutes * SECONDS_PER_MINUTE,
        **cookie_kwargs,
    )

    # Refresh token (HttpOnly, long-lived)
    response.set_cookie(
        key="tg_refresh",
        value=refresh_token,
        httponly=True,
        max_age=settings.refresh_token_expire_days * SECONDS_PER_DAY,
        **cookie_kwargs,
    )

    # CSRF token (NOT HttpOnly, readable by JavaScript)
    response.set_cookie(
        key="tg_csrf",
        value=csrf_token,
        httponly=False,  # Must be readable by JavaScript for X-CSRF-Token header
        max_age=settings.refresh_token_expire_days * SECONDS_PER_DAY,
        **cookie_kwargs,
    )


def clear_auth_cookies(response: Response) -> None:
    """
    Clear all authentication cookies.

    Used for logout to invalidate all auth-related cookies.

    :param response: FastAPI Response object
    """
    cookie_kwargs = {
        "secure": settings.cookie_secure,
        "samesite": settings.cookie_samesite,
        "path": "/",
    }

    if settings.cookie_domain:
        cookie_kwargs["domain"] = settings.cookie_domain

    response.delete_cookie("tg_access", **cookie_kwargs)
    response.delete_cookie("tg_refresh", **cookie_kwargs)
    response.delete_cookie("tg_csrf", **cookie_kwargs)


def generate_csrf_token() -> str:
    """
    Generate cryptographically secure CSRF token.

    Uses secrets.token_urlsafe for URL-safe random string generation.

    :return: CSRF token string
    """
    return secrets.token_urlsafe(settings.csrf_token_length)
