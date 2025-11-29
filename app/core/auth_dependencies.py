from datetime import datetime, timedelta, timezone

import structlog
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.csrf_dependencies import validate_csrf
from app.core.jwt_utils import get_user_from_token
from app.models.user import User
from app.repositories.repository_dependencies import get_user_repository
from app.repositories.user_repository import IUserRepository

logger = structlog.get_logger(__name__)

security = HTTPBearer(auto_error=False)


def get_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> str:
    """
    Extract JWT token from HttpOnly cookie (primary) or Authorization header (fallback).

    Cookie-based authentication is the recommended approach for web applications.
    Header-based authentication is maintained for backward compatibility with:
    - API documentation tools (Swagger UI)
    - Development/testing tools (Postman, curl)
    - Mobile apps (if not using cookie-based auth)

    :param request: FastAPI Request object for reading cookies
    :param credentials: HTTP authorization credentials (optional fallback)
    :return: JWT token string
    """
    # Primary: HttpOnly Cookie
    token = request.cookies.get("tg_access")
    if token:
        return token

    # Fallback: Authorization Header (Swagger/Dev/Mobile)
    if credentials:
        logger.warning(
            "header_auth_used",
            path=request.url.path,
            user_agent=request.headers.get("user-agent", "unknown"),
            message="Header-based auth is deprecated for web clients. Use cookie-based login.",
        )
        return credentials.credentials

    # Neither cookie nor header present
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Use POST /api/v1/auth/login to obtain cookie.",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_user(
    token: str = Depends(get_token),
    user_repo: IUserRepository = Depends(get_user_repository),
) -> User:
    """
    Get current authenticated user from JWT token.
    :param token: JWT token string
    :param user_repo: User repository instance
    :return: Current user object
    """
    username = get_user_from_token(token)

    user = await user_repo.get_by_username(username)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"User '{username}' not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Get current active user.
    :param current_user: Current user from token
    :return: Active user object
    """
    return current_user


def get_current_admin_user(
    current_user: User = Depends(get_current_active_user),
) -> User:
    """
    Get current user and verify admin status.
    :param current_user: Current authenticated user
    :return: Admin user object
    """

    if not current_user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin privileges required")
    return current_user


# Composite Dependencies (Authentication + CSRF Protection)


def get_authenticated_user_with_csrf(
    current_user: User = Depends(get_current_active_user),
    _csrf: None = Depends(validate_csrf),
) -> User:
    """
    Get authenticated user with CSRF protection.

    Validates CSRF token and returns the current authenticated user.
    For use in mutation endpoints (POST/PUT/PATCH/DELETE).

    :param current_user: Current authenticated user
    :param _csrf: CSRF validation (injected automatically)
    :return: Authenticated user object
    """
    return current_user


def get_admin_user_with_csrf(
    admin_user: User = Depends(get_current_admin_user),
    _csrf: None = Depends(validate_csrf),
) -> User:
    """
    Get admin user with CSRF protection.

    Validates CSRF token and admin privileges, returns the admin user.
    For use in admin-only mutation endpoints.

    :param admin_user: Current admin user
    :param _csrf: CSRF validation (injected automatically)
    :return: Admin user object
    """
    return admin_user


async def get_user_with_message_quota(
    current_user: User = Depends(get_authenticated_user_with_csrf),
    user_repo: IUserRepository = Depends(get_user_repository),
) -> User:
    """
    Get authenticated user with CSRF protection and verify message quota.

    Automatically resets weekly counter if 7 days have passed.
    Raises 429 if user has exceeded their weekly message limit.

    :param current_user: Current authenticated user (with CSRF validation)
    :param user_repo: User repository instance
    :return: User object (if quota not exceeded)
    :raises HTTPException: 429 if quota exceeded
    """
    # Admin users with limit=-1 have unlimited messages
    if current_user.weekly_message_limit == -1:
        return current_user

    # Check if 7 days have passed since last reset
    now = datetime.now(timezone.utc)
    days_since_reset = (now - current_user.weekly_reset_date).days

    if days_since_reset >= 7:
        # Reset counter and date
        current_user.weekly_message_count = 0
        current_user.weekly_reset_date = now
        await user_repo.update(current_user)
        logger.info(
            "weekly_quota_reset",
            user_id=current_user.id,
            username=current_user.username,
            reset_date=now.isoformat(),
        )

    # Check if quota exceeded
    if current_user.weekly_message_count >= current_user.weekly_message_limit:
        logger.warning(
            "quota_exceeded",
            user_id=current_user.id,
            username=current_user.username,
            email=current_user.email,
            limit=current_user.weekly_message_limit,
            used=current_user.weekly_message_count,
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "Weekly message limit reached",
                "message": "You have reached your weekly message limit. Please contact the administrator if you need to continue using the application.",
                "limit": current_user.weekly_message_limit,
                "used": current_user.weekly_message_count,
                "last_reset_date": current_user.weekly_reset_date.isoformat(),
                "next_reset_date": (current_user.weekly_reset_date + timedelta(days=7)).isoformat(),
            },
        )

    return current_user
