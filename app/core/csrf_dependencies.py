"""CSRF validation dependencies for state-changing operations."""

import structlog
from fastapi import HTTPException, Request, status

logger = structlog.get_logger(__name__)

# Endpoints that are exempt from CSRF validation
CSRF_EXEMPT_PATHS = {
    "/api/v1/auth/login",
    "/api/v1/auth/refresh",
    "/api/v1/auth/register",
}


def validate_csrf(request: Request) -> None:
    """
    Validate CSRF token for state-changing operations using Double-Submit Cookie pattern.

    CSRF protection is required for all POST, PUT, PATCH, DELETE requests except:
    - GET, HEAD, OPTIONS (safe methods)
    - /auth/login, /auth/refresh, /auth/register (special auth endpoints)

    Double-Submit Cookie Pattern:
    1. CSRF token is stored in a readable cookie (tg_csrf)
    2. Frontend reads cookie and sends value in X-CSRF-Token header
    3. Backend compares cookie value with header value
    4. Attack prevention: Attacker cannot read cookie from different origin (Same-Origin Policy)

    :param request: FastAPI Request object
    :raises HTTPException: 403 if CSRF validation fails
    """
    # Skip CSRF for safe methods
    if request.method in ["GET", "HEAD", "OPTIONS"]:
        return

    # Skip CSRF for exempt endpoints
    if request.url.path in CSRF_EXEMPT_PATHS:
        return

    # Get CSRF token from cookie and header
    csrf_from_cookie = request.cookies.get("tg_csrf")
    csrf_from_header = request.headers.get("X-CSRF-Token")

    # Both must be present
    if not csrf_from_cookie or not csrf_from_header:
        logger.warning(
            "csrf_validation_failed",
            reason="missing_token",
            path=request.url.path,
            method=request.method,
            has_cookie=csrf_from_cookie is not None,
            has_header=csrf_from_header is not None,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF token missing. Ensure you are logged in and X-CSRF-Token header is set.",
        )

    # Both must match
    if csrf_from_cookie != csrf_from_header:
        logger.warning(
            "csrf_validation_failed",
            reason="token_mismatch",
            path=request.url.path,
            method=request.method,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF token mismatch. Please refresh your session.",
        )

    # Validation successful (no logging to avoid noise)
