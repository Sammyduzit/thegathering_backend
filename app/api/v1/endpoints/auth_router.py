from datetime import timedelta

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from redis.asyncio import Redis

from app.core.auth_dependencies import (
    get_admin_user_with_csrf,
    get_authenticated_user_with_csrf,
    get_current_active_user,
)
from app.core.auth_utils import hash_password, verify_password
from app.core.config import settings
from app.core.constants import SECONDS_PER_DAY, SECONDS_PER_MINUTE
from app.core.cookie_utils import clear_auth_cookies, generate_csrf_token, set_auth_cookies
from app.core.jwt_utils import create_access_token, create_refresh_token, verify_token
from app.core.redis_client import get_redis
from app.core.validators import validate_language_code
from app.models import User
from app.repositories.repository_dependencies import get_user_repository
from app.repositories.user_repository import IUserRepository
from app.schemas.auth_schemas import (
    Token,
    UserLogin,
    UserQuotaExceededResponse,
    UserQuotaResponse,
    UserRegister,
    UserResponse,
    UserUpdate,
)
from app.schemas.common_schemas import MessageResponse
from app.services.domain.avatar_service import generate_avatar_url

router = APIRouter(prefix="/auth", tags=["authentication"])
logger = structlog.get_logger(__name__)


async def _revoke_token_family(redis: Redis, user_id: int, family_id: str) -> None:
    """
    Revoke entire token family when reuse is detected.

    Security measure to prevent token theft exploitation.
    If a stolen token is reused, all tokens in the family are invalidated.

    :param redis: Redis client
    :param user_id: User ID
    :param family_id: Token family ID
    """
    family_key = f"refresh_token_family:{user_id}:{family_id}"

    # Get all tokens in family
    token_jtis = await redis.smembers(family_key)

    # Delete all tokens in family
    for jti in token_jtis:
        jti_str = jti.decode("utf-8") if isinstance(jti, bytes) else jti
        token_key = f"refresh_token:{user_id}:{jti_str}"
        await redis.delete(token_key)

    # Delete family set
    await redis.delete(family_key)

    logger.warning(
        "token_family_revoked",
        user_id=user_id,
        family_id=family_id,
        revoked_tokens=len(token_jtis) if token_jtis else 0,
        reason="token_reuse_detected",
    )


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserRegister, user_repo: IUserRepository = Depends(get_user_repository)):
    """
    Register a new user.
    :param user_data: New user data
    :param user_repo: User Repository instance
    :return: Created user object
    """

    if await user_repo.email_exists(user_data.email):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

    if await user_repo.username_exists(user_data.username):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already taken")

    hashed_password = hash_password(user_data.password)
    avatar_url = await generate_avatar_url(user_data.username)

    new_user = User(
        email=user_data.email,
        username=user_data.username,
        password_hash=hashed_password,
        avatar_url=avatar_url,
    )

    created_user = await user_repo.create(new_user)

    return created_user


@router.post("/login", response_model=Token)
async def login_user(
    response: Response,
    user_credentials: UserLogin,
    user_repo: IUserRepository = Depends(get_user_repository),
    redis: Redis = Depends(get_redis),
):
    """
    Login user and set HttpOnly cookies (access + refresh + CSRF).

    Sets three cookies for secure authentication:
    - tg_access: Short-lived access token (HttpOnly)
    - tg_refresh: Long-lived refresh token (HttpOnly)
    - tg_csrf: CSRF token (readable by JavaScript for header injection)

    Also returns token in JSON response for backward compatibility with existing clients.

    :param response: FastAPI Response object for setting cookies
    :param user_credentials: User login credentials
    :param user_repo: User Repository instance
    :param redis: Redis client for refresh token tracking
    :return: JWT token object (use cookies instead for production)
    """
    # Validate credentials
    user = await user_repo.get_by_email(user_credentials.email)

    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    if not verify_password(user_credentials.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User account is inactive")

    # Generate token family ID for rotation tracking
    token_family_id = generate_csrf_token()  # Reuse secure random generator

    # Generate tokens
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)

    refresh_token_expires = timedelta(days=settings.refresh_token_expire_days)
    refresh_token = create_refresh_token(
        data={"sub": user.username, "family_id": token_family_id}, expires_delta=refresh_token_expires
    )

    # Generate CSRF token
    csrf_token = generate_csrf_token()

    # Store refresh token in Redis (for revocation capability)
    refresh_payload = verify_token(refresh_token)
    refresh_jti = refresh_payload["jti"]

    redis_key = f"refresh_token:{user.id}:{refresh_jti}"
    await redis.setex(
        redis_key,
        settings.refresh_token_expire_days * SECONDS_PER_DAY,
        csrf_token,  # Store CSRF with refresh token for validation
    )

    # Initialize token family for reuse detection
    family_key = f"refresh_token_family:{user.id}:{token_family_id}"
    await redis.sadd(family_key, refresh_jti)
    await redis.expire(family_key, settings.refresh_token_expire_days * SECONDS_PER_DAY)

    # Set authentication cookies
    set_auth_cookies(response, access_token, refresh_token, csrf_token)

    # JSON response (for backward compatibility, but production apps should use cookies)
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": settings.access_token_expire_minutes * SECONDS_PER_MINUTE,
    }


@router.post("/refresh", response_model=Token)
async def refresh_access_token(
    request: Request,
    response: Response,
    user_repo: IUserRepository = Depends(get_user_repository),
    redis: Redis = Depends(get_redis),
):
    """
    Refresh access token using refresh token from HttpOnly cookie with rotation.

    Security Features (OWASP 2025):
    - Refresh token rotation: New refresh token issued on every refresh
    - Reuse detection: Detects token theft and revokes entire token family
    - Single-use tokens: Old refresh token is immediately invalidated

    This endpoint does NOT require CSRF validation (read-only operation).

    :param request: FastAPI Request object for reading cookies
    :param response: FastAPI Response object for updating cookies
    :param user_repo: User Repository instance
    :param redis: Redis client for refresh token validation
    :return: New access token
    """
    # Get refresh token from HttpOnly cookie
    refresh_token = request.cookies.get("tg_refresh")
    if not refresh_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token missing. Please log in again.",
        )

    # Verify and decode refresh token
    try:
        payload = verify_token(refresh_token)
    except HTTPException:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token. Please log in again.",
        )

    # Validate token type
    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type. Expected refresh token.",
        )

    username = payload.get("sub")
    old_refresh_jti = payload.get("jti")
    token_family_id = payload.get("family_id")

    # Verify user still exists
    user = await user_repo.get_by_username(username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found. Please log in again.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is inactive.",
        )

    # Check if refresh token exists in Redis
    redis_key = f"refresh_token:{user.id}:{old_refresh_jti}"
    token_data = await redis.get(redis_key)

    if not token_data:
        # Token not found - could be revoked or reused
        # Check if token family exists (indicates potential reuse)
        family_key = f"refresh_token_family:{user.id}:{token_family_id}"
        family_exists = await redis.exists(family_key)

        if family_exists:
            # SECURITY EVENT: Token reuse detected!
            # Revoke entire token family to prevent further compromise
            await _revoke_token_family(redis, user.id, token_family_id)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token reuse detected. All sessions revoked. Please log in again.",
            )
        else:
            # Token expired or manually revoked
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token has been revoked. Please log in again.",
            )

    # Decode token data from Redis (format: "csrf_token")
    csrf_token = token_data.decode("utf-8") if isinstance(token_data, bytes) else token_data

    # Delete old refresh token (single-use token pattern)
    await redis.delete(redis_key)

    # Generate NEW tokens (rotation)
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    new_access_token = create_access_token(data={"sub": username}, expires_delta=access_token_expires)

    refresh_token_expires = timedelta(days=settings.refresh_token_expire_days)
    new_refresh_token = create_refresh_token(
        data={"sub": username, "family_id": token_family_id}, expires_delta=refresh_token_expires
    )

    # Store new refresh token in Redis
    new_refresh_payload = verify_token(new_refresh_token)
    new_refresh_jti = new_refresh_payload["jti"]

    new_redis_key = f"refresh_token:{user.id}:{new_refresh_jti}"
    await redis.setex(
        new_redis_key,
        settings.refresh_token_expire_days * SECONDS_PER_DAY,
        csrf_token,  # Keep same CSRF token for session continuity
    )

    # Add new token to family for reuse detection
    family_key = f"refresh_token_family:{user.id}:{token_family_id}"
    await redis.sadd(family_key, new_refresh_jti)
    await redis.expire(family_key, settings.refresh_token_expire_days * SECONDS_PER_DAY)

    # Update BOTH cookies (access + refresh)
    set_auth_cookies(response, new_access_token, new_refresh_token, csrf_token)

    return {
        "access_token": new_access_token,
        "token_type": "bearer",
        "expires_in": settings.access_token_expire_minutes * SECONDS_PER_MINUTE,
    }


@router.post("/logout", response_model=MessageResponse)
async def logout_user(
    request: Request,
    response: Response,
    redis: Redis = Depends(get_redis),
    current_user: User = Depends(get_authenticated_user_with_csrf),
) -> MessageResponse:
    """
    Logout user by invalidating cookies and revoking entire token family.

    With token rotation, we revoke the entire token family to ensure
    all rotated tokens from this login session are invalidated.

    Performs three actions:
    1. Retrieves refresh token from cookie
    2. Revokes entire token family from Redis
    3. Clears all auth cookies

    Requires authentication to prevent unauthorized logout attacks.

    :param request: FastAPI Request object for reading cookies
    :param response: FastAPI Response object for clearing cookies
    :param redis: Redis client for token revocation
    :param current_user: Current authenticated user
    :return: Logout confirmation message
    """
    # Get refresh token for revocation
    refresh_token = request.cookies.get("tg_refresh")

    if refresh_token:
        try:
            payload = verify_token(refresh_token)
            refresh_jti = payload.get("jti")
            token_family_id = payload.get("family_id")

            if token_family_id:
                # Revoke entire token family (all rotated tokens from this login)
                await _revoke_token_family(redis, current_user.id, token_family_id)
            else:
                # Fallback: Delete single token (for old tokens without family_id)
                redis_key = f"refresh_token:{current_user.id}:{refresh_jti}"
                await redis.delete(redis_key)

        except Exception:
            # Continue with logout even if revocation fails
            # (token might be already expired or invalid)
            pass

    # Clear all auth cookies
    clear_auth_cookies(response)

    return MessageResponse(message="Logged out successfully")


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """
    Get current user info.
    :param current_user: Current authenticated user
    :return: User information
    """
    return current_user


@router.patch("/me", response_model=UserResponse)
async def update_user_preferences(
    user_update: UserUpdate,
    current_user: User = Depends(get_authenticated_user_with_csrf),
    user_repo: IUserRepository = Depends(get_user_repository),
):
    """
    Update current user preferences.
    :param user_update: User update data
    :param current_user: Current authenticated user
    :param user_repo: User Repository instance
    :return: Updated user object
    """
    if user_update.username:
        if await user_repo.username_exists(user_update.username):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already taken")
        current_user.username = user_update.username

    if user_update.preferred_language:
        if not validate_language_code(user_update.preferred_language):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported language code: {user_update.preferred_language}",
            )
        current_user.preferred_language = user_update.preferred_language.lower()

    updated_user = await user_repo.update(current_user)
    return updated_user


@router.get("/users/me/quota", response_model=UserQuotaResponse)
async def get_my_quota(
    current_user: User = Depends(get_current_active_user),
) -> UserQuotaResponse:
    """
    Get current user's weekly message quota status.

    Returns quota information including:
    - Weekly limit
    - Messages used this week
    - Messages remaining
    - Last reset date (when current week started)
    - Next reset date (when quota will reset)
    - Percentage used

    :param current_user: Current authenticated user
    :return: User quota status
    """
    # Calculate remaining messages
    remaining = max(0, current_user.weekly_message_limit - current_user.weekly_message_count)

    # Calculate percentage (handle unlimited case)
    if current_user.weekly_message_limit == -1:
        percentage_used = 0.0
    else:
        percentage_used = (
            (current_user.weekly_message_count / current_user.weekly_message_limit) * 100
            if current_user.weekly_message_limit > 0
            else 0.0
        )

    # Calculate next reset date (current week start + 7 days)
    next_reset_date = current_user.weekly_reset_date + timedelta(days=7)

    return UserQuotaResponse(
        weekly_limit=current_user.weekly_message_limit,
        used=current_user.weekly_message_count,
        remaining=remaining,
        last_reset_date=current_user.weekly_reset_date,
        next_reset_date=next_reset_date,
        percentage_used=round(percentage_used, 2),
    )


@router.get("/admin/users/quota-exceeded", response_model=list[UserQuotaExceededResponse])
async def get_quota_exceeded_users(
    current_admin: User = Depends(get_admin_user_with_csrf),
    user_repo: IUserRepository = Depends(get_user_repository),
) -> list[UserQuotaExceededResponse]:
    """
    Get all users who have exceeded their weekly message quota.

    Admin-only endpoint. Returns list of users with quota exceeded status.

    :param current_admin: Current admin user
    :param user_repo: User repository instance
    :return: List of users who exceeded quota
    """
    exceeded_users = await user_repo.get_quota_exceeded_users()

    return [
        UserQuotaExceededResponse(
            user_id=user.id,
            username=user.username,
            email=user.email,
            limit=user.weekly_message_limit,
            used=user.weekly_message_count,
            last_reset_date=user.weekly_reset_date,
            next_reset_date=user.weekly_reset_date + timedelta(days=7),
        )
        for user in exceeded_users
    ]
