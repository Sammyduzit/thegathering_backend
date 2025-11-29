import uuid
from datetime import UTC, datetime, timedelta

import jwt
from fastapi import HTTPException, status

from app.core.config import settings


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """
    Create JWT access token with unique JTI (JWT ID).

    JTI enables token revocation and tracking. Each token gets a unique identifier
    that can be used for blacklisting or session management.

    :param data: Data to encode. Must contain 'sub' key with username value.
                 Example: {"sub": "alice"}
    :param expires_delta: Token expiration time
    :return: JWT token string with jti, exp, iat claims
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(minutes=settings.access_token_expire_minutes)

    to_encode.update(
        {
            "exp": expire,
            "iat": datetime.now(UTC),
            "jti": str(uuid.uuid4()),
            "type": "access",
        }
    )

    encode_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)

    return encode_jwt


def create_refresh_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """
    Create JWT refresh token with unique JTI (JWT ID).

    Refresh tokens are long-lived and stored in Redis for revocation capability.
    The JTI is used as part of the Redis key for tracking active refresh tokens.

    :param data: Data to encode. Must contain 'sub' key with username value.
                 Example: {"sub": "alice"}
    :param expires_delta: Token expiration time
    :return: JWT token string with jti, exp, iat claims
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(days=settings.refresh_token_expire_days)

    to_encode.update(
        {
            "exp": expire,
            "iat": datetime.now(UTC),
            "jti": str(uuid.uuid4()),
            "type": "refresh",
        }
    )

    encode_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)

    return encode_jwt


def verify_token(token: str) -> dict:
    """
    Verify and decode JWT token.
    :param token: JWT token string
    :return: Decoded token
    """
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        return payload

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_user_from_token(token: str) -> str:
    """
    Extract username from JWT token.
    :param token: JWT token string
    :return: Username from token
    """
    payload = verify_token(token)
    username: str = payload.get("sub")

    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return username
