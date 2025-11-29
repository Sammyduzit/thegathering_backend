from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.core.validators import SanitizedString, SanitizedUsername


class UserRegister(BaseModel):
    """
    Schema for user registration.
    """

    email: EmailStr = Field(description="User email address")
    username: SanitizedString = Field(min_length=3, max_length=20, description="Username")
    password: str = Field(min_length=8, max_length=70, description="Password")


class UserLogin(BaseModel):
    """
    Schema for user login.
    """

    email: EmailStr = Field(description="User email address")
    password: str = Field(min_length=8, max_length=70, description="Password")


class Token(BaseModel):
    """
    JWT Token response.
    """

    access_token: str = Field(description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(description="Token expiration time")


class UserResponse(BaseModel):
    """
    User data response.
    """

    id: int
    email: EmailStr
    username: str
    avatar_url: str | None = None
    preferred_language: str | None = None
    is_active: bool
    is_admin: bool
    created_at: datetime
    last_active: datetime
    current_room_id: int | None = None

    model_config = ConfigDict(from_attributes=True)


class UserUpdate(BaseModel):
    """
    Schema for Updating User information.
    """

    preferred_language: str | None = Field(
        None,
        description="Preferred language code (EN, DE, FR, etc.",
        min_length=2,
        max_length=5,
    )
    username: SanitizedUsername | None = Field(None, min_length=3, max_length=20, description="New username")


class UserQuotaResponse(BaseModel):
    """
    User's weekly message quota status.
    """

    weekly_limit: int = Field(description="Weekly message limit (-1 = unlimited)")
    used: int = Field(description="Messages sent this week")
    remaining: int = Field(description="Messages remaining this week")
    last_reset_date: datetime = Field(description="Date when current week started (last reset)")
    next_reset_date: datetime = Field(description="Date when quota will reset (in 7 days)")
    percentage_used: float = Field(description="Percentage of quota used")

    model_config = ConfigDict(from_attributes=True)


class UserQuotaExceededResponse(BaseModel):
    """
    User information for quota exceeded admin endpoint.
    """

    user_id: int = Field(description="User ID")
    username: str = Field(description="Username")
    email: EmailStr = Field(description="User email")
    limit: int = Field(description="Weekly message limit")
    used: int = Field(description="Messages sent this week")
    last_reset_date: datetime = Field(description="Date when current week started (last reset)")
    next_reset_date: datetime = Field(description="Date when quota will reset (in 7 days)")

    model_config = ConfigDict(from_attributes=True)
