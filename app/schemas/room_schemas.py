from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.core.validators import SanitizedString


class RoomResponse(BaseModel):
    """
    Schema for room responses.
    """

    id: int
    name: str
    description: str | None = None
    max_users: int | None = None
    is_translation_enabled: bool
    is_active: bool
    has_ai: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class RoomCreate(BaseModel):
    """
    Schema for creating a new room.
    Input validation for POST requests.
    """

    name: SanitizedString = Field(min_length=1, max_length=100)
    description: SanitizedString | None = Field(None, max_length=500)
    max_users: int | None = Field(None, ge=1, le=1000)
    is_translation_enabled: bool = Field(False, description="Enable automatic translation in this room")


class RoomDeleteResponse(BaseModel):
    """
    Room deletion cleanup summary.

    Returns detailed statistics about the cleanup operation when a room is deleted.
    """

    message: str = Field(description="Deletion confirmation message")
    room_id: int = Field(description="ID of deleted room")
    users_removed: int = Field(ge=0, description="Number of users removed from room")
    conversations_archived: int = Field(ge=0, description="Number of conversations archived")
    messages_deleted: int = Field(ge=0, description="Number of messages deleted")
