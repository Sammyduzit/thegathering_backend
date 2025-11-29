from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.core.validators import SanitizedString
from app.models.ai_entity import AIEntityStatus
from app.models.conversation import ConversationType
from app.models.message import MessageType
from app.models.user import UserStatus


class MessageCreate(BaseModel):
    """
    Schema for creating a room message.
    """

    content: SanitizedString = Field(min_length=1, max_length=500, description="Message content")


class MessageResponse(BaseModel):
    """
    Message response.
    """

    id: int
    sender_id: int
    sender_username: str
    content: str
    message_type: MessageType
    sent_at: datetime

    room_id: int | None = Field(None, description="Room ID for room-wide chat")
    conversation_id: int | None = Field(None, description="Conversation ID for private/group chat")

    model_config = ConfigDict(from_attributes=True)


class ConversationCreate(BaseModel):
    """
    Schema for creating conversations.
    """

    participant_usernames: list[str] = Field(
        min_length=1,
        max_length=20,
        description="List of usernames to include in conversation",
    )
    conversation_type: ConversationType = Field(description="Conversation type: private or group")


class ConversationUpdate(BaseModel):
    """
    Schema for updating conversation metadata.
    Currently supports archiving/unarchiving conversations.
    """

    is_active: bool = Field(description="Set to false to archive, true to unarchive")


class ConversationResponse(BaseModel):
    """
    Conversation response.
    """

    id: int
    conversation_type: ConversationType
    room_id: int
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ParticipantInfo(BaseModel):
    """
    Participant information for conversation responses.
    """

    id: int
    username: str
    avatar_url: str | None = None
    status: UserStatus | AIEntityStatus
    is_ai: bool

    model_config = ConfigDict(from_attributes=True)


class ConversationPermissions(BaseModel):
    """
    User permissions for a conversation.
    """

    can_post: bool = Field(description="User can send messages")
    can_manage_participants: bool = Field(description="User can add/remove participants")
    can_leave: bool = Field(description="User can leave conversation")


class ConversationListItemResponse(BaseModel):
    """
    Conversation list item response for overview pages.
    Compact format with essential metadata.
    """

    id: int
    type: ConversationType = Field(description="Conversation type: private or group")
    room_id: int
    room_name: str | None = Field(None, description="Name of the room this conversation belongs to")
    participants: list[str] = Field(description="List of participant usernames (excluding current user)")
    participant_count: int = Field(description="Total number of participants")
    created_at: datetime
    latest_message_at: datetime | None = Field(None, description="Timestamp of most recent message")
    latest_message_preview: str | None = Field(None, description="Preview of latest message (first 50 chars)")


class ConversationDetailResponse(BaseModel):
    """
    Detailed conversation response for conversation detail pages.
    Includes full participant info, permissions, and message metadata.
    """

    id: int
    type: ConversationType = Field(description="Conversation type: private or group")
    room_id: int
    room_name: str | None = Field(None, description="Name of the room this conversation belongs to")
    is_active: bool
    created_at: datetime

    # Participants
    participants: list[ParticipantInfo] = Field(description="Full participant details")
    participant_count: int

    # Message metadata
    message_count: int = Field(description="Total number of messages in conversation")
    latest_message: MessageResponse | None = Field(None, description="Most recent message")

    # Permissions
    permissions: ConversationPermissions = Field(description="Current user's permissions")


class PaginatedMessagesResponse(BaseModel):
    """
    Paginated message response with metadata.
    """

    messages: list[MessageResponse] = Field(description="List of messages for current page")
    total: int = Field(description="Total number of messages in conversation")
    page: int = Field(ge=1, description="Current page number")
    page_size: int = Field(ge=1, le=100, description="Number of messages per page")
    total_pages: int = Field(description="Total number of pages")
    has_more: bool = Field(description="Whether more pages are available")


class ConversationCreateResponse(BaseModel):
    """
    Response for successful conversation creation.
    """

    message: str = Field(description="Confirmation message")
    conversation_id: int = Field(description="ID of created conversation")
    participants: int = Field(ge=1, description="Total number of participants")


class ConversationUpdateResponse(BaseModel):
    """
    Response for conversation update (archive/unarchive).
    """

    message: str = Field(description="Update confirmation message")
    conversation_id: int = Field(description="ID of updated conversation")
    is_active: bool = Field(description="Current active status")


class ParticipantAddResponse(BaseModel):
    """
    Response for adding participant to conversation.
    """

    message: str = Field(description="Success message")
    conversation_id: int = Field(description="Conversation ID")
    username: str = Field(description="Added participant username")
    participant_count: int = Field(ge=1, description="Total participants after addition")


class ParticipantRemoveResponse(BaseModel):
    """
    Response for removing participant from conversation.
    """

    message: str = Field(description="Success message")
    conversation_id: int = Field(description="Conversation ID")
    username: str = Field(description="Removed participant username")
    participant_count: int = Field(ge=0, description="Total participants after removal")
