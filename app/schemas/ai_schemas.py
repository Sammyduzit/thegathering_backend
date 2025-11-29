from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.core.constants import (
    MAX_AI_COOLDOWN_SECONDS,
    MAX_AI_MAX_TOKENS,
    MAX_AI_TEMPERATURE,
    MIN_AI_COOLDOWN_SECONDS,
    MIN_AI_MAX_TOKENS,
    MIN_AI_TEMPERATURE,
)
from app.core.validators import SanitizedString
from app.models.ai_entity import AIEntityStatus, AIResponseStrategy


class AIEntityCreate(BaseModel):
    """Schema for creating a new AI entity."""

    username: SanitizedString = Field(min_length=1, max_length=200, description="Unique AI username")
    description: str | None = Field(None, max_length=1000, description="AI entity description")
    system_prompt: str = Field(min_length=1, max_length=5000, description="AI system prompt/instructions")
    model_name: SanitizedString = Field(min_length=1, max_length=100, description="LLM model name")
    temperature: float | None = Field(None, ge=MIN_AI_TEMPERATURE, le=MAX_AI_TEMPERATURE, description="LLM temperature")
    max_tokens: int | None = Field(None, ge=MIN_AI_MAX_TOKENS, le=MAX_AI_MAX_TOKENS, description="Max response tokens")

    # Response Strategies
    room_response_strategy: AIResponseStrategy | None = Field(
        None, description="How AI responds in rooms (MENTION_ONLY, PROBABILISTIC, ACTIVE, NO_RESPONSE)"
    )
    conversation_response_strategy: AIResponseStrategy | None = Field(
        None, description="How AI responds in conversations (EVERY_MESSAGE, ON_QUESTIONS, SMART, NO_RESPONSE)"
    )
    response_probability: float | None = Field(
        None, ge=0.0, le=1.0, description="Response probability for PROBABILISTIC strategy (0.0-1.0)"
    )

    # Rate Limiting
    cooldown_seconds: int | None = Field(
        None,
        ge=MIN_AI_COOLDOWN_SECONDS,
        le=MAX_AI_COOLDOWN_SECONDS,
        description="Minimum seconds between responses (None = no cooldown)",
    )

    config: dict | None = Field(None, description="Additional LangChain configuration")


class AIEntityUpdate(BaseModel):
    """Schema for updating an AI entity."""

    username: SanitizedString | None = Field(None, min_length=1, max_length=200)
    description: str | None = Field(None, max_length=1000)
    system_prompt: str | None = Field(None, min_length=1, max_length=5000)
    model_name: SanitizedString | None = Field(None, min_length=1, max_length=100)
    temperature: float | None = Field(None, ge=MIN_AI_TEMPERATURE, le=MAX_AI_TEMPERATURE)
    max_tokens: int | None = Field(None, ge=MIN_AI_MAX_TOKENS, le=MAX_AI_MAX_TOKENS)

    # Response Strategies
    room_response_strategy: AIResponseStrategy | None = Field(None, description="Room response strategy")
    conversation_response_strategy: AIResponseStrategy | None = Field(
        None, description="Conversation response strategy"
    )
    response_probability: float | None = Field(None, ge=0.0, le=1.0, description="Response probability (0.0-1.0)")

    # Rate Limiting
    cooldown_seconds: int | None = Field(
        None,
        ge=MIN_AI_COOLDOWN_SECONDS,
        le=MAX_AI_COOLDOWN_SECONDS,
        description="Cooldown seconds (None = no cooldown)",
    )

    config: dict | None = None
    status: AIEntityStatus | None = Field(None, description="AI online/offline status")
    current_room_id: int | None = Field(
        None, description="Room assignment (None=leave room, int=assign to room, omit=no change)"
    )


class AIEntityResponse(BaseModel):
    """Schema for AI entity responses."""

    id: int
    username: str
    description: str | None
    system_prompt: str
    model_name: str
    temperature: float | None
    max_tokens: int | None

    # Response Strategies
    room_response_strategy: AIResponseStrategy
    conversation_response_strategy: AIResponseStrategy
    response_probability: float

    # Rate Limiting
    cooldown_seconds: int | None

    config: dict | None
    status: AIEntityStatus
    is_active: bool
    current_room_id: int | None
    current_room_name: str | None  # Populated from AIEntity.current_room_name @property
    created_at: datetime
    updated_at: datetime | None

    model_config = ConfigDict(from_attributes=True)


class AIInviteRequest(BaseModel):
    """Schema for inviting AI to a conversation."""

    ai_entity_id: int = Field(gt=0, description="ID of AI entity to invite")


class AIAvailableResponse(BaseModel):
    """Schema for available AI entities in a room."""

    id: int
    username: str
    model_name: str
    status: AIEntityStatus

    model_config = ConfigDict(from_attributes=True)


class AIGoodbyeResponse(BaseModel):
    """
    Response for AI goodbye initiation.

    Returned when an AI entity initiates a graceful goodbye sequence.
    """

    message: str = Field(description="Goodbye initiation confirmation")
    ai_entity_id: int = Field(description="ID of AI entity saying goodbye")
    conversation_id: int | None = Field(None, description="Conversation ID if leaving conversation")
    room_id: int | None = Field(None, description="Room ID if leaving room")
