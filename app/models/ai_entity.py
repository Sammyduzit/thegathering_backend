"""
AI Entity model for LangChain-powered chat agents.

Follows User model pattern with configuration for OpenAI/LangChain integration.
"""

import enum
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, Boolean, DateTime, Enum, Float, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates
from sqlalchemy.sql import func, text

from app.core.constants import (
    DEFAULT_AI_COOLDOWN_SECONDS,
    DEFAULT_AI_MAX_TOKENS,
    DEFAULT_AI_MODEL,
    DEFAULT_AI_TEMPERATURE,
    MAX_AI_COOLDOWN_SECONDS,
    MIN_AI_COOLDOWN_SECONDS,
)
from app.core.database import Base

if TYPE_CHECKING:
    from app.models.ai_cooldown import AICooldown
    from app.models.ai_memory import AIMemory
    from app.models.conversation_participant import ConversationParticipant
    from app.models.message import Message
    from app.models.room import Room


class AIEntityStatus(enum.Enum):
    """AI entity online status."""

    ONLINE = "online"
    OFFLINE = "offline"


class AIResponseStrategy(str, enum.Enum):
    """AI response behavior strategies."""

    ROOM_MENTION_ONLY = "room_mention_only"
    ROOM_PROBABILISTIC = "room_probabilistic"
    ROOM_ACTIVE = "room_active"
    CONV_EVERY_MESSAGE = "conv_every_message"
    CONV_ON_QUESTIONS = "conv_on_questions"
    CONV_SMART = "conv_smart"
    NO_RESPONSE = "no_response"


class AIEntity(Base):
    """AI entity - equal to User in The Gathering world."""

    __tablename__ = "ai_entities"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    description: Mapped[str | None] = mapped_column(Text, default=None)

    # LangChain/OpenAI Configuration
    system_prompt: Mapped[str] = mapped_column(Text)
    model_name: Mapped[str] = mapped_column(String(100), default=DEFAULT_AI_MODEL)
    temperature: Mapped[float] = mapped_column(Float, default=DEFAULT_AI_TEMPERATURE)
    max_tokens: Mapped[int] = mapped_column(default=DEFAULT_AI_MAX_TOKENS)

    # Response Strategies
    room_response_strategy: Mapped[AIResponseStrategy] = mapped_column(
        Enum(AIResponseStrategy), default=AIResponseStrategy.ROOM_MENTION_ONLY, index=True
    )
    conversation_response_strategy: Mapped[AIResponseStrategy] = mapped_column(
        Enum(AIResponseStrategy), default=AIResponseStrategy.CONV_ON_QUESTIONS, index=True
    )
    response_probability: Mapped[float] = mapped_column(Float, default=0.3)

    # Cooldown Configuration (rate limiting between responses)
    cooldown_seconds: Mapped[int | None] = mapped_column(default=DEFAULT_AI_COOLDOWN_SECONDS)

    # Flexible config storage (JSONB for additional LangChain parameters)
    config: Mapped[dict[str, Any] | None] = mapped_column(
        JSON().with_variant(JSONB(none_as_null=True), "postgresql"), default=None
    )

    status: Mapped[AIEntityStatus] = mapped_column(Enum(AIEntityStatus), default=AIEntityStatus.OFFLINE, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), onupdate=func.now(), default=None)

    # Room presence
    current_room_id: Mapped[int | None] = mapped_column(
        ForeignKey("rooms.id", ondelete="SET NULL"), default=None, index=True
    )

    # Relationships
    current_room: Mapped["Room | None"] = relationship(back_populates="ai_entities")
    memories: Mapped[list["AIMemory"]] = relationship(back_populates="entity", lazy="raise")
    sent_messages: Mapped[list["Message"]] = relationship(
        back_populates="sender_ai", foreign_keys="Message.sender_ai_id", lazy="raise"
    )
    conversation_participations: Mapped[list["ConversationParticipant"]] = relationship(
        back_populates="ai_entity", foreign_keys="ConversationParticipant.ai_entity_id", lazy="raise"
    )

    # Cooldown Tracking
    cooldowns: Mapped[list["AICooldown"]] = relationship(
        back_populates="ai_entity", lazy="raise", cascade="all, delete-orphan"
    )

    __table_args__ = (
        # Partial unique index: Only one AI can be assigned to a room at a time
        # This prevents race conditions when multiple AIs try to join the same room
        Index(
            "idx_unique_ai_per_room",
            "current_room_id",
            unique=True,
            postgresql_where=text("current_room_id IS NOT NULL"),
        ),
    )

    @validates("response_probability")
    def validate_probability(self, key, value):
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"response_probability must be 0.0-1.0, got {value}")
        return value

    @validates("temperature")
    def validate_temperature(self, key, value):
        if value is not None and not 0.0 <= value <= 2.0:
            raise ValueError(f"temperature must be 0.0-2.0, got {value}")
        return value

    @validates("cooldown_seconds")
    def validate_cooldown_seconds(self, key, value):
        if value is not None and not (MIN_AI_COOLDOWN_SECONDS <= value <= MAX_AI_COOLDOWN_SECONDS):
            raise ValueError(
                f"cooldown_seconds must be {MIN_AI_COOLDOWN_SECONDS}-{MAX_AI_COOLDOWN_SECONDS}, got {value}"
            )
        return value

    @property
    def current_room_name(self) -> str | None:
        """Get current room name if assigned."""
        return self.current_room.name if self.current_room else None

    def __repr__(self):
        return f"<AIEntity(id={self.id}, username='{self.username}')>"
