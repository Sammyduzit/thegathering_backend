import enum
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.conversation_participant import ConversationParticipant
    from app.models.message import Message
    from app.models.room import Room


class UserStatus(enum.Enum):
    """User presence status"""

    AVAILABLE = "available"
    BUSY = "busy"
    AWAY = "away"


class User(Base):
    """User model"""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))

    avatar_url: Mapped[str | None] = mapped_column(String(500), default=None)
    preferred_language: Mapped[str | None] = mapped_column(String(5), default="en")
    status: Mapped[UserStatus] = mapped_column(Enum(UserStatus), default=UserStatus.AVAILABLE)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_active: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)

    # Weekly message quota tracking
    weekly_message_count: Mapped[int] = mapped_column(Integer, default=0)
    weekly_reset_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    weekly_message_limit: Mapped[int] = mapped_column(Integer, default=100)

    current_room_id: Mapped[int | None] = mapped_column(ForeignKey("rooms.id", ondelete="SET NULL"), default=None)
    current_room: Mapped["Room | None"] = relationship(back_populates="users", lazy="raise")
    conversation_participations: Mapped[list["ConversationParticipant"]] = relationship(back_populates="user")
    sent_messages: Mapped[list["Message"]] = relationship(
        back_populates="sender_user",
        foreign_keys="Message.sender_user_id",
        lazy="raise",
    )

    def __repr__(self):
        return f"<User (id={self.id}, username='{self.username}')>"
