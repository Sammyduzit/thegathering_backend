import enum
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.conversation_participant import ConversationParticipant
    from app.models.message import Message
    from app.models.room import Room


class ConversationType(enum.Enum):
    """Conversation types"""

    PRIVATE = "private"
    GROUP = "group"


class Conversation(Base):
    """Conversation Model"""

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    room_id: Mapped[int] = mapped_column(ForeignKey("rooms.id", ondelete="CASCADE"))
    conversation_type: Mapped[ConversationType] = mapped_column(
        Enum(ConversationType), default=ConversationType.PRIVATE
    )
    max_participants: Mapped[int | None] = mapped_column(default=None)  # 2 for private, NULL for group chat

    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())

    room: Mapped["Room"] = relationship(back_populates="conversations")
    participants: Mapped[list["ConversationParticipant"]] = relationship(back_populates="conversation")
    messages: Mapped[list["Message"]] = relationship(back_populates="conversation", lazy="dynamic")

    def __repr__(self):
        return f"<Conversation(id={self.id}, type={self.conversation_type}, room_id={self.room_id})>"
