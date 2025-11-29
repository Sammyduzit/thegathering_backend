from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.ai_entity import AIEntity
    from app.models.conversation import Conversation
    from app.models.message import Message
    from app.models.user import User


class Room(Base):
    """Room model"""

    __tablename__ = "rooms"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    description: Mapped[str | None] = mapped_column(Text, default=None)
    max_users: Mapped[int | None] = mapped_column(default=None)

    is_translation_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    has_ai: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    users: Mapped[list["User"]] = relationship(back_populates="current_room")
    ai_entities: Mapped[list["AIEntity"]] = relationship(back_populates="current_room", lazy="raise")
    conversations: Mapped[list["Conversation"]] = relationship(back_populates="room")
    room_messages: Mapped[list["Message"]] = relationship(back_populates="room", lazy="dynamic")

    def __repr__(self):
        return f"<Room(id={self.id}, name='{self.name}')>"
