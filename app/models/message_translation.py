from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.message import Message


class MessageTranslation(Base):
    """Translation storage for messages in different languages"""

    __tablename__ = "message_translation"

    id: Mapped[int] = mapped_column(primary_key=True)
    message_id: Mapped[int] = mapped_column(ForeignKey("messages.id", ondelete="CASCADE"))
    target_language: Mapped[str] = mapped_column(String(5))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    message: Mapped["Message"] = relationship(back_populates="translations")

    __table_args__ = (
        Index("idx_message_language_unique", "message_id", "target_language", unique=True),
        Index("idx_message_translations", "message_id"),
        Index("idx_language_translations", "target_language"),
    )

    def __repr__(self):
        return f"<MessageTranslation(message_id={self.message_id}, lang={self.target_language})>"
