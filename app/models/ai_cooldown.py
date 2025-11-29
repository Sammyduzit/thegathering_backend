"""AI Cooldown Model - Separate table for performance & atomicity."""

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.ai_entity import AIEntity
    from app.models.conversation import Conversation
    from app.models.room import Room


class AICooldown(Base):
    """AI Cooldown tracking for rate limiting."""

    __tablename__ = "ai_cooldowns"

    id: Mapped[int] = mapped_column(primary_key=True)

    ai_entity_id: Mapped[int] = mapped_column(ForeignKey("ai_entities.id", ondelete="CASCADE"), index=True)

    # Context: EITHER Room OR Conversation (XOR enforced by check constraint)
    room_id: Mapped[int | None] = mapped_column(ForeignKey("rooms.id", ondelete="CASCADE"), default=None, index=True)
    conversation_id: Mapped[int | None] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), default=None, index=True
    )

    last_response_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    # Relationships
    ai_entity: Mapped["AIEntity"] = relationship(back_populates="cooldowns")
    room: Mapped["Room | None"] = relationship()
    conversation: Mapped["Conversation | None"] = relationship()

    __table_args__ = (
        CheckConstraint(
            "(room_id IS NULL) != (conversation_id IS NULL)",
            name="ai_cooldown_xor_room_conversation",
        ),
        UniqueConstraint(
            "ai_entity_id",
            "room_id",
            "conversation_id",
            name="uq_ai_cooldown_context",
        ),
    )

    def __repr__(self):
        context = f"room={self.room_id}" if self.room_id else f"conv={self.conversation_id}"
        return f"<AICooldown(ai={self.ai_entity_id}, {context}, last={self.last_response_at})>"
