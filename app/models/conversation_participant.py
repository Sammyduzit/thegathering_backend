from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, CheckConstraint, DateTime, ForeignKey, Index, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.ai_entity import AIEntity
    from app.models.conversation import Conversation
    from app.models.user import User


class ConversationParticipant(Base):
    """
    Junction table managing user/AI participation in conversations.

    Business Rules:
    - Polymorphic: User OR AI entity (XOR)
    - Private conversations: exactly 2 participants
    - Group conversations: 3+ participants
    - Participants can join/leave conversations (temporal participation)
    - Farewell tracking for AI entities
    """

    __tablename__ = "conversation_participants"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"))

    # Polymorphic: User OR AI (XOR constraint)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), default=None)
    ai_entity_id: Mapped[int | None] = mapped_column(ForeignKey("ai_entities.id", ondelete="CASCADE"), default=None)

    joined_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    left_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    # Farewell tracking for AI (prevents duplicate goodbyes)
    farewell_sent: Mapped[bool] = mapped_column(Boolean, default=False)

    # Relationships
    conversation: Mapped["Conversation"] = relationship(back_populates="participants")
    user: Mapped["User | None"] = relationship(back_populates="conversation_participations")
    ai_entity: Mapped["AIEntity | None"] = relationship(back_populates="conversation_participations")

    __table_args__ = (
        CheckConstraint(
            "(user_id IS NULL) != (ai_entity_id IS NULL)",
            name="participant_xor_user_ai",
        ),
        UniqueConstraint("conversation_id", "user_id", name="uq_conversation_user"),
        UniqueConstraint("conversation_id", "ai_entity_id", name="uq_conversation_ai"),
        Index("idx_conversation_user", "conversation_id", "user_id"),
        Index("idx_conversation_ai", "conversation_id", "ai_entity_id"),
        Index("idx_user_participation_history", "user_id", "joined_at"),
        Index("idx_active_participants", "conversation_id", "left_at"),
    )

    @property
    def participant_name(self) -> str:
        """Get participant username regardless of User or AI."""
        if self.user_id and self.user:
            return self.user.username
        if self.ai_entity:
            return self.ai_entity.username
        return ""

    @property
    def is_ai(self) -> bool:
        """Check if participant is AI."""
        return self.ai_entity_id is not None

    @property
    def participant_id(self) -> int:
        """Get ID regardless of type."""
        return self.user_id if self.user_id else self.ai_entity_id  # type: ignore

    def __repr__(self):
        participant_type = "ai" if self.is_ai else "user"
        return (
            f"<ConversationParticipant(conversation={self.conversation_id}, {participant_type}={self.participant_id})>"
        )
