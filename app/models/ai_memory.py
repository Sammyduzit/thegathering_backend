from datetime import datetime
from typing import TYPE_CHECKING, Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, Integer, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.ai_entity import AIEntity


class AIMemory(Base):
    """AI conversation memory storage using PostgreSQL JSONB."""

    __tablename__ = "ai_memories"

    id: Mapped[int] = mapped_column(primary_key=True)
    entity_id: Mapped[int] = mapped_column(ForeignKey("ai_entities.id", ondelete="CASCADE"))
    user_ids: Mapped[list[int]] = mapped_column(ARRAY(Integer).with_variant(JSON(), "sqlite"), default=list)

    # Context linking
    room_id: Mapped[int | None] = mapped_column(ForeignKey("rooms.id", ondelete="CASCADE"), default=None)
    conversation_id: Mapped[int | None] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), default=None
    )

    # Memory data (AI-readable summary + structured JSONB content)
    summary: Mapped[str] = mapped_column(Text)
    memory_content: Mapped[dict[str, Any]] = mapped_column(JSON().with_variant(JSONB(none_as_null=True), "postgresql"))

    # Retrieval metadata
    keywords: Mapped[list[str] | None] = mapped_column(
        JSON().with_variant(JSONB(none_as_null=True), "postgresql"), default=None
    )
    importance_score: Mapped[float] = mapped_column(Float, default=1.0)

    # Vector search support (pgvector for semantic search, PostgreSQL only)
    embedding: Mapped[Any | None] = mapped_column(Vector(1536), default=None)  # type: ignore

    # Access tracking for importance adjustment
    access_count: Mapped[int] = mapped_column(Integer, default=0)

    # Flexible metadata (extractor version, creation method, etc.)
    memory_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        JSON().with_variant(JSONB(none_as_null=True), "postgresql"), default=None
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_accessed: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    entity: Mapped["AIEntity"] = relationship(back_populates="memories")

    __table_args__ = (
        Index("idx_ai_memory_entity_room", "entity_id", "room_id"),
        Index("idx_ai_memory_keywords", "keywords", postgresql_using="gin"),
        Index("idx_ai_memory_access_count", "access_count"),
        Index(
            "idx_ai_memory_user_ids",
            "user_ids",
            postgresql_using="gin",
            postgresql_ops={"user_ids": "array_ops"},
        ),
        Index("ai_memories_created_at_idx", "created_at"),
        Index(
            "ai_memories_embedding_idx",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    def __repr__(self):
        return f"<AIMemory(id={self.id}, entity_id={self.entity_id})>"
