"""Pydantic schemas for AI Memory API requests and responses."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class MemoryTextCreate(BaseModel):
    """Schema for manual long-term memory creation (context-aware, admin-only)."""

    entity_id: int = Field(gt=0, description="AI entity ID")
    conversation_id: int = Field(gt=0, description="Conversation ID (required)")
    user_ids: list[int] = Field(min_length=1, description="Participant user IDs (from frontend)")
    text: str = Field(min_length=1, max_length=500, description="Raw conversation text (max 500 chars)")
    keywords: list[str] | None = Field(None, description="Keywords (auto-extracted if None)")


class MemoryUpdate(BaseModel):
    """Schema for updating an existing AI memory (partial update)."""

    summary: str | None = Field(None, min_length=1, max_length=500, description="Memory summary")
    memory_content: dict | None = Field(None, description="Structured memory content")
    keywords: list[str] | None = Field(None, description="Keywords (re-extracted if summary changes)")
    importance_score: float | None = Field(None, ge=0.0, le=10.0, description="Importance score")


class MemoryResponse(BaseModel):
    """Schema for AI memory responses."""

    id: int
    entity_id: int
    conversation_id: int | None
    room_id: int | None
    summary: str
    memory_content: dict
    keywords: list[str] | None
    importance_score: float
    embedding: list[float] | None  # pgvector array serialized as list
    access_count: int
    memory_metadata: dict | None
    created_at: datetime
    last_accessed: datetime

    model_config = ConfigDict(from_attributes=True)


class MemoryListResponse(BaseModel):
    """Schema for paginated memory list responses."""

    memories: list[MemoryResponse]
    total: int = Field(description="Total number of memories")
    page: int = Field(ge=1, description="Current page number")
    page_size: int = Field(ge=1, le=100, description="Items per page")
    total_pages: int = Field(description="Total number of pages")

    model_config = ConfigDict(from_attributes=True)


class PersonalityUploadRequest(BaseModel):
    """Schema for uploading personality knowledge base."""

    text: str = Field(min_length=1, description="Text content to upload")
    category: str = Field(min_length=1, max_length=50, description="Category (e.g., books, docs)")
    metadata: dict | None = Field(None, description="Additional metadata (e.g., book_title, chapter)")


class PersonalityUploadResponse(BaseModel):
    """Schema for personality upload response."""

    created_memories: int = Field(description="Number of memories created")
    memory_ids: list[int] = Field(description="List of created memory IDs")
    category: str = Field(description="Category of uploaded content")
    chunks: int = Field(description="Number of chunks created")
