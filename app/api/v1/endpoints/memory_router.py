"""
Memory API endpoints for AI memory management.

Provides CRUD operations and search functionality for AI memories.
Admin-only access for security and privacy.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.core.auth_dependencies import get_admin_user_with_csrf
from app.core.utils import calculate_pagination
from app.interfaces.keyword_extractor import IKeywordExtractor
from app.models.ai_memory import AIMemory
from app.models.user import User
from app.repositories.ai_memory_repository import IAIMemoryRepository
from app.repositories.conversation_repository import IConversationRepository
from app.repositories.repository_dependencies import (
    get_ai_memory_repository,
    get_conversation_repository,
)
from app.schemas.memory_schemas import (
    MemoryListResponse,
    MemoryResponse,
    MemoryTextCreate,
    MemoryUpdate,
    PersonalityUploadRequest,
    PersonalityUploadResponse,
)
from app.services.embedding.embedding_factory import create_embedding_service
from app.services.memory.personality_memory_service import PersonalityMemoryService
from app.services.service_dependencies import get_keyword_extractor, get_personality_memory_service

router = APIRouter(prefix="/memories", tags=["memories"])


@router.get("", response_model=MemoryListResponse)
async def get_memories(
    entity_id: int | None = Query(None, description="Filter by AI entity ID"),
    conversation_id: int | None = Query(None, description="Filter by conversation ID"),
    room_id: int | None = Query(None, description="Filter by room ID"),
    include_short_term: bool = Query(False, description="Include short-term memories (default: False)"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page"),
    current_admin: User = Depends(get_admin_user_with_csrf),
    memory_repo: IAIMemoryRepository = Depends(get_ai_memory_repository),
) -> MemoryListResponse:
    """
    Get AI memories with pagination and filtering (Admin only).

    By default, short-term memories are excluded. Use include_short_term=true to show all.

    :param entity_id: Optional AI entity ID filter
    :param conversation_id: Optional conversation ID filter
    :param room_id: Optional room ID filter
    :param include_short_term: Include short-term memories (default: False)
    :param page: Page number (starts at 1)
    :param page_size: Items per page (max 100)
    :param current_admin: Current authenticated admin
    :param memory_repo: Memory repository instance
    :return: Paginated list of memories (excludes short-term by default)
    """
    # Calculate offset
    offset = (page - 1) * page_size

    # Get memories based on filters
    if entity_id:
        memories = await memory_repo.get_entity_memories(
            entity_id=entity_id,
            room_id=room_id,
            limit=page_size,
        )
        # For entity-specific queries, we don't have total count easily
        # This is a simplified approach - in production, add count query
        total = len(memories)
    else:
        # Get all memories with pagination
        memories = await memory_repo.get_all(limit=page_size, offset=offset)
        # Simplified total count
        total = len(memories) + offset

    # Filter out short-term memories unless explicitly requested
    if not include_short_term:
        memories = [m for m in memories if not (m.memory_metadata and m.memory_metadata.get("type") == "short_term")]
        total = len(memories)  # Adjust total after filtering

    total_pages, _ = calculate_pagination(total, page, page_size)

    return MemoryListResponse(
        memories=memories,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get("/{memory_id}", response_model=MemoryResponse)
async def get_memory_by_id(
    memory_id: int,
    current_admin: User = Depends(get_admin_user_with_csrf),
    memory_repo: IAIMemoryRepository = Depends(get_ai_memory_repository),
) -> MemoryResponse:
    """
    Get single memory by ID (Admin only).

    :param memory_id: Memory ID
    :param current_admin: Current authenticated admin
    :param memory_repo: Memory repository instance
    :return: Memory details
    :raises HTTPException: 404 if memory not found
    """
    memory = await memory_repo.get_by_id(memory_id)
    if not memory:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Memory {memory_id} not found")

    return memory


@router.post("", response_model=MemoryResponse, status_code=status.HTTP_201_CREATED)
async def create_memory(
    memory_data: MemoryTextCreate,
    current_admin: User = Depends(get_admin_user_with_csrf),
    memory_repo: IAIMemoryRepository = Depends(get_ai_memory_repository),
    conversation_repo: IConversationRepository = Depends(get_conversation_repository),
    keyword_extractor: IKeywordExtractor = Depends(get_keyword_extractor),
) -> MemoryResponse:
    """
    Create manual long-term memory (Admin only, context-aware).

    Frontend provides entity_id, conversation_id, user_ids from context.
    Backend creates embedding and extracts keywords automatically.

    :param memory_data: Memory creation data (text, entity_id, conversation_id, user_ids)
    :param current_admin: Current authenticated admin
    :param memory_repo: Memory repository instance
    :param conversation_repo: Conversation repository instance
    :param keyword_extractor: Keyword extractor dependency
    :return: Created memory with embedding
    :raises HTTPException: 404 if conversation not found, 400 if validation fails
    """
    # Validate conversation exists
    conversation = await conversation_repo.get_by_id(memory_data.conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {memory_data.conversation_id} not found",
        )

    # Calculate chunk_index (max existing + 1)
    existing_memories = await memory_repo.get_entity_memories(
        entity_id=memory_data.entity_id,
        limit=1000,  # Get all to find max chunk_index
    )
    conversation_memories = [m for m in existing_memories if m.conversation_id == memory_data.conversation_id]

    max_chunk_index = -1
    for mem in conversation_memories:
        if mem.memory_metadata and "chunk_index" in mem.memory_metadata:
            max_chunk_index = max(max_chunk_index, mem.memory_metadata["chunk_index"])

    chunk_index = max_chunk_index + 1

    # Auto-extract keywords if not provided
    keywords = memory_data.keywords
    if not keywords:
        keywords = await keyword_extractor.extract_keywords(memory_data.text, max_keywords=10)

    # Create embedding from text
    embedding_service = create_embedding_service()
    embedding = await embedding_service.embed(memory_data.text)

    # Create summary (first 200 chars)
    summary = memory_data.text[:200] + "..." if len(memory_data.text) > 200 else memory_data.text

    # Build metadata
    memory_metadata = {
        "type": "long_term",
        "chunk_index": chunk_index,
        "created_by": "admin",
        "extractor_used": "yake" if not memory_data.keywords else "manual",
    }

    # Create memory
    memory = AIMemory(
        entity_id=memory_data.entity_id,
        user_ids=memory_data.user_ids,
        conversation_id=memory_data.conversation_id,
        room_id=None,
        summary=summary,
        memory_content={"full_text": memory_data.text},
        keywords=keywords,
        importance_score=1.0,
        embedding=embedding,
        memory_metadata=memory_metadata,
    )

    created_memory = await memory_repo.create(memory)
    return created_memory


@router.patch("/{memory_id}", response_model=MemoryResponse)
async def update_memory(
    memory_id: int,
    memory_data: MemoryUpdate,
    current_admin: User = Depends(get_admin_user_with_csrf),
    memory_repo: IAIMemoryRepository = Depends(get_ai_memory_repository),
    keyword_extractor: IKeywordExtractor = Depends(get_keyword_extractor),
) -> MemoryResponse:
    """
    Update existing AI memory (Admin only).

    Keywords are re-extracted if summary changes.

    :param memory_id: Memory ID to update
    :param memory_data: Memory update data (partial)
    :param current_admin: Current authenticated admin
    :param memory_repo: Memory repository instance
    :param keyword_extractor: Keyword extractor dependency
    :return: Updated memory
    :raises HTTPException: 404 if memory not found
    """
    # Get existing memory
    memory = await memory_repo.get_by_id(memory_id)
    if not memory:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Memory {memory_id} not found")

    # Track if summary changed for keyword re-extraction
    summary_changed = False

    # Update fields if provided
    if memory_data.summary is not None:
        summary_changed = memory.summary != memory_data.summary
        memory.summary = memory_data.summary

    if memory_data.memory_content is not None:
        memory.memory_content = memory_data.memory_content

    if memory_data.importance_score is not None:
        memory.importance_score = memory_data.importance_score

    # Re-extract keywords if summary changed or explicitly provided
    if memory_data.keywords is not None:
        memory.keywords = memory_data.keywords
        # Update metadata
        if memory.memory_metadata:
            memory.memory_metadata["extractor_used"] = "manual"
    elif summary_changed:
        memory.keywords = await keyword_extractor.extract_keywords(memory.summary, max_keywords=10)
        # Update metadata
        if memory.memory_metadata:
            memory.memory_metadata["extractor_used"] = "yake"

    # Update version in metadata
    if memory.memory_metadata:
        memory.memory_metadata["version"] = memory.memory_metadata.get("version", 1) + 1

    updated_memory = await memory_repo.update(memory)
    return updated_memory


@router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(
    memory_id: int,
    current_admin: User = Depends(get_admin_user_with_csrf),
    memory_repo: IAIMemoryRepository = Depends(get_ai_memory_repository),
) -> None:
    """
    Delete AI memory (Admin only).

    Hard delete - memory is permanently removed.

    :param memory_id: Memory ID to delete
    :param current_admin: Current authenticated admin
    :param memory_repo: Memory repository instance
    :raises HTTPException: 404 if memory not found
    """
    deleted = await memory_repo.delete(memory_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Memory {memory_id} not found")


@router.get("/search", response_model=list[MemoryResponse])
async def search_memories(
    entity_id: int = Query(..., description="AI entity ID to search"),
    keywords: str = Query(..., description="Comma-separated keywords"),
    limit: int = Query(10, ge=1, le=50, description="Maximum results"),
    current_admin: User = Depends(get_admin_user_with_csrf),
    memory_repo: IAIMemoryRepository = Depends(get_ai_memory_repository),
) -> list[MemoryResponse]:
    """
    Search AI memories by keywords (Admin only).

    :param entity_id: AI entity ID to search
    :param keywords: Comma-separated keywords (e.g., "python,fastapi")
    :param limit: Maximum number of results (max 50)
    :param current_admin: Current authenticated admin
    :param memory_repo: Memory repository instance
    :return: List of matching memories ordered by importance
    """
    # Parse keywords
    keyword_list = [kw.strip().lower() for kw in keywords.split(",") if kw.strip()]

    if not keyword_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one keyword required",
        )

    # Search memories
    memories = await memory_repo.search_by_keywords(
        entity_id=entity_id,
        keywords=keyword_list,
        limit=limit,
    )

    return memories


@router.post(
    "/admin/ai-entities/{entity_id}/personality",
    response_model=PersonalityUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_personality(
    entity_id: int,
    request: PersonalityUploadRequest,
    current_admin: User = Depends(get_admin_user_with_csrf),
    personality_service: PersonalityMemoryService = Depends(get_personality_memory_service),
) -> PersonalityUploadResponse:
    """
    Upload personality knowledge base for AI entity (Admin only).

    Creates global personality memories (not user-specific).
    Text is chunked and embedded for semantic search.

    :param entity_id: AI entity ID
    :param request: Upload request with text, category, and metadata
    :param current_admin: Current authenticated admin
    :param personality_service: Personality memory service instance
    :return: Upload result with memory count and IDs
    :raises HTTPException: 400 if upload fails
    """
    if not request.text or not request.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text content cannot be empty",
        )

    try:
        memories = await personality_service.upload_personality(
            entity_id=entity_id,
            text=request.text,
            category=request.category,
            metadata=request.metadata or {},
        )

        return PersonalityUploadResponse(
            created_memories=len(memories),
            memory_ids=[m.id for m in memories],
            category=request.category,
            chunks=len(memories),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Personality upload failed: {str(e)}",
        )
