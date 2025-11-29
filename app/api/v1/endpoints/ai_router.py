from fastapi import APIRouter, Depends, status

from app.core.auth_dependencies import get_admin_user_with_csrf, get_current_admin_user
from app.models.user import User
from app.schemas.ai_schemas import (
    AIAvailableResponse,
    AIEntityCreate,
    AIEntityResponse,
    AIEntityUpdate,
    AIGoodbyeResponse,
)
from app.schemas.common_schemas import MessageResponse
from app.services.ai.ai_entity_service import AIEntityService
from app.services.service_dependencies import get_ai_entity_service

router = APIRouter(prefix="/ai", tags=["ai"])


@router.get("/entities", response_model=list[AIEntityResponse])
async def get_all_ai_entities(
    current_admin: User = Depends(get_current_admin_user),
    ai_service: AIEntityService = Depends(get_ai_entity_service),
) -> list[AIEntityResponse]:
    """
    Get all AI entities (Admin only).
    :param current_admin: Current authenticated admin
    :param ai_service: AI entity service instance
    :return: List of all AI entities
    """
    return await ai_service.get_all_entities()


@router.get("/entities/available", response_model=list[AIEntityResponse])
async def get_available_ai_entities(
    current_admin: User = Depends(get_current_admin_user),
    ai_service: AIEntityService = Depends(get_ai_entity_service),
) -> list[AIEntityResponse]:
    """
    Get all available AI entities (online and not deleted) (Admin only).
    :param current_admin: Current authenticated admin
    :param ai_service: AI entity service instance
    :return: List of available AI entities
    """
    return await ai_service.get_available_entities()


@router.get("/entities/{entity_id}", response_model=AIEntityResponse)
async def get_ai_entity_by_id(
    entity_id: int,
    current_admin: User = Depends(get_current_admin_user),
    ai_service: AIEntityService = Depends(get_ai_entity_service),
) -> AIEntityResponse:
    """
    Get AI entity by ID (Admin only).
    :param entity_id: AI entity ID
    :param current_admin: Current authenticated admin
    :param ai_service: AI entity service instance
    :return: AI entity details
    """
    return await ai_service.get_entity_by_id(entity_id)


@router.post("/entities", response_model=AIEntityResponse, status_code=status.HTTP_201_CREATED)
async def create_ai_entity(
    entity_data: AIEntityCreate,
    current_admin: User = Depends(get_admin_user_with_csrf),
    ai_service: AIEntityService = Depends(get_ai_entity_service),
) -> AIEntityResponse:
    """
    Create new AI entity (Admin only).
    :param entity_data: AI entity creation data
    :param current_admin: Current authenticated admin
    :param ai_service: AI entity service instance
    :return: Created AI entity
    """
    return await ai_service.create_entity(
        username=entity_data.username,
        description=entity_data.description,
        system_prompt=entity_data.system_prompt,
        model_name=entity_data.model_name,
        temperature=entity_data.temperature,
        max_tokens=entity_data.max_tokens,
        room_response_strategy=entity_data.room_response_strategy,
        conversation_response_strategy=entity_data.conversation_response_strategy,
        response_probability=entity_data.response_probability,
        cooldown_seconds=entity_data.cooldown_seconds,
        config=entity_data.config,
    )


@router.patch("/entities/{entity_id}", response_model=AIEntityResponse)
async def update_ai_entity(
    entity_id: int,
    entity_data: AIEntityUpdate,
    current_admin: User = Depends(get_admin_user_with_csrf),
    ai_service: AIEntityService = Depends(get_ai_entity_service),
) -> AIEntityResponse:
    """
    Update AI entity (Admin only) - Partial update.

    Supports room assignment and status management:
    - Set status=OFFLINE to automatically remove AI from room
    - Set current_room_id to assign AI to room (requires AI to be ONLINE)
    - Set current_room_id=null to remove AI from current room
    - Omit current_room_id to keep current assignment

    :param entity_id: AI entity ID
    :param entity_data: AI entity update data (partial)
    :param current_admin: Current authenticated admin
    :param ai_service: AI entity service instance
    :return: Updated AI entity
    """
    return await ai_service.update_entity(
        entity_id=entity_id,
        username=entity_data.username,
        description=entity_data.description,
        system_prompt=entity_data.system_prompt,
        model_name=entity_data.model_name,
        temperature=entity_data.temperature,
        max_tokens=entity_data.max_tokens,
        room_response_strategy=entity_data.room_response_strategy,
        conversation_response_strategy=entity_data.conversation_response_strategy,
        response_probability=entity_data.response_probability,
        cooldown_seconds=entity_data.cooldown_seconds,
        config=entity_data.config,
        status=entity_data.status,
        current_room_id=entity_data.current_room_id,
    )


@router.delete("/entities/{entity_id}", response_model=MessageResponse)
async def delete_ai_entity(
    entity_id: int,
    current_admin: User = Depends(get_admin_user_with_csrf),
    ai_service: AIEntityService = Depends(get_ai_entity_service),
) -> MessageResponse:
    """
    Delete AI entity (Admin only).
    :param entity_id: AI entity ID
    :param current_admin: Current authenticated admin
    :param ai_service: AI entity service instance
    :return: Deletion confirmation
    """
    result = await ai_service.delete_entity(entity_id)
    return MessageResponse(message=result["message"])


@router.get("/rooms/{room_id}/available", response_model=list[AIAvailableResponse])
async def get_available_ai_in_room(
    room_id: int,
    current_admin: User = Depends(get_current_admin_user),
    ai_service: AIEntityService = Depends(get_ai_entity_service),
) -> list[AIAvailableResponse]:
    """
    Get available AI entities in a room (Admin only).
    :param room_id: Room ID
    :param current_admin: Current authenticated admin
    :param ai_service: AI entity service instance
    :return: List of available AI entities
    """
    return await ai_service.get_available_in_room(room_id)


@router.post("/entities/{entity_id}/goodbye", response_model=AIGoodbyeResponse)
async def initiate_ai_goodbye(
    entity_id: int,
    current_admin: User = Depends(get_admin_user_with_csrf),
    ai_service: AIEntityService = Depends(get_ai_entity_service),
) -> AIGoodbyeResponse:
    """
    Initiate graceful goodbye for AI entity (Admin only).

    The AI will:
    1. Say contextual farewell in all active conversations and leave them
    2. Say contextual farewell in assigned room (if any) and leave it
    3. Set response strategies to NO_RESPONSE to prevent further responses

    This is useful for gracefully retiring an AI or preparing it for reconfiguration.

    :param entity_id: AI entity ID
    :param current_admin: Current authenticated admin
    :param ai_service: AI entity service instance
    :return: Summary of goodbye actions (room/conversation farewells)
    """
    result = await ai_service.initiate_graceful_goodbye(entity_id)
    return AIGoodbyeResponse(
        message=result.get("message", "AI goodbye initiated"),
        ai_entity_id=entity_id,
        conversation_id=result.get("conversation_id"),
        room_id=result.get("room_id"),
    )
