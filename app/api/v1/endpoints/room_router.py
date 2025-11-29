from arq.connections import ArqRedis
from fastapi import APIRouter, BackgroundTasks, Body, Depends, status

from app.core.arq_pool import get_arq_pool
from app.core.auth_dependencies import (
    get_admin_user_with_csrf,
    get_authenticated_user_with_csrf,
    get_current_active_user,
    get_user_with_message_quota,
)
from app.core.background_tasks import async_bg_task_manager
from app.core.config import settings
from app.core.utils import calculate_pagination, enqueue_arq_job_safe
from app.models.user import User
from app.schemas.chat_schemas import MessageCreate, MessageResponse, PaginatedMessagesResponse
from app.schemas.common_schemas import CountResponse, HealthResponse, StatusUpdateResponse
from app.schemas.room_schemas import RoomCreate, RoomDeleteResponse, RoomResponse
from app.schemas.room_user_schemas import (
    RoomJoinResponse,
    RoomLeaveResponse,
    RoomParticipantsResponse,
    UserStatusUpdate,
)
from app.services.domain.background_service import BackgroundService
from app.services.domain.room_service import RoomService
from app.services.service_dependencies import get_background_service, get_room_service

router = APIRouter(prefix="/rooms", tags=["rooms"])


@router.get("/", response_model=list[RoomResponse])
async def get_all_rooms(
    current_user: User = Depends(get_current_active_user),
    room_service: RoomService = Depends(get_room_service),
) -> list[RoomResponse]:
    """
    Get all active rooms.
    :param current_user: Current authenticated user
    :param room_service: Service instance handling room logic
    :return: List of active rooms
    """
    return await room_service.get_all_rooms()


@router.post("/", response_model=RoomResponse, status_code=status.HTTP_201_CREATED)
async def create_room(
    room_data: RoomCreate,
    current_admin: User = Depends(get_admin_user_with_csrf),
    room_service: RoomService = Depends(get_room_service),
) -> RoomResponse:
    """
    Create a new room.
    :param room_data: Room creation data
    :param current_admin: Current authenticated admin
    :param room_service: Service instance handling room logic
    :return: Created room object
    """
    return await room_service.create_room(
        name=room_data.name,
        description=room_data.description,
        max_users=room_data.max_users,
        is_translation_enabled=room_data.is_translation_enabled,
    )


@router.put("/{room_id}", response_model=RoomResponse)
async def update_room(
    room_id: int,
    room_data: RoomCreate,
    current_admin: User = Depends(get_admin_user_with_csrf),
    room_service: RoomService = Depends(get_room_service),
) -> RoomResponse:
    """
    Update existing room data.
    :param room_id: ID of the room to update
    :param room_data: Room data to update
    :param current_admin: Current authenticated admin
    :param room_service: Service instance handling room logic
    :return: Updated room object
    """
    return await room_service.update_room(
        room_id=room_id,
        name=room_data.name,
        description=room_data.description,
        max_users=room_data.max_users,
        is_translation_enabled=room_data.is_translation_enabled,
    )


@router.delete("/{room_id}", response_model=RoomDeleteResponse)
async def delete_room(
    room_id: int,
    current_admin: User = Depends(get_admin_user_with_csrf),
    room_service: RoomService = Depends(get_room_service),
) -> RoomDeleteResponse:
    """
    Close room with cleanup, kick users and archive conversations.
    :param room_id: ID of room to delete
    :param current_admin: Current authenticated admin
    :param room_service: Service instance handling room logic
    :return: Cleanup summary with statistics
    """
    result = await room_service.delete_room(room_id)
    return RoomDeleteResponse(**result)


@router.get("/count", response_model=CountResponse)
async def get_room_count(
    current_user: User = Depends(get_current_active_user),
    room_service: RoomService = Depends(get_room_service),
) -> CountResponse:
    """
    Get count of active rooms.
    :param current_user: Current authenticated user
    :param room_service: Service instance handling room logic
    :return: Room count response
    """
    result = await room_service.get_room_count()
    return CountResponse(count=result["active_rooms"])


@router.get("/health", response_model=HealthResponse)
async def rooms_health() -> HealthResponse:
    """Health check"""
    return HealthResponse(status="rooms endpoint working")


@router.get("/{room_id}", response_model=RoomResponse)
async def get_room_by_id(
    room_id: int,
    current_user: User = Depends(get_current_active_user),
    room_service: RoomService = Depends(get_room_service),
) -> RoomResponse:
    """
    Get single room by ID.
    :param room_id: ID of room
    :param current_user: Current authenticated user
    :param room_service: Service instance handling room logic
    :return: Room object
    """
    return await room_service.get_room_by_id(room_id)


@router.post("/{room_id}/join", response_model=RoomJoinResponse)
async def join_room(
    room_id: int,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_authenticated_user_with_csrf),
    room_service: RoomService = Depends(get_room_service),
    background_service: BackgroundService = Depends(get_background_service),
) -> RoomJoinResponse:
    """
    User joins room.
    :param room_id: ID of room to join
    :param background_tasks: FastAPI background tasks
    :param current_user: Current authenticated user
    :param room_service: Service instance handling room logic
    :param background_service: Background service for async tasks
    :return: Join confirmation with room info
    """
    join_response = await room_service.join_room(current_user, room_id)

    # Schedule background tasks
    await async_bg_task_manager.add_async_task(
        background_tasks,
        background_service.log_user_activity_background,
        current_user.id,
        "room_joined",
        {"room_id": room_id, "room_name": join_response["room_name"]},
    )

    await async_bg_task_manager.add_async_task(
        background_tasks,
        background_service.notify_room_users_background,
        room_id,
        f"{current_user.username} joined the room",
        [current_user.id],  # Exclude the joining user
    )

    return join_response


@router.post("/{room_id}/leave", response_model=RoomLeaveResponse)
async def leave_room(
    room_id: int,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_authenticated_user_with_csrf),
    room_service: RoomService = Depends(get_room_service),
    background_service: BackgroundService = Depends(get_background_service),
) -> RoomLeaveResponse:
    """
    User leaves room.
    :param room_id: ID of room to leave
    :param background_tasks: FastAPI background tasks
    :param current_user: Current authenticated user
    :param room_service: Service instance handling room logic
    :param background_service: Background service for async tasks
    :return: Leave confirmation
    """
    leave_response = await room_service.leave_room(current_user, room_id)

    # Schedule background tasks
    await async_bg_task_manager.add_async_task(
        background_tasks,
        background_service.log_user_activity_background,
        current_user.id,
        "room_left",
        {"room_id": room_id},
    )

    await async_bg_task_manager.add_async_task(
        background_tasks,
        background_service.notify_room_users_background,
        room_id,
        f"{current_user.username} left the room",
        [current_user.id],  # Exclude the leaving user
    )

    return leave_response


@router.get("/{room_id}/participants", response_model=RoomParticipantsResponse)
async def get_room_participants(
    room_id: int,
    current_user: User = Depends(get_current_active_user),
    room_service: RoomService = Depends(get_room_service),
) -> RoomParticipantsResponse:
    """
    Get all participants (humans + AI) currently in a room.
    Returns unified list consistent with conversation participants structure.
    :param room_id: Room ID
    :param current_user: Current authenticated user
    :param room_service: Service instance handling room logic
    :return: Unified list of all room participants
    """
    return await room_service.get_room_participants(room_id)


@router.patch("/users/status", response_model=StatusUpdateResponse)
async def update_user_status(
    status_update: UserStatusUpdate,
    current_user: User = Depends(get_authenticated_user_with_csrf),
    room_service: RoomService = Depends(get_room_service),
) -> StatusUpdateResponse:
    """
    Update current user status.
    :param status_update: New status data
    :param current_user: Current authenticated user
    :param room_service: Service instance handling room logic
    :return: Status update confirmation
    """
    result = await room_service.update_user_status(current_user, status_update.status)
    return StatusUpdateResponse(message=result["message"], status=result["status"])


@router.post(
    "/{room_id}/messages",
    response_model=MessageResponse,
    status_code=status.HTTP_201_CREATED,
)
async def send_room_message(
    room_id: int,
    message_data: MessageCreate = Body(...),
    current_user: User = Depends(get_user_with_message_quota),
    room_service: RoomService = Depends(get_room_service),
    background_service: BackgroundService = Depends(get_background_service),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    arq_pool: ArqRedis | None = Depends(get_arq_pool),
) -> MessageResponse:
    """
    Send message to room, visible for every member.
    Background tasks handle translation and notifications.
    ARQ task triggers AI response if AI is present in room.
    :param room_id: Target room ID
    :param message_data: Message content
    :param background_tasks: FastAPI background tasks
    :param current_user: Current authenticated user
    :param room_service: Service instance handling room logic
    :param background_service: Background service for async tasks
    :param arq_pool: ARQ Redis pool for AI response jobs
    :return: Created message object
    """
    # Send message immediately
    message_response = await room_service.send_room_message(current_user, room_id, message_data.content)

    # Get room info for translation settings
    room = await room_service.get_room_by_id(room_id)

    # Trigger AI response check if AI is in room
    if room.has_ai and settings.is_ai_available and arq_pool:
        await enqueue_arq_job_safe(
            arq_pool,
            "check_and_generate_ai_response",
            {
                "message_id": message_response["id"],
                "room_id": room_id,
            },
            message_id=message_response["id"],
            room_id=room_id,
        )

    # Schedule background translation if enabled
    if room.is_translation_enabled:
        # Get room participants to determine target languages
        room_participants = await room_service.get_room_participants(room_id)
        target_languages = [
            participant.get("preferred_language", "en")
            for participant in room_participants["participants"]
            if not participant.get("is_ai")  # Exclude AI entities
            and participant.get("preferred_language")
            and participant.get("preferred_language") != "en"
        ]

        if target_languages:
            await async_bg_task_manager.add_async_task(
                background_tasks,
                background_service.process_message_translation_background,
                message_response,  # Message object
                list(set(target_languages)),  # Unique target languages
                room.is_translation_enabled,
            )

    # Schedule activity logging
    await async_bg_task_manager.add_async_task(
        background_tasks,
        background_service.log_user_activity_background,
        current_user.id,
        "message_sent",
        {"room_id": room_id, "message_length": len(message_data.content)},
    )

    return message_response


@router.get("/{room_id}/messages", response_model=PaginatedMessagesResponse)
async def get_room_messages(
    room_id: int,
    page: int = 1,
    page_size: int = 50,
    current_user: User = Depends(get_current_active_user),
    room_service: RoomService = Depends(get_room_service),
) -> PaginatedMessagesResponse:
    """
    Get room message history with pagination metadata.
    :param room_id: Room ID to get messages from
    :param page: Page number
    :param page_size: Messages per page
    :param current_user: Current authenticated User
    :param room_service: Service instance handling room logic
    :return: Paginated message response with metadata
    """
    messages, total_count = await room_service.get_room_messages(current_user, room_id, page, page_size)

    total_pages, has_more = calculate_pagination(total_count, page, page_size)

    return PaginatedMessagesResponse(
        messages=messages,
        total=total_count,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        has_more=has_more,
    )
