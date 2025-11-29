"""
Domain exceptions for business logic validation.

These exceptions are framework-independent and represent business rule violations.
They are converted to HTTP responses at the API layer via exception handlers.

Best Practice (2025):
- Domain layer raises pure Python exceptions
- API layer converts to HTTPException via @app.exception_handler
- Structured error codes for client-side handling
"""


class DomainException(Exception):
    """Base exception for all domain/business logic errors."""

    def __init__(self, message: str, error_code: str):
        super().__init__(message)
        self.message = message
        self.error_code = error_code


# ============================================================================
# Authentication & Authorization Exceptions
# ============================================================================


class UnauthorizedException(DomainException):
    """User is not authenticated or credentials are invalid."""

    def __init__(self, message: str = "Authentication required"):
        super().__init__(message=message, error_code="UNAUTHORIZED")


class ForbiddenException(DomainException):
    """User does not have permission to perform this action."""

    def __init__(self, message: str = "Permission denied"):
        super().__init__(message=message, error_code="FORBIDDEN")


# ============================================================================
# Resource Not Found Exceptions
# ============================================================================


class NotFoundException(DomainException):
    """Base exception for resource not found errors."""

    def __init__(self, message: str, error_code: str = "NOT_FOUND"):
        super().__init__(message=message, error_code=error_code)


class RoomNotFoundException(NotFoundException):
    """Room with given ID does not exist."""

    def __init__(self, room_id: int | None = None):
        message = f"Room with ID {room_id} not found" if room_id else "Room not found"
        super().__init__(message=message, error_code="ROOM_NOT_FOUND")


class ConversationNotFoundException(NotFoundException):
    """Conversation with given ID does not exist."""

    def __init__(self, conversation_id: int | None = None):
        message = f"Conversation with ID {conversation_id} not found" if conversation_id else "Conversation not found"
        super().__init__(message=message, error_code="CONVERSATION_NOT_FOUND")


class UserNotFoundException(NotFoundException):
    """User with given identifier does not exist."""

    def __init__(self, identifier: str | int | None = None):
        message = f"User '{identifier}' not found" if identifier else "User not found"
        super().__init__(message=message, error_code="USER_NOT_FOUND")


class AIEntityNotFoundException(NotFoundException):
    """AI entity with given ID or name does not exist."""

    def __init__(self, identifier: str | int | None = None):
        message = f"AI entity '{identifier}' not found" if identifier else "AI entity not found"
        super().__init__(message=message, error_code="AI_ENTITY_NOT_FOUND")


# ============================================================================
# Validation Exceptions
# ============================================================================


class ValidationException(DomainException):
    """Base exception for validation errors."""

    def __init__(self, message: str, error_code: str = "VALIDATION_ERROR"):
        super().__init__(message=message, error_code=error_code)


class DuplicateResourceException(ValidationException):
    """Resource with given identifier already exists."""

    def __init__(self, resource_type: str, identifier: str):
        message = f"{resource_type} '{identifier}' already exists"
        super().__init__(message=message, error_code="DUPLICATE_RESOURCE")


class InvalidOperationException(ValidationException):
    """Operation is not valid in current state."""

    def __init__(self, message: str):
        super().__init__(message=message, error_code="INVALID_OPERATION")


# ============================================================================
# Room-Specific Exceptions
# ============================================================================


class RoomValidationException(ValidationException):
    """Room-specific validation error."""

    def __init__(self, message: str):
        super().__init__(message=message, error_code="ROOM_VALIDATION_ERROR")


class UserNotInRoomException(ForbiddenException):
    """User must be in a room to perform this action."""

    def __init__(self, message: str = "User must be in a room to perform this action"):
        super().__init__(message=message)
        self.error_code = "USER_NOT_IN_ROOM"


class NotRoomAdminException(ForbiddenException):
    """User must be room admin to perform this action."""

    def __init__(self, message: str = "Only room admin can perform this action"):
        super().__init__(message=message)
        self.error_code = "NOT_ROOM_ADMIN"


# ============================================================================
# Conversation-Specific Exceptions
# ============================================================================


class ConversationValidationException(ValidationException):
    """Conversation-specific validation error."""

    def __init__(self, message: str):
        super().__init__(message=message, error_code="CONVERSATION_VALIDATION_ERROR")


class NotConversationParticipantException(ForbiddenException):
    """User is not a participant in this conversation."""

    def __init__(self, message: str = "User is not a participant in this conversation"):
        super().__init__(message=message)
        self.error_code = "NOT_CONVERSATION_PARTICIPANT"


# ============================================================================
# AI Entity-Specific Exceptions
# ============================================================================


class AIEntityValidationException(ValidationException):
    """AI entity-specific validation error."""

    def __init__(self, message: str):
        super().__init__(message=message, error_code="AI_ENTITY_VALIDATION_ERROR")


class AIEntityOfflineException(InvalidOperationException):
    """AI entity must be online to perform this action."""

    def __init__(self, ai_name: str):
        super().__init__(message=f"AI entity '{ai_name}' is offline and cannot be invited")
        self.error_code = "AI_ENTITY_OFFLINE"
