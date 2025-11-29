"""
Common response schemas used across multiple endpoints.

These generic models promote DRY principles and consistency across the API.
"""

from pydantic import BaseModel, Field


class MessageResponse(BaseModel):
    """
    Generic message response for simple operations.

    Used for endpoints that return a success/confirmation message.
    Examples: logout, delete operations, simple confirmations.
    """

    message: str = Field(description="Response message")


class CountResponse(BaseModel):
    """
    Generic count response.

    Used for endpoints that return a simple count of resources.
    """

    count: int = Field(ge=0, description="Count of resources")


class HealthResponse(BaseModel):
    """
    Generic health check response.

    Used for health/status check endpoints.
    """

    status: str = Field(description="Health status message")


class StatusUpdateResponse(BaseModel):
    """
    Generic status update confirmation.

    Used for endpoints that update status and return confirmation.
    """

    message: str = Field(description="Confirmation message")
    status: str = Field(description="Updated status value")
