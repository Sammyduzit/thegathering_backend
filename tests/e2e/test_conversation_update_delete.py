"""
E2E tests for conversation update and delete operations.

Tests REST-compliant PATCH and DELETE operations on conversations.
"""

import pytest
from httpx import AsyncClient
from sqlalchemy import update

from app.models.user import User


@pytest.mark.e2e
class TestConversationUpdateDelete:
    """Test conversation update and delete (archive) operations."""

    async def test_archive_conversation(
        self, authenticated_admin_client: AsyncClient, db_session, room_factory, user_factory, created_admin
    ):
        """Test archiving a conversation (soft delete via PATCH)."""
        # Create room and second user
        room = await room_factory.create(db_session, name="Archive Test Room")
        other_user = await user_factory.create(
            db_session, username="other_user", email="other@example.com", current_room_id=room.id
        )

        # Align admin to same room
        await db_session.execute(update(User).where(User.id == created_admin.id).values(current_room_id=room.id))
        await db_session.commit()

        # Create conversation
        create_response = await authenticated_admin_client.post(
            "/api/v1/conversations/",
            json={"participant_usernames": [other_user.username], "conversation_type": "private"},
        )
        assert create_response.status_code == 201
        conversation_id = create_response.json()["conversation_id"]

        # Archive conversation
        response = await authenticated_admin_client.patch(
            f"/api/v1/conversations/{conversation_id}",
            json={"is_active": False},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Conversation archived successfully"
        assert data["conversation_id"] == conversation_id
        assert data["is_active"] is False

    async def test_delete_conversation_soft_delete(
        self, authenticated_admin_client: AsyncClient, db_session, room_factory, user_factory, created_admin
    ):
        """Test deleting conversation (soft delete via DELETE endpoint)."""
        # Setup
        room = await room_factory.create(db_session, name="Delete Test Room")
        other_user = await user_factory.create(
            db_session, username="delete_user", email="delete@example.com", current_room_id=room.id
        )
        await db_session.execute(update(User).where(User.id == created_admin.id).values(current_room_id=room.id))
        await db_session.commit()

        # Create conversation
        create_response = await authenticated_admin_client.post(
            "/api/v1/conversations/",
            json={"participant_usernames": [other_user.username], "conversation_type": "private"},
        )
        conversation_id = create_response.json()["conversation_id"]

        # Delete conversation (soft delete)
        response = await authenticated_admin_client.delete(
            f"/api/v1/conversations/{conversation_id}",
        )

        assert response.status_code == 204

    async def test_update_nonexistent_conversation(self, authenticated_admin_client: AsyncClient):
        """Test updating non-existent conversation returns 404."""
        response = await authenticated_admin_client.patch(
            "/api/v1/conversations/999999",
            json={"is_active": False},
        )

        assert response.status_code == 404
