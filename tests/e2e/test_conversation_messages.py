"""
E2E tests for conversation messaging flows.

Scenario: Admin + user + AI share a room, admin creates conversation,
user sends message, detail + history endpoints reflect new message.
"""

import pytest
from sqlalchemy import update

from app.models.ai_entity import AIEntity
from app.models.user import User


@pytest.mark.e2e
class TestConversationMessages:
    """Conversation messaging end-to-end tests."""

    async def test_user_sends_message_and_fetches_history(
        self,
        authenticated_admin_client,
        authenticated_user_client,
        db_session,
        room_factory,
        created_admin,
        created_user,
        created_ai_entity,
    ):
        """User posts a message; history and detail endpoints expose it."""
        # Arrange shared room for admin, user, and AI
        room = await room_factory.create(db_session, name="Conversation Room")

        await db_session.execute(
            update(User)
            .where(User.username.in_([created_admin.username, created_user.username]))
            .values(current_room_id=room.id)
        )
        await db_session.execute(
            update(AIEntity).where(AIEntity.id == created_ai_entity.id).values(current_room_id=room.id)
        )
        await db_session.commit()

        # Step 1: Admin creates a group conversation including testuser
        create_response = await authenticated_admin_client.post(
            "/api/v1/conversations/",
            json={
                "participant_usernames": [created_user.username],
                "conversation_type": "group",
            },
        )
        assert create_response.status_code == 201, create_response.text
        conversation_id = create_response.json()["conversation_id"]

        # Step 2: Admin invites AI participant
        invite_response = await authenticated_admin_client.post(
            f"/api/v1/conversations/{conversation_id}/participants",
            json={"username": created_ai_entity.username},
        )
        assert invite_response.status_code == 200, invite_response.text

        # Step 3: User sends a message
        message_content = "Hello from testuser"
        send_response = await authenticated_user_client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json={"content": message_content},
        )
        assert send_response.status_code in (200, 201), send_response.text
        sent_message = send_response.json()
        assert sent_message["content"] == message_content
        assert sent_message["conversation_id"] == conversation_id

        # Step 4: Fetch message history
        history_response = await authenticated_user_client.get(f"/api/v1/conversations/{conversation_id}/messages")
        assert history_response.status_code == 200, history_response.text
        history = history_response.json()
        assert isinstance(history, dict)
        assert "messages" in history
        assert history["messages"], "Expected at least one message in history"

        latest = history["messages"][0]
        assert latest["content"] == message_content
        assert latest["sender_username"] == created_user.username

        # Step 5: Conversation detail should reflect latest message info
        detail_response = await authenticated_user_client.get(f"/api/v1/conversations/{conversation_id}")
        assert detail_response.status_code == 200, detail_response.text
        detail = detail_response.json()

        assert detail["message_count"] >= 1
        assert detail["latest_message"] is not None
        assert detail["latest_message"]["content"] == message_content
        assert detail["latest_message"]["sender_username"] == created_user.username

        ai_entries = [p for p in detail["participants"] if p["is_ai"]]
        assert len(ai_entries) == 1
        assert ai_entries[0]["username"] == created_ai_entity.username
