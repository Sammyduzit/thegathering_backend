"""
E2E tests for conversation detail flows.

Covers creation → detail endpoint → participant/metadata assertions.
"""

import pytest
from sqlalchemy import update

from app.models.ai_entity import AIEntity
from app.models.user import User


@pytest.mark.e2e
class TestConversationDetails:
    """Validate the dedicated conversation detail endpoint."""

    async def test_conversation_detail_for_private_chat(
        self,
        authenticated_user_client,
        db_session,
        user_factory,
        room_factory,
    ):
        """User creates a private conversation and retrieves its detail view."""
        # Arrange: shared room for both participants
        room = await room_factory.create(db_session, name="Detail Room")

        other_user = await user_factory.create(
            db_session,
            username="detail_target_user",
            email="detail_target_user@example.com",
            current_room_id=room.id,
        )
        # Update primary user (testuser) to same room
        await db_session.execute(update(User).where(User.username == "testuser").values(current_room_id=room.id))
        await db_session.commit()

        # Step 1: Create private conversation via primary user
        create_response = await authenticated_user_client.post(
            "/api/v1/conversations/",
            json={
                "participant_usernames": [other_user.username],
                "conversation_type": "private",
            },
        )
        assert create_response.status_code == 201, create_response.text
        conversation_id = create_response.json()["conversation_id"]

        # Step 2: Fetch conversation detail
        detail_response = await authenticated_user_client.get(f"/api/v1/conversations/{conversation_id}")
        assert detail_response.status_code == 200, detail_response.text
        detail = detail_response.json()

        # Step 3: Validate core fields
        assert detail["id"] == conversation_id
        assert detail["type"] == "private"
        assert detail["participant_count"] == 2
        assert detail["participants"], "Participant details missing"

        usernames = {p["username"] for p in detail["participants"]}
        assert "testuser" in usernames  # primary user
        assert other_user.username in usernames

        assert detail["latest_message"] is None

    async def test_conversation_detail_includes_ai_participant(
        self,
        authenticated_admin_client,
        db_session,
        user_factory,
        room_factory,
        created_admin,
        created_ai_entity,
    ):
        """Admin adds an AI participant; detail endpoint reflects AI metadata."""
        # Arrange: shared room for admin, human participant and AI
        room = await room_factory.create(db_session, name="AI Detail Room")

        second_user = await user_factory.create(
            db_session,
            username="ai_detail_user",
            email="ai_detail_user@example.com",
            current_room_id=room.id,
        )

        await db_session.execute(
            update(User).where(User.username == created_admin.username).values(current_room_id=room.id)
        )
        await db_session.execute(
            update(AIEntity).where(AIEntity.id == created_ai_entity.id).values(current_room_id=room.id)
        )
        await db_session.commit()

        # Step 1: Admin creates group conversation with secondary user
        create_response = await authenticated_admin_client.post(
            "/api/v1/conversations/",
            json={
                "participant_usernames": [second_user.username],
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

        # Step 3: Fetch detail as admin
        detail_response = await authenticated_admin_client.get(f"/api/v1/conversations/{conversation_id}")
        assert detail_response.status_code == 200, detail_response.text
        detail = detail_response.json()

        assert detail["type"] == "group"
        assert detail["participant_count"] == 3

        usernames = {p["username"] for p in detail["participants"]}
        assert created_admin.username in usernames
        assert second_user.username in usernames

        ai_entries = [p for p in detail["participants"] if p["is_ai"]]
        assert len(ai_entries) == 1
        assert ai_entries[0]["username"] == created_ai_entity.username
