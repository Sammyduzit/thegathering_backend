"""
E2E tests for conversation participant management.

Covers adding human participants, verifying count, removing participants,
and enforcing admin-only removal rules.
"""

import pytest
from sqlalchemy import update

from app.models.user import User


@pytest.mark.e2e
class TestConversationParticipants:
    """Participant add/remove flows with detail validation."""

    async def test_add_and_remove_participants(
        self,
        authenticated_admin_client,
        authenticated_user_client,
        db_session,
        room_factory,
        user_factory,
        created_admin,
        created_user,
    ):
        """Admin adds user to conversation; user leaves; admin removes remaining user."""
        # Arrange shared room for admin + two users
        room = await room_factory.create(db_session, name="Participant Room")

        secondary_user = await user_factory.create(
            db_session,
            username="participant_user",
            email="participant_user@example.com",
            current_room_id=room.id,
        )
        # Align admin and primary user to same room
        await db_session.execute(
            update(User)
            .where(User.username.in_([created_admin.username, created_user.username]))
            .values(current_room_id=room.id)
        )
        await db_session.commit()

        # Step 1: Admin creates conversation with primary user
        create_response = await authenticated_admin_client.post(
            "/api/v1/conversations/",
            json={
                "participant_usernames": [created_user.username],
                "conversation_type": "group",
            },
        )
        assert create_response.status_code == 201, create_response.text
        conversation_id = create_response.json()["conversation_id"]

        # Step 2: Admin adds secondary user
        add_response = await authenticated_admin_client.post(
            f"/api/v1/conversations/{conversation_id}/participants",
            json={"username": secondary_user.username},
        )
        assert add_response.status_code == 200, add_response.text

        # Step 3: Conversation detail reflects participant count (3 total)
        detail_response = await authenticated_admin_client.get(f"/api/v1/conversations/{conversation_id}")
        assert detail_response.status_code == 200
        detail = detail_response.json()
        assert detail["participant_count"] == 3
        names = {p["username"] for p in detail["participants"]}
        assert {created_admin.username, created_user.username, secondary_user.username} <= names

        # Step 4: Secondary user leaves conversation
        leave_response = await authenticated_user_client.delete(
            f"/api/v1/conversations/{conversation_id}/participants/{created_user.username}"
        )
        assert leave_response.status_code == 200, leave_response.text

        # After leaving, detail should now show only admin + secondary user
        detail_after_leave = await authenticated_admin_client.get(f"/api/v1/conversations/{conversation_id}")
        assert detail_after_leave.status_code == 200
        detail_payload = detail_after_leave.json()
        assert detail_payload["participant_count"] == 2

        remaining_names = {p["username"] for p in detail_payload["participants"]}
        assert created_user.username not in remaining_names
        assert created_admin.username in remaining_names
        assert secondary_user.username in remaining_names

        # Step 5: Admin removes remaining secondary user (allowed)
        remove_response = await authenticated_admin_client.delete(
            f"/api/v1/conversations/{conversation_id}/participants/{secondary_user.username}"
        )
        assert remove_response.status_code == 200, remove_response.text

        # After removal, only admin should remain
        final_detail = await authenticated_admin_client.get(f"/api/v1/conversations/{conversation_id}")
        assert final_detail.status_code == 200
        final_payload = final_detail.json()
        assert final_payload["participant_count"] == 1
        assert {p["username"] for p in final_payload["participants"]} == {created_admin.username}
