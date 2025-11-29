"""
E2E tests for room workflows.

Scenarios covered:
- Admin can create rooms
- Regular users can list/join/leave/message rooms
- CSRF + cookie authentication via dedicated clients
"""

import pytest


@pytest.mark.e2e
class TestRoomWorkflows:
    """E2E tests for room workflows."""

    async def test_admin_create_room_success(self, authenticated_admin_client):
        """Admins can create rooms successfully."""
        response = await authenticated_admin_client.post(
            "/api/v1/rooms/",
            json={
                "name": "Test Room",
                "description": "A test room for E2E testing",
                "max_users": 10,
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test Room"
        assert data["max_users"] == 10
        assert "id" in data

    async def test_regular_user_cannot_create_room(self, authenticated_user_client):
        """Regular users must be forbidden from creating rooms."""
        response = await authenticated_user_client.post(
            "/api/v1/rooms/",
            json={
                "name": "Unauthorized Room",
                "description": "Should fail",
                "max_users": 10,
            },
        )

        assert response.status_code == 403

    async def test_get_all_rooms(self, authenticated_user_client, created_room):
        """Authenticated users can list rooms."""
        response = await authenticated_user_client.get("/api/v1/rooms/")

        assert response.status_code == 200
        rooms = response.json()
        assert isinstance(rooms, list)
        assert len(rooms) >= 1
        assert any(room["name"] == created_room.name for room in rooms)

    async def test_get_room_by_id(self, authenticated_user_client, created_room):
        """Authenticated users can retrieve individual room details."""
        response = await authenticated_user_client.get(f"/api/v1/rooms/{created_room.id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == created_room.id
        assert data["name"] == created_room.name

    async def test_get_nonexistent_room(self, authenticated_user_client):
        """Requesting a non-existent room returns 404."""
        response = await authenticated_user_client.get("/api/v1/rooms/99999")
        assert response.status_code == 404

    async def test_user_join_room(self, authenticated_user_client, created_room):
        """Authenticated users can join a room they are not already in."""
        response = await authenticated_user_client.post(f"/api/v1/rooms/{created_room.id}/join")
        assert response.status_code in (200, 204)

    async def test_user_leave_room(self, authenticated_user_client, created_room):
        """Users can leave a room they previously joined."""
        await authenticated_user_client.post(f"/api/v1/rooms/{created_room.id}/join")
        response = await authenticated_user_client.post(f"/api/v1/rooms/{created_room.id}/leave")
        assert response.status_code in (200, 204)

    async def test_send_message_to_room(self, authenticated_user_client, created_room):
        """Users can send messages to rooms they have joined."""
        await authenticated_user_client.post(f"/api/v1/rooms/{created_room.id}/join")
        response = await authenticated_user_client.post(
            f"/api/v1/rooms/{created_room.id}/messages",
            json={"content": "Hello everyone in the room!"},
        )

        assert response.status_code in (200, 201)
        data = response.json()
        assert data["content"] == "Hello everyone in the room!"
        assert data["room_id"] == created_room.id

    async def test_get_room_messages(self, authenticated_user_client, created_room):
        """Users can read messages from rooms they joined."""
        await authenticated_user_client.post(f"/api/v1/rooms/{created_room.id}/join")
        await authenticated_user_client.post(
            f"/api/v1/rooms/{created_room.id}/messages",
            json={"content": "Test message"},
        )

        response = await authenticated_user_client.get(f"/api/v1/rooms/{created_room.id}/messages")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict) or isinstance(data, list)

    async def test_send_message_without_joining_room(self, authenticated_user_client, created_room):
        """Users cannot send messages without joining the room first."""
        response = await authenticated_user_client.post(
            f"/api/v1/rooms/{created_room.id}/messages",
            json={"content": "This should fail"},
        )
        assert response.status_code in (400, 403)

    async def test_complete_room_workflow(self, authenticated_admin_client, authenticated_user_client):
        """End-to-end: admin creates room, user joins/messages/leaves."""
        create_response = await authenticated_admin_client.post(
            "/api/v1/rooms/",
            json={
                "name": "Workflow Room",
                "description": "Testing complete workflow",
                "max_users": 5,
            },
        )
        assert create_response.status_code == 201
        room_id = create_response.json()["id"]

        join_response = await authenticated_user_client.post(f"/api/v1/rooms/{room_id}/join")
        assert join_response.status_code in (200, 204)

        message_response = await authenticated_user_client.post(
            f"/api/v1/rooms/{room_id}/messages",
            json={"content": "Workflow message"},
        )
        assert message_response.status_code in (200, 201)

        get_messages_response = await authenticated_user_client.get(f"/api/v1/rooms/{room_id}/messages")
        assert get_messages_response.status_code == 200

        leave_response = await authenticated_user_client.post(f"/api/v1/rooms/{room_id}/leave")
        assert leave_response.status_code in (200, 204)

    async def test_unauthenticated_cannot_access_rooms(self, async_client, created_room):
        """Unauthenticated requests must be rejected."""
        response = await async_client.get(f"/api/v1/rooms/{created_room.id}")
        assert response.status_code == 401
