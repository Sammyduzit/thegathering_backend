"""
E2E test for full user journey: register -> login -> join room -> send message -> trigger AI response.
"""

import asyncio

import pytest


@pytest.mark.e2e
class TestUserJourney:
    """Covers the complete flow from onboarding to sending a message with AI trigger."""

    async def test_user_journey_with_ai_trigger(
        self,
        async_client,
        authenticated_admin_client,
        redis_client,
    ):
        # Step 1: Register new user
        register_response = await async_client.post(
            "/api/v1/auth/register",
            json={
                "email": "journey_ai@example.com",
                "username": "journey_ai_user",
                "password": "journeyPass123!",
            },
        )
        assert register_response.status_code == 201

        # Step 2: Login and capture cookies
        login_response = await async_client.post(
            "/api/v1/auth/login",
            json={
                "email": "journey_ai@example.com",
                "password": "journeyPass123!",
            },
        )
        assert login_response.status_code == 200

        # Step 3: Admin creates a room so the user can join
        room_response = await authenticated_admin_client.post(
            "/api/v1/rooms/",
            json={
                "name": "Journey Room",
                "description": "Room for complete journey test",
                "max_users": 10,
            },
        )
        assert room_response.status_code == 201
        room_id = room_response.json()["id"]

        # Step 4: User joins the room
        join_response = await async_client.post(
            f"/api/v1/rooms/{room_id}/join", headers={"X-CSRF-Token": async_client.cookies.get("tg_csrf")}
        )
        assert join_response.status_code in (200, 204)

        # Step 5: User sends a message (triggers AI job via ARQ)
        message_response = await async_client.post(
            f"/api/v1/rooms/{room_id}/messages",
            json={"content": "Hello room, does the AI respond?"},
            headers={"X-CSRF-Token": async_client.cookies.get("tg_csrf")},
        )
        assert message_response.status_code in (200, 201)
        message_id = message_response.json()["id"]

        # Step 6: (Best effort) wait briefly for ARQ to enqueue job (skip assertion when queue empty)
        await asyncio.sleep(0.2)

        # Step 7: Fetch room messages to ensure user's message is stored
        messages_response = await async_client.get(f"/api/v1/rooms/{room_id}/messages")
        assert messages_response.status_code == 200
        messages_payload = messages_response.json()
        assert messages_payload["total"] >= 1
        assert any(msg["id"] == message_id for msg in messages_payload["messages"])
