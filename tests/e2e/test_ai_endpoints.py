"""
E2E tests for AI entity admin endpoints.

Focus on the critical admin-only flows that remain in the API:
- Admin lifecycle (create → list → detail → patch → delete)
- Non-admin authorization checks
- Listing available AIs for a room
"""

import pytest


@pytest.mark.e2e
class TestAIEntityAdminEndpoints:
    """E2E tests covering AI entity management for admins."""

    async def test_admin_can_manage_ai_entity_lifecycle(self, authenticated_admin_client):
        """Full CRUD lifecycle for an AI entity via admin endpoints."""
        # Create
        create_response = await authenticated_admin_client.post(
            "/api/v1/ai/entities",
            json={
                "username": "assistant_gpt4",
                "system_prompt": "You are a helpful AI assistant.",
                "model_name": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 2000,
            },
        )
        assert create_response.status_code == 201, create_response.text
        created = create_response.json()
        ai_id = created["id"]
        assert created["username"] == "assistant_gpt4"
        assert created["status"] == "offline"

        # List (admin)
        list_response = await authenticated_admin_client.get("/api/v1/ai/entities")
        assert list_response.status_code == 200, list_response.text
        entities = list_response.json()
        assert any(entity["id"] == ai_id for entity in entities)

        # Detail (admin)
        detail_response = await authenticated_admin_client.get(f"/api/v1/ai/entities/{ai_id}")
        assert detail_response.status_code == 200, detail_response.text
        assert detail_response.json()["username"] == "assistant_gpt4"

        # Patch (username + prompt)
        patch_response = await authenticated_admin_client.patch(
            f"/api/v1/ai/entities/{ai_id}",
            json={
                "username": "updated_assistant",
                "system_prompt": "Updated prompt",
                "current_room_id": None,
            },
        )
        assert patch_response.status_code == 200, patch_response.text
        patched = patch_response.json()
        assert patched["username"] == "updated_assistant"
        assert patched["system_prompt"] == "Updated prompt"

        # Delete
        delete_response = await authenticated_admin_client.delete(f"/api/v1/ai/entities/{ai_id}")
        assert delete_response.status_code == 200, delete_response.text
        assert "deleted" in delete_response.json()["message"].lower()

    async def test_non_admin_is_forbidden(self, authenticated_user_client):
        """Regular users should not access admin-only AI endpoints."""
        endpoints = [
            ("POST", "/api/v1/ai/entities", {"username": "unauthorized", "system_prompt": "x"}),
            ("GET", "/api/v1/ai/entities", None),
            ("GET", "/api/v1/ai/entities/1", None),
            ("PATCH", "/api/v1/ai/entities/1", {"username": "nope"}),
            ("DELETE", "/api/v1/ai/entities/1", None),
            ("GET", "/api/v1/ai/rooms/1/available", None),
        ]

        for method, path, payload in endpoints:
            response = await authenticated_user_client.request(method, path, json=payload)
            assert response.status_code == 403, f"{method} {path} should be forbidden"

    async def test_get_available_ai_in_room_returns_list(
        self,
        authenticated_admin_client,
        created_room,
    ):
        """Admin can query available AIs in a room (empty list when none configured)."""
        response = await authenticated_admin_client.get(f"/api/v1/ai/rooms/{created_room.id}/available")
        assert response.status_code == 200, response.text
        assert isinstance(response.json(), list)
