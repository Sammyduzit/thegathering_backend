"""
E2E tests for token rotation and reuse detection.

Tests OWASP 2025 security features:
- Refresh token rotation
- Token reuse detection
- Token family revocation
"""

import pytest
from httpx import AsyncClient

from app.core.jwt_utils import verify_token


@pytest.mark.e2e
class TestTokenRotation:
    """Test refresh token rotation security features."""

    async def test_refresh_token_rotation(self, async_client: AsyncClient, sample_user_data):
        """Test that refresh endpoint issues new refresh token."""
        # Register and login
        await async_client.post("/api/v1/auth/register", json=sample_user_data)
        login_response = await async_client.post(
            "/api/v1/auth/login",
            json={"email": sample_user_data["email"], "password": sample_user_data["password"]},
        )
        assert login_response.status_code == 200

        # Extract first refresh token
        first_refresh_token = login_response.cookies.get("tg_refresh")
        assert first_refresh_token is not None

        # Decode to get JTI and family ID
        first_payload = verify_token(first_refresh_token)
        first_jti = first_payload["jti"]
        family_id = first_payload.get("family_id")
        assert family_id is not None, "Token should have family_id"

        # Refresh access token
        refresh_response = await async_client.post("/api/v1/auth/refresh")
        assert refresh_response.status_code == 200

        # Extract second refresh token
        second_refresh_token = refresh_response.cookies.get("tg_refresh")
        assert second_refresh_token is not None

        # Decode second token
        second_payload = verify_token(second_refresh_token)
        second_jti = second_payload["jti"]
        second_family_id = second_payload.get("family_id")

        # Assertions
        assert first_jti != second_jti, "Refresh token should rotate (different JTI)"
        assert family_id == second_family_id, "Family ID should remain the same"
        assert first_refresh_token != second_refresh_token, "Refresh token should change"

    async def test_old_refresh_token_becomes_invalid(self, async_client: AsyncClient, sample_user_data):
        """Test that old refresh token cannot be reused after rotation."""
        # Register and login
        await async_client.post("/api/v1/auth/register", json=sample_user_data)
        login_response = await async_client.post(
            "/api/v1/auth/login",
            json={"email": sample_user_data["email"], "password": sample_user_data["password"]},
        )

        # Save old refresh token
        old_refresh_token = login_response.cookies.get("tg_refresh")

        # Refresh once (this invalidates old token)
        refresh_response = await async_client.post("/api/v1/auth/refresh")
        assert refresh_response.status_code == 200

        # Manually set old refresh token cookie
        async_client.cookies.set("tg_refresh", old_refresh_token)

        # Try to use old refresh token again
        reuse_response = await async_client.post("/api/v1/auth/refresh")

        # Should fail with 401
        assert reuse_response.status_code == 401
        assert "revoked" in reuse_response.json()["detail"].lower()

    async def test_token_reuse_detection_revokes_family(self, async_client: AsyncClient, sample_user_data):
        """Test that token reuse detection revokes entire token family."""
        # Register and login
        await async_client.post("/api/v1/auth/register", json=sample_user_data)
        login_response = await async_client.post(
            "/api/v1/auth/login",
            json={"email": sample_user_data["email"], "password": sample_user_data["password"]},
        )

        # Save first token
        _ = login_response.cookies.get("tg_refresh")

        # Refresh to get second token
        refresh1 = await async_client.post("/api/v1/auth/refresh")
        second_token = refresh1.cookies.get("tg_refresh")

        # Refresh again to get third token
        refresh2 = await async_client.post("/api/v1/auth/refresh")
        third_token = refresh2.cookies.get("tg_refresh")
        assert refresh2.status_code == 200

        # Now try to reuse the second token (already used)
        async_client.cookies.set("tg_refresh", second_token)
        reuse_response = await async_client.post("/api/v1/auth/refresh")

        # Should detect reuse and revoke family
        assert reuse_response.status_code == 401
        assert "reuse detected" in reuse_response.json()["detail"].lower()

        # Now the third token (latest valid token) should also be invalid
        async_client.cookies.set("tg_refresh", third_token)
        third_token_response = await async_client.post("/api/v1/auth/refresh")

        # Should be revoked
        assert third_token_response.status_code == 401

    async def test_logout_revokes_token_family(self, async_client: AsyncClient, sample_user_data):
        """Test that logout revokes entire token family."""
        # Register and login
        await async_client.post("/api/v1/auth/register", json=sample_user_data)
        login_response = await async_client.post(
            "/api/v1/auth/login",
            json={"email": sample_user_data["email"], "password": sample_user_data["password"]},
        )

        # Refresh a few times to create token family
        await async_client.post("/api/v1/auth/refresh")
        refresh2 = await async_client.post("/api/v1/auth/refresh")
        latest_token = refresh2.cookies.get("tg_refresh")

        # Get CSRF token for logout
        csrf_token = login_response.cookies.get("tg_csrf")

        # Logout
        logout_response = await async_client.post("/api/v1/auth/logout", headers={"X-CSRF-Token": csrf_token})
        assert logout_response.status_code == 200

        # Try to use latest token after logout
        async_client.cookies.set("tg_refresh", latest_token)
        refresh_after_logout = await async_client.post("/api/v1/auth/refresh")

        # Should be revoked
        assert refresh_after_logout.status_code == 401

    async def test_multiple_refresh_cycles(self, async_client: AsyncClient, sample_user_data):
        """Test multiple refresh cycles maintain family continuity."""
        # Register and login
        await async_client.post("/api/v1/auth/register", json=sample_user_data)
        await async_client.post(
            "/api/v1/auth/login",
            json={"email": sample_user_data["email"], "password": sample_user_data["password"]},
        )

        # Perform 5 refresh cycles
        family_ids = []
        for i in range(5):
            refresh_response = await async_client.post("/api/v1/auth/refresh")
            assert refresh_response.status_code == 200, f"Refresh {i + 1} failed"

            refresh_token = refresh_response.cookies.get("tg_refresh")
            payload = verify_token(refresh_token)
            family_ids.append(payload.get("family_id"))

        # All should have same family_id
        assert len(set(family_ids)) == 1, "All tokens should belong to same family"
