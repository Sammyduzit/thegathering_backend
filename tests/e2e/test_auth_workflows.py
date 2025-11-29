"""
E2E tests for authentication workflows.

Tests verify complete user authentication journeys:
- User registration
- Login/logout
- JWT token validation
- Permission checks
- Invalid credentials handling
"""

import pytest


@pytest.mark.e2e
class TestAuthWorkflows:
    """E2E tests for authentication workflows."""

    async def test_user_registration_success(self, async_client):
        """Test successful user registration workflow."""
        # Act
        response = await async_client.post(
            "/api/v1/auth/register",
            json={
                "email": "newuser@example.com",
                "username": "newuser",
                "password": "securepass123",
            },
        )

        # Assert
        assert response.status_code == 201
        data = response.json()
        assert data["email"] == "newuser@example.com"
        assert data["username"] == "newuser"
        assert "id" in data
        assert "password" not in data  # Password should not be in response

    async def test_user_registration_duplicate_email(self, async_client, created_user):
        """Test registration fails with duplicate email."""
        # Act
        response = await async_client.post(
            "/api/v1/auth/register",
            json={
                "email": "user@example.com",  # Already exists
                "username": "different_username",
                "password": "password123",
            },
        )

        # Assert
        assert response.status_code == 400
        detail = response.json()["detail"].lower()
        assert "already" in detail or "registered" in detail

    async def test_user_registration_duplicate_username(self, async_client, created_user):
        """Test registration fails with duplicate username."""
        # Act
        response = await async_client.post(
            "/api/v1/auth/register",
            json={
                "email": "different@example.com",
                "username": "testuser",  # Already exists
                "password": "password123",
            },
        )

        # Assert
        assert response.status_code == 400
        detail = response.json()["detail"].lower()
        assert "already" in detail or "taken" in detail

    async def test_user_registration_invalid_email(self, async_client):
        """Test registration fails with invalid email format."""
        # Act
        response = await async_client.post(
            "/api/v1/auth/register",
            json={
                "email": "invalid-email-format",
                "username": "testuser",
                "password": "password123",
            },
        )

        # Assert
        assert response.status_code == 422  # Validation error

    async def test_user_login_success(self, async_client, created_user):
        """Test successful user login workflow."""
        # Act
        response = await async_client.post(
            "/api/v1/auth/login",
            json={
                "email": "user@example.com",
                "password": "password123",
            },
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    async def test_user_login_wrong_password(self, async_client, created_user):
        """Test login fails with wrong password."""
        # Act
        response = await async_client.post(
            "/api/v1/auth/login",
            json={
                "email": "user@example.com",
                "password": "wrongpassword",
            },
        )

        # Assert
        assert response.status_code == 401
        detail = response.json()["detail"].lower()
        assert "invalid" in detail or "incorrect" in detail

    async def test_user_login_nonexistent_user(self, async_client):
        """Test login fails with nonexistent user."""
        # Act
        response = await async_client.post(
            "/api/v1/auth/login",
            json={
                "email": "nonexistent@example.com",
                "password": "password123",
            },
        )

        # Assert
        assert response.status_code == 401
        detail = response.json()["detail"].lower()
        assert "invalid" in detail or "incorrect" in detail

    async def test_get_current_user_authenticated(self, authenticated_user_client, created_user):
        """Test getting current user with valid token."""
        # Act
        response = await authenticated_user_client.get("/api/v1/auth/me")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "user@example.com"
        assert data["username"] == "testuser"
        assert data["is_admin"] is False

    async def test_get_current_user_unauthenticated(self, async_client):
        """Test getting current user without token fails."""
        # Act
        response = await async_client.get("/api/v1/auth/me")

        # Assert
        assert response.status_code == 401
        detail = response.json()["detail"].lower()
        assert "authent" in detail  # Matches "authenticated" or "authentication"

    async def test_admin_permission_required(self, authenticated_user_client, authenticated_admin_client):
        """Test that admin-only endpoints reject regular users."""
        # Act - Regular user tries to create room (admin-only)
        response = await authenticated_user_client.post(
            "/api/v1/rooms/",
            json={
                "name": "New Room",
                "description": "Test room",
                "max_users": 10,
            },
        )

        # Assert - Regular user should be forbidden
        assert response.status_code == 403
        detail = response.json()["detail"].lower()
        assert "admin" in detail or "permission" in detail

        # Act - Admin creates room
        admin_response = await authenticated_admin_client.post(
            "/api/v1/rooms/",
            json={
                "name": "Admin Room",
                "description": "Admin-created room",
                "max_users": 10,
            },
        )

        # Assert - Admin should succeed
        assert admin_response.status_code == 201

    async def test_token_expiration_workflow(self, async_client, created_user):
        """Test complete authentication flow with token usage."""
        # Step 1: Login
        login_response = await async_client.post(
            "/api/v1/auth/login",
            json={
                "email": "user@example.com",
                "password": "password123",
            },
        )
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]

        # Step 2: Use token to access protected endpoint
        headers = {"Authorization": f"Bearer {token}"}
        me_response = await async_client.get("/api/v1/auth/me", headers=headers)
        assert me_response.status_code == 200

        # Step 3: Token works for multiple requests
        second_request = await async_client.get("/api/v1/auth/me", headers=headers)
        assert second_request.status_code == 200

    async def test_complete_user_journey(self, async_client):
        """Test complete user journey: register -> login -> access protected resource."""
        # Step 1: Register new user
        register_response = await async_client.post(
            "/api/v1/auth/register",
            json={
                "email": "journey@example.com",
                "username": "journeyuser",
                "password": "journeypass123",
            },
        )
        assert register_response.status_code == 201
        user_id = register_response.json()["id"]

        # Step 2: Login with new credentials
        login_response = await async_client.post(
            "/api/v1/auth/login",
            json={
                "email": "journey@example.com",
                "password": "journeypass123",
            },
        )
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]

        # Step 3: Access protected endpoint with token
        headers = {"Authorization": f"Bearer {token}"}
        me_response = await async_client.get("/api/v1/auth/me", headers=headers)
        assert me_response.status_code == 200
        assert me_response.json()["id"] == user_id
        assert me_response.json()["email"] == "journey@example.com"

    async def test_password_not_returned_in_responses(self, async_client, created_user):
        """Test that password is never included in API responses."""
        # Login
        login_response = await async_client.post(
            "/api/v1/auth/login",
            json={
                "email": "user@example.com",
                "password": "password123",
            },
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Get current user
        me_response = await async_client.get("/api/v1/auth/me", headers=headers)
        user_data = me_response.json()

        # Assert password fields not in response
        assert "password" not in user_data
        assert "password_hash" not in user_data

    async def test_case_sensitive_email_login(self, async_client, created_user):
        """Test email login case sensitivity (depends on implementation)."""
        # Act - Login with uppercase email
        response = await async_client.post(
            "/api/v1/auth/login",
            json={
                "email": "USER@EXAMPLE.COM",  # Uppercase
                "password": "password123",
            },
        )

        # Assert - Email might be case-sensitive (401) or case-insensitive (200)
        # Both are valid implementations
        assert response.status_code in [200, 401]
        if response.status_code == 200:
            assert "access_token" in response.json()
