"""
Unit tests for UserRepository.

Tests focus on CRUD operations and query methods using SQLite in-memory database.
All tests are isolated, fast, and deterministic.
"""

import pytest

from app.repositories.user_repository import UserRepository


@pytest.mark.unit
class TestUserRepository:
    """Unit tests for UserRepository CRUD operations."""

    async def test_create_user_success(self, db_session, user_factory):
        """Test successful user creation."""
        # Arrange
        repo = UserRepository(db_session)
        user = user_factory.build(email="test@example.com", username="testuser")

        # Act
        created_user = await repo.create(user)

        # Assert
        assert created_user.id is not None
        assert created_user.email == "test@example.com"
        assert created_user.username == "testuser"
        assert created_user.is_active is True

    async def test_get_by_id_success(self, db_session, user_factory):
        """Test successful user retrieval by ID."""
        # Arrange
        repo = UserRepository(db_session)
        user = await user_factory.create(db_session, email="user@example.com")

        # Act
        found_user = await repo.get_by_id(user.id)

        # Assert
        assert found_user is not None
        assert found_user.id == user.id
        assert found_user.email == "user@example.com"

    async def test_get_by_id_not_found(self, db_session):
        """Test user retrieval when ID does not exist."""
        # Arrange
        repo = UserRepository(db_session)

        # Act
        found_user = await repo.get_by_id(99999)

        # Assert
        assert found_user is None

    async def test_get_by_email_success(self, db_session, user_factory):
        """Test successful user retrieval by email."""
        # Arrange
        repo = UserRepository(db_session)
        await user_factory.create(db_session, email="findme@example.com")

        # Act
        found_user = await repo.get_by_email("findme@example.com")

        # Assert
        assert found_user is not None
        assert found_user.email == "findme@example.com"

    async def test_get_by_email_not_found(self, db_session):
        """Test user retrieval when email does not exist."""
        # Arrange
        repo = UserRepository(db_session)

        # Act
        found_user = await repo.get_by_email("nonexistent@example.com")

        # Assert
        assert found_user is None

    async def test_get_by_username_success(self, db_session, user_factory):
        """Test successful user retrieval by username."""
        # Arrange
        repo = UserRepository(db_session)
        await user_factory.create(db_session, username="findme")

        # Act
        found_user = await repo.get_by_username("findme")

        # Assert
        assert found_user is not None
        assert found_user.username == "findme"

    async def test_get_by_username_not_found(self, db_session):
        """Test user retrieval when username does not exist."""
        # Arrange
        repo = UserRepository(db_session)

        # Act
        found_user = await repo.get_by_username("nonexistent")

        # Assert
        assert found_user is None

    async def test_get_all_with_pagination(self, db_session, user_factory):
        """Test retrieving all users with pagination."""
        # Arrange
        repo = UserRepository(db_session)
        for i in range(5):
            await user_factory.create(db_session, email=f"user{i}@example.com", username=f"user{i}")

        # Act
        users = await repo.get_all(limit=3, offset=0)

        # Assert
        assert len(users) == 3

    async def test_get_active_users(self, db_session, user_factory):
        """Test retrieving only active users."""
        # Arrange
        repo = UserRepository(db_session)
        await user_factory.create(db_session, email="active1@example.com", is_active=True)
        await user_factory.create(db_session, email="active2@example.com", is_active=True)
        await user_factory.create(db_session, email="inactive@example.com", is_active=False)

        # Act
        active_users = await repo.get_active_users()

        # Assert
        assert len(active_users) == 2
        assert all(user.is_active for user in active_users)

    async def test_get_users_in_room(self, db_session, user_factory, room_factory):
        """Test retrieving users in a specific room."""
        # Arrange
        repo = UserRepository(db_session)
        room = await room_factory.create(db_session)
        await user_factory.create(db_session, email="inroom1@example.com", current_room_id=room.id)
        await user_factory.create(db_session, email="inroom2@example.com", current_room_id=room.id)
        await user_factory.create(db_session, email="notinroom@example.com", current_room_id=None)

        # Act
        users_in_room = await repo.get_users_in_room(room.id)

        # Assert
        assert len(users_in_room) == 2
        assert all(user.current_room_id == room.id for user in users_in_room)

    async def test_update_user_success(self, db_session, user_factory):
        """Test successful user update."""
        # Arrange
        repo = UserRepository(db_session)
        user = await user_factory.create(db_session, email="before@example.com")

        # Act
        user.email = "after@example.com"
        updated_user = await repo.update(user)

        # Assert
        assert updated_user.email == "after@example.com"

        # Verify in database
        found_user = await repo.get_by_id(user.id)
        assert found_user.email == "after@example.com"

    async def test_delete_user_success(self, db_session, user_factory):
        """Test successful user deletion (soft delete - sets is_active=False)."""
        # Arrange
        repo = UserRepository(db_session)
        user = await user_factory.create(db_session, email="delete@example.com")
        user_id = user.id

        # Act
        result = await repo.delete(user_id)

        # Assert
        assert result is True

        # Verify user is soft-deleted (still exists but inactive)
        found_user = await repo.get_by_id(user_id)
        assert found_user is not None
        assert found_user.is_active is False

    async def test_email_exists_true(self, db_session, user_factory):
        """Test email existence check when email exists."""
        # Arrange
        repo = UserRepository(db_session)
        await user_factory.create(db_session, email="exists@example.com")

        # Act
        exists = await repo.email_exists("exists@example.com")

        # Assert
        assert exists is True

    async def test_email_exists_false(self, db_session):
        """Test email existence check when email does not exist."""
        # Arrange
        repo = UserRepository(db_session)

        # Act
        exists = await repo.email_exists("nonexistent@example.com")

        # Assert
        assert exists is False

    async def test_username_exists_true(self, db_session, user_factory):
        """Test username existence check when username exists."""
        # Arrange
        repo = UserRepository(db_session)
        await user_factory.create(db_session, username="existinguser")

        # Act
        exists = await repo.username_exists("existinguser")

        # Assert
        assert exists is True

    async def test_username_exists_false(self, db_session):
        """Test username existence check when username does not exist."""
        # Arrange
        repo = UserRepository(db_session)

        # Act
        exists = await repo.username_exists("nonexistentuser")

        # Assert
        assert exists is False
