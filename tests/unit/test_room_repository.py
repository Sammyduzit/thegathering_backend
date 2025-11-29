"""
Unit tests for RoomRepository.

Tests focus on CRUD operations and query methods using SQLite in-memory database.
All tests are isolated, fast, and deterministic.
"""

import pytest

from app.repositories.room_repository import RoomRepository


@pytest.mark.unit
class TestRoomRepository:
    """Unit tests for RoomRepository CRUD operations."""

    async def test_create_room_success(self, db_session, room_factory):
        """Test successful room creation."""
        # Arrange
        repo = RoomRepository(db_session)
        room = room_factory.build(name="Test Room", max_users=10)

        # Act
        created_room = await repo.create(room)

        # Assert
        assert created_room.id is not None
        assert created_room.name == "Test Room"
        assert created_room.max_users == 10
        assert created_room.is_active is True

    async def test_get_by_id_success(self, db_session, room_factory):
        """Test successful room retrieval by ID."""
        # Arrange
        repo = RoomRepository(db_session)
        room = await room_factory.create(db_session, name="My Room")

        # Act
        found_room = await repo.get_by_id(room.id)

        # Assert
        assert found_room is not None
        assert found_room.id == room.id
        assert found_room.name == "My Room"

    async def test_get_by_id_inactive_room(self, db_session, room_factory):
        """Test that inactive rooms are not returned by get_by_id."""
        # Arrange
        repo = RoomRepository(db_session)
        room = await room_factory.create(db_session, name="Inactive Room", is_active=False)

        # Act
        found_room = await repo.get_by_id(room.id)

        # Assert
        assert found_room is None

    async def test_get_by_name_success(self, db_session, room_factory):
        """Test successful room retrieval by name."""
        # Arrange
        repo = RoomRepository(db_session)
        await room_factory.create(db_session, name="Unique Room Name")

        # Act
        found_room = await repo.get_by_name("Unique Room Name")

        # Assert
        assert found_room is not None
        assert found_room.name == "Unique Room Name"

    async def test_get_by_name_not_found(self, db_session):
        """Test room retrieval when name does not exist."""
        # Arrange
        repo = RoomRepository(db_session)

        # Act
        found_room = await repo.get_by_name("Nonexistent Room")

        # Assert
        assert found_room is None

    async def test_get_active_rooms(self, db_session, room_factory):
        """Test retrieving only active rooms."""
        # Arrange
        repo = RoomRepository(db_session)
        await room_factory.create(db_session, name="Active 1", is_active=True)
        await room_factory.create(db_session, name="Active 2", is_active=True)
        await room_factory.create(db_session, name="Inactive", is_active=False)

        # Act
        active_rooms = await repo.get_active_rooms()

        # Assert
        assert len(active_rooms) == 2
        assert all(room.is_active for room in active_rooms)

    async def test_get_all_with_pagination(self, db_session, room_factory):
        """Test retrieving all rooms with pagination."""
        # Arrange
        repo = RoomRepository(db_session)
        for i in range(5):
            await room_factory.create(db_session, name=f"Room {i}")

        # Act
        rooms = await repo.get_all(limit=3, offset=0)

        # Assert
        assert len(rooms) == 3

    async def test_get_user_count(self, db_session, room_factory, user_factory):
        """Test getting user count in a room."""
        # Arrange
        repo = RoomRepository(db_session)
        room = await room_factory.create(db_session, name="Populated Room")
        await user_factory.create(db_session, current_room_id=room.id)
        await user_factory.create(db_session, current_room_id=room.id)

        # Act
        count = await repo.get_user_count(room.id)

        # Assert
        assert count == 2

    async def test_update_room_success(self, db_session, room_factory):
        """Test successful room update."""
        # Arrange
        repo = RoomRepository(db_session)
        room = await room_factory.create(db_session, name="Old Name", max_users=10)

        # Act
        room.name = "New Name"
        room.max_users = 20
        updated_room = await repo.update(room)

        # Assert
        assert updated_room.name == "New Name"
        assert updated_room.max_users == 20

        # Verify in database
        found_room = await repo.get_by_id(room.id)
        assert found_room.name == "New Name"
        assert found_room.max_users == 20

    async def test_soft_delete_room_success(self, db_session, room_factory):
        """Test successful room soft deletion."""
        # Arrange
        repo = RoomRepository(db_session)
        room = await room_factory.create(db_session, name="Delete Me")
        room_id = room.id

        # Act
        result = await repo.soft_delete(room_id)

        # Assert
        assert result is True

        # Verify room is soft-deleted (not returned by get_by_id)
        found_room = await repo.get_by_id(room_id)
        assert found_room is None

    async def test_name_exists_true(self, db_session, room_factory):
        """Test room name existence check when name exists."""
        # Arrange
        repo = RoomRepository(db_session)
        await room_factory.create(db_session, name="Existing Room")

        # Act
        exists = await repo.name_exists("Existing Room")

        # Assert
        assert exists is True

    async def test_name_exists_false(self, db_session):
        """Test room name existence check when name does not exist."""
        # Arrange
        repo = RoomRepository(db_session)

        # Act
        exists = await repo.name_exists("Nonexistent Room")

        # Assert
        assert exists is False

    async def test_name_exists_with_exclusion(self, db_session, room_factory):
        """Test room name existence check with exclusion."""
        # Arrange
        repo = RoomRepository(db_session)
        room = await room_factory.create(db_session, name="My Room")

        # Act
        exists = await repo.name_exists("My Room", exclude_room_id=room.id)

        # Assert
        assert exists is False  # Excluded, so should not exist
