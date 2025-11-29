"""
Unit tests for MessageRepository.

Tests focus on message CRUD operations and pagination using SQLite in-memory database.
"""

import pytest

from app.repositories.message_repository import MessageRepository


@pytest.mark.unit
class TestMessageRepository:
    """Unit tests for MessageRepository operations."""

    async def test_create_room_message_success(self, db_session, user_factory, room_factory):
        """Test successful room message creation."""
        # Arrange
        repo = MessageRepository(db_session)
        user = await user_factory.create(db_session)
        room = await room_factory.create(db_session)

        # Act
        message = await repo.create_room_message(room_id=room.id, content="Hello room!", sender_user_id=user.id)

        # Assert
        assert message.id is not None
        assert message.sender_id == user.id
        assert message.sender_user_id == user.id
        assert message.room_id == room.id
        assert message.conversation_id is None
        assert message.content == "Hello room!"

    async def test_create_conversation_message_success(
        self, db_session, user_factory, room_factory, conversation_factory
    ):
        """Test successful conversation message creation within a room."""
        # Arrange
        repo = MessageRepository(db_session)
        user = await user_factory.create(db_session)
        room = await room_factory.create(db_session)
        # Create private conversation within the room
        conversation = await conversation_factory.create_private_conversation(db_session, room=room)

        # Act
        message = await repo.create_conversation_message(
            conversation_id=conversation.id, content="Private message", sender_user_id=user.id
        )

        # Assert
        assert message.id is not None
        assert message.sender_id == user.id
        assert message.conversation_id == conversation.id
        assert message.room_id is None  # Message routes via conversation, not room
        assert message.content == "Private message"

    async def test_get_by_id_success(self, db_session, message_factory, user_factory, room_factory):
        """Test successful message retrieval by ID."""
        # Arrange
        repo = MessageRepository(db_session)
        user = await user_factory.create(db_session)
        room = await room_factory.create(db_session)
        message = await message_factory.create_room_message(db_session, sender=user, room=room)

        # Act
        found_message = await repo.get_by_id(message.id)

        # Assert
        assert found_message is not None
        assert found_message.id == message.id
        assert found_message.content == message.content

    async def test_get_by_id_not_found(self, db_session):
        """Test message retrieval when ID does not exist."""
        # Arrange
        repo = MessageRepository(db_session)

        # Act
        found_message = await repo.get_by_id(99999)

        # Assert
        assert found_message is None

    async def test_get_room_messages_with_pagination(self, db_session, user_factory, room_factory, message_factory):
        """Test retrieving room messages with pagination."""
        # Arrange
        repo = MessageRepository(db_session)
        user = await user_factory.create(db_session)
        room = await room_factory.create(db_session)

        # Create 5 messages
        for i in range(5):
            await message_factory.create_room_message(db_session, sender=user, room=room, content=f"Message {i}")

        # Act
        messages, total = await repo.get_room_messages(room.id, page=1, page_size=3)

        # Assert
        assert len(messages) == 3
        assert total == 5

    async def test_get_room_messages_empty(self, db_session, room_factory):
        """Test retrieving room messages when room has no messages."""
        # Arrange
        repo = MessageRepository(db_session)
        room = await room_factory.create(db_session)

        # Act
        messages, total = await repo.get_room_messages(room.id)

        # Assert
        assert len(messages) == 0
        assert total == 0

    async def test_get_conversation_messages_with_pagination(
        self, db_session, user_factory, room_factory, conversation_factory, message_factory
    ):
        """Test retrieving conversation messages with pagination."""
        # Arrange
        repo = MessageRepository(db_session)
        user = await user_factory.create(db_session)
        room = await room_factory.create(db_session)
        # Create conversation within room
        conversation = await conversation_factory.create_private_conversation(db_session, room=room)

        # Create 4 messages
        for i in range(4):
            await message_factory.create_conversation_message(
                db_session, sender=user, conversation=conversation, content=f"Conv msg {i}"
            )

        # Act
        messages, total = await repo.get_conversation_messages(conversation.id, page=1, page_size=2)

        # Assert
        assert len(messages) == 2
        assert total == 4

    async def test_get_user_messages(self, db_session, user_factory, room_factory, message_factory):
        """Test retrieving messages sent by a specific user."""
        # Arrange
        repo = MessageRepository(db_session)
        user1 = await user_factory.create(db_session, username="user1")
        user2 = await user_factory.create(db_session, username="user2")
        room = await room_factory.create(db_session)

        # User1 sends 3 messages, user2 sends 1
        await message_factory.create_room_message(db_session, sender=user1, room=room)
        await message_factory.create_room_message(db_session, sender=user1, room=room)
        await message_factory.create_room_message(db_session, sender=user1, room=room)
        await message_factory.create_room_message(db_session, sender=user2, room=room)

        # Act
        user1_messages = await repo.get_user_messages(user1.id, limit=10)

        # Assert
        assert len(user1_messages) == 3
        assert all(msg.sender_id == user1.id for msg in user1_messages)

    async def test_get_latest_room_messages(self, db_session, user_factory, room_factory, message_factory):
        """Test retrieving latest messages from a room."""
        # Arrange
        repo = MessageRepository(db_session)
        user = await user_factory.create(db_session)
        room = await room_factory.create(db_session)

        # Create 15 messages
        for i in range(15):
            await message_factory.create_room_message(db_session, sender=user, room=room, content=f"Message {i}")

        # Act
        latest_messages = await repo.get_latest_room_messages(room.id, limit=5)

        # Assert
        assert len(latest_messages) == 5
        # Latest messages should be returned (descending order)

    async def test_update_message_success(self, db_session, user_factory, room_factory, message_factory):
        """Test successful message update."""
        # Arrange
        repo = MessageRepository(db_session)
        user = await user_factory.create(db_session)
        room = await room_factory.create(db_session)
        message = await message_factory.create_room_message(db_session, sender=user, room=room, content="Original")

        # Act
        message.content = "Updated"
        updated_message = await repo.update(message)

        # Assert
        assert updated_message.content == "Updated"

        # Verify in database
        found_message = await repo.get_by_id(message.id)
        assert found_message.content == "Updated"

    async def test_delete_message_success(self, db_session, user_factory, room_factory, message_factory):
        """Test successful message deletion."""
        # Arrange
        repo = MessageRepository(db_session)
        user = await user_factory.create(db_session)
        room = await room_factory.create(db_session)
        message = await message_factory.create_room_message(db_session, sender=user, room=room)
        message_id = message.id

        # Act
        result = await repo.delete(message_id)

        # Assert
        assert result is True

        # Verify message is deleted
        found_message = await repo.get_by_id(message_id)
        assert found_message is None

    async def test_exists_message_true(self, db_session, user_factory, room_factory, message_factory):
        """Test message existence check when message exists."""
        # Arrange
        repo = MessageRepository(db_session)
        user = await user_factory.create(db_session)
        room = await room_factory.create(db_session)
        message = await message_factory.create_room_message(db_session, sender=user, room=room)

        # Act
        exists = await repo.exists(message.id)

        # Assert
        assert exists is True

    async def test_exists_message_false(self, db_session):
        """Test message existence check when message does not exist."""
        # Arrange
        repo = MessageRepository(db_session)

        # Act
        exists = await repo.exists(99999)

        # Assert
        assert exists is False
