"""
Unit tests for ConversationRepository.

Tests focus on conversation CRUD operations, participant management,
and conversation queries using SQLite in-memory database.
"""

import pytest

from app.models.conversation import ConversationType
from app.repositories.conversation_repository import ConversationRepository


@pytest.mark.unit
class TestConversationRepository:
    """Unit tests for ConversationRepository operations."""

    async def test_create_private_conversation_success(self, db_session, user_factory, room_factory):
        """Test successful private conversation creation."""
        # Arrange
        repo = ConversationRepository(db_session)
        room = await room_factory.create(db_session)
        user1 = await user_factory.create(db_session, username="user1")
        user2 = await user_factory.create(db_session, username="user2")

        # Act
        conversation = await repo.create_private_conversation(
            room_id=room.id,
            user_ids=[user1.id, user2.id],
            ai_ids=[],
        )

        # Assert
        assert conversation.id is not None
        assert conversation.room_id == room.id
        assert conversation.conversation_type == ConversationType.PRIVATE
        assert conversation.max_participants == 2
        assert conversation.is_active is True

    async def test_create_private_conversation_invalid_participant_count(self, db_session, user_factory, room_factory):
        """Test private conversation creation with wrong number of participants."""
        # Arrange
        repo = ConversationRepository(db_session)
        room = await room_factory.create(db_session)
        user = await user_factory.create(db_session)

        # Act & Assert
        with pytest.raises(ValueError, match="exactly 2 participants"):
            await repo.create_private_conversation(
                room_id=room.id,
                user_ids=[user.id],
                ai_ids=[],
            )

    async def test_create_group_conversation_success(self, db_session, user_factory, room_factory):
        """Test successful group conversation creation."""
        # Arrange
        repo = ConversationRepository(db_session)
        room = await room_factory.create(db_session)
        user1 = await user_factory.create(db_session, username="user1")
        user2 = await user_factory.create(db_session, username="user2")
        user3 = await user_factory.create(db_session, username="user3")

        # Act
        conversation = await repo.create_group_conversation(
            room_id=room.id,
            user_ids=[user1.id, user2.id, user3.id],
            ai_ids=[],
        )

        # Assert
        assert conversation.id is not None
        assert conversation.room_id == room.id
        assert conversation.conversation_type == ConversationType.GROUP
        assert conversation.max_participants is None
        assert conversation.is_active is True

    async def test_create_group_conversation_invalid_participant_count(self, db_session, user_factory, room_factory):
        """Test group conversation creation with too few participants."""
        # Arrange
        repo = ConversationRepository(db_session)
        room = await room_factory.create(db_session)
        user = await user_factory.create(db_session)

        # Act & Assert
        with pytest.raises(ValueError, match="at least 2 participants"):
            await repo.create_group_conversation(
                room_id=room.id,
                user_ids=[user.id],
                ai_ids=[],
            )

    async def test_get_by_id_success(self, db_session, room_factory, conversation_factory):
        """Test successful conversation retrieval by ID."""
        # Arrange
        repo = ConversationRepository(db_session)
        room = await room_factory.create(db_session)
        conversation = await conversation_factory.create_private_conversation(db_session, room=room)

        # Act
        found_conversation = await repo.get_by_id(conversation.id)

        # Assert
        assert found_conversation is not None
        assert found_conversation.id == conversation.id
        assert found_conversation.room_id == room.id

    async def test_get_by_id_inactive_conversation(self, db_session, room_factory, conversation_factory):
        """Test that inactive conversations are not returned by get_by_id."""
        # Arrange
        repo = ConversationRepository(db_session)
        room = await room_factory.create(db_session)
        conversation = await conversation_factory.create_private_conversation(db_session, room=room, is_active=False)

        # Act
        found_conversation = await repo.get_by_id(conversation.id)

        # Assert
        assert found_conversation is None

    async def test_add_participant_success(self, db_session, user_factory, room_factory, conversation_factory):
        """Test adding participant to conversation."""
        # Arrange
        repo = ConversationRepository(db_session)
        room = await room_factory.create(db_session)
        conversation = await conversation_factory.create_group_conversation(db_session, room=room)
        new_user = await user_factory.create(db_session, username="newuser")

        # Act
        participant = await repo.add_participant(conversation.id, user_id=new_user.id)

        # Assert
        assert participant.id is not None
        assert participant.conversation_id == conversation.id
        assert participant.user_id == new_user.id
        assert participant.left_at is None

    async def test_add_participant_duplicate(self, db_session, user_factory, room_factory, conversation_factory):
        """Test adding duplicate participant raises error."""
        # Arrange
        repo = ConversationRepository(db_session)
        room = await room_factory.create(db_session)
        user = await user_factory.create(db_session)
        conversation = await conversation_factory.create_group_conversation(db_session, room=room)
        await repo.add_participant(conversation.id, user_id=user.id)

        # Act & Assert
        with pytest.raises(ValueError, match="already a participant"):
            await repo.add_participant(conversation.id, user_id=user.id)

    async def test_remove_participant_success(self, db_session, user_factory, room_factory, conversation_factory):
        """Test removing participant from conversation."""
        # Arrange
        repo = ConversationRepository(db_session)
        room = await room_factory.create(db_session)
        user = await user_factory.create(db_session)
        conversation = await conversation_factory.create_group_conversation(db_session, room=room)
        await repo.add_participant(conversation.id, user.id)

        # Act
        result = await repo.remove_participant(conversation.id, user_id=user.id)

        # Assert
        assert result is True

        # Verify participant is removed (left_at is set)
        is_participant = await repo.is_participant(conversation.id, user.id)
        assert is_participant is False

    async def test_remove_participant_not_found(self, db_session, room_factory, conversation_factory):
        """Test removing non-existent participant."""
        # Arrange
        repo = ConversationRepository(db_session)
        room = await room_factory.create(db_session)
        conversation = await conversation_factory.create_private_conversation(db_session, room=room)

        # Act
        result = await repo.remove_participant(conversation.id, user_id=99999)

        # Assert
        assert result is False

    async def test_is_participant_true(self, db_session, user_factory, room_factory, conversation_factory):
        """Test participant check when user is participant."""
        # Arrange
        repo = ConversationRepository(db_session)
        room = await room_factory.create(db_session)
        user = await user_factory.create(db_session)
        conversation = await conversation_factory.create_group_conversation(db_session, room=room)
        await repo.add_participant(conversation.id, user_id=user.id)

        # Act
        is_participant = await repo.is_participant(conversation.id, user.id)

        # Assert
        assert is_participant is True

    async def test_is_participant_false(self, db_session, room_factory, conversation_factory):
        """Test participant check when user is not participant."""
        # Arrange
        repo = ConversationRepository(db_session)
        room = await room_factory.create(db_session)
        conversation = await conversation_factory.create_private_conversation(db_session, room=room)

        # Act
        is_participant = await repo.is_participant(conversation.id, 99999)

        # Assert
        assert is_participant is False

    async def test_get_participants(self, db_session, user_factory, room_factory, conversation_factory):
        """Test retrieving all participants in conversation."""
        # Arrange
        repo = ConversationRepository(db_session)
        room = await room_factory.create(db_session)
        user1 = await user_factory.create(db_session, username="user1")
        user2 = await user_factory.create(db_session, username="user2")
        conversation = await conversation_factory.create_group_conversation(db_session, room=room)
        await repo.add_participant(conversation.id, user1.id)
        await repo.add_participant(conversation.id, user2.id)

        # Act
        participants = await repo.get_participants(conversation.id)

        # Assert
        assert len(participants) == 2
        participant_user_ids = [p.user_id for p in participants]
        assert user1.id in participant_user_ids
        assert user2.id in participant_user_ids

    async def test_get_user_conversations(self, db_session, user_factory, room_factory, conversation_factory):
        """Test retrieving all conversations for a user."""
        # Arrange
        repo = ConversationRepository(db_session)
        room = await room_factory.create(db_session)
        user = await user_factory.create(db_session)

        # Create 2 conversations
        conv1 = await conversation_factory.create_private_conversation(db_session, room=room)
        conv2 = await conversation_factory.create_group_conversation(db_session, room=room)

        # Add user to both
        await repo.add_participant(conv1.id, user_id=user.id)
        await repo.add_participant(conv2.id, user_id=user.id)

        # Act
        conversations = await repo.get_user_conversations(user.id)

        # Assert
        assert len(conversations) == 2
        conversation_ids = [c.id for c in conversations]
        assert conv1.id in conversation_ids
        assert conv2.id in conversation_ids

    async def test_get_room_conversations(self, db_session, room_factory, conversation_factory):
        """Test retrieving all conversations in a room."""
        # Arrange
        repo = ConversationRepository(db_session)
        room1 = await room_factory.create(db_session, name="Room 1")
        room2 = await room_factory.create(db_session, name="Room 2")

        # Create conversations in different rooms
        conv1 = await conversation_factory.create_private_conversation(db_session, room=room1)
        conv2 = await conversation_factory.create_group_conversation(db_session, room=room1)
        await conversation_factory.create_private_conversation(db_session, room=room2)

        # Act
        room1_conversations = await repo.get_room_conversations(room1.id)

        # Assert
        assert len(room1_conversations) == 2
        conversation_ids = [c.id for c in room1_conversations]
        assert conv1.id in conversation_ids
        assert conv2.id in conversation_ids

    async def test_update_conversation_success(self, db_session, room_factory, conversation_factory):
        """Test successful conversation update."""
        # Arrange
        repo = ConversationRepository(db_session)
        room = await room_factory.create(db_session)
        conversation = await conversation_factory.create_group_conversation(db_session, room=room, max_participants=5)

        # Act
        conversation.max_participants = 10
        updated_conversation = await repo.update(conversation)

        # Assert
        assert updated_conversation.max_participants == 10

        # Verify in database
        found_conversation = await repo.get_by_id(conversation.id)
        assert found_conversation.max_participants == 10

    async def test_soft_delete_conversation_success(self, db_session, room_factory, conversation_factory):
        """Test successful conversation soft deletion."""
        # Arrange
        repo = ConversationRepository(db_session)
        room = await room_factory.create(db_session)
        conversation = await conversation_factory.create_private_conversation(db_session, room=room)
        conversation_id = conversation.id

        # Act
        result = await repo.delete(conversation_id)

        # Assert
        assert result is True

        # Verify conversation is soft-deleted (not returned by get_by_id)
        found_conversation = await repo.get_by_id(conversation_id)
        assert found_conversation is None

    async def test_exists_conversation_true(self, db_session, room_factory, conversation_factory):
        """Test conversation existence check when conversation exists."""
        # Arrange
        repo = ConversationRepository(db_session)
        room = await room_factory.create(db_session)
        conversation = await conversation_factory.create_private_conversation(db_session, room=room)

        # Act
        exists = await repo.exists(conversation.id)

        # Assert
        assert exists is True

    async def test_exists_conversation_false(self, db_session):
        """Test conversation existence check when conversation does not exist."""
        # Arrange
        repo = ConversationRepository(db_session)

        # Act
        exists = await repo.exists(99999)

        # Assert
        assert exists is False
