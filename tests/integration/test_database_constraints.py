"""
Integration tests for database constraints with PostgreSQL.

Tests verify PostgreSQL-specific features:
- XOR constraint on message routing (room XOR conversation)
- Foreign key constraints and CASCADE behavior
- NOT NULL constraints
- Unique constraints
- Check constraints

These tests require PostgreSQL and will fail with SQLite.
"""

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.models.conversation import Conversation, ConversationType
from app.models.message import Message
from app.models.room import Room
from app.models.user import User


@pytest.mark.integration
class TestDatabaseConstraints:
    """Integration tests for database constraints with PostgreSQL."""

    async def test_message_xor_constraint_both_null_fails(self, db_session, user_factory):
        """Test XOR constraint fails when both room_id and conversation_id are NULL."""
        # Arrange
        user = await user_factory.create(db_session)

        # Act & Assert - both NULL violates XOR constraint
        message = Message(
            sender_user_id=user.id,
            content="Test message",
            room_id=None,
            conversation_id=None,
        )
        db_session.add(message)

        with pytest.raises(IntegrityError, match="message_xor_room_conversation"):
            await db_session.commit()

        await db_session.rollback()

    async def test_message_xor_constraint_both_set_fails(
        self, db_session, user_factory, room_factory, conversation_factory
    ):
        """Test XOR constraint fails when both room_id and conversation_id are set."""
        # Arrange
        user = await user_factory.create(db_session)
        room = await room_factory.create(db_session)
        conversation = await conversation_factory.create_private_conversation(db_session, room=room)

        # Act & Assert - both set violates XOR constraint
        message = Message(
            sender_user_id=user.id,
            content="Test message",
            room_id=room.id,
            conversation_id=conversation.id,
        )
        db_session.add(message)

        with pytest.raises(IntegrityError, match="message_xor_room_conversation"):
            await db_session.commit()

        await db_session.rollback()

    async def test_message_xor_constraint_room_only_succeeds(self, db_session, user_factory, room_factory):
        """Test XOR constraint succeeds with room_id set, conversation_id NULL."""
        # Arrange
        user = await user_factory.create(db_session)
        room = await room_factory.create(db_session)

        # Act
        message = Message(
            sender_user_id=user.id,
            content="Room message",
            room_id=room.id,
            conversation_id=None,
        )
        db_session.add(message)
        await db_session.commit()

        # Assert
        assert message.id is not None
        assert message.room_id == room.id
        assert message.conversation_id is None

    async def test_message_xor_constraint_conversation_only_succeeds(
        self, db_session, user_factory, room_factory, conversation_factory
    ):
        """Test XOR constraint succeeds with conversation_id set, room_id NULL."""
        # Arrange
        user = await user_factory.create(db_session)
        room = await room_factory.create(db_session)
        conversation = await conversation_factory.create_private_conversation(db_session, room=room)

        # Act
        message = Message(
            sender_user_id=user.id,
            content="Conversation message",
            room_id=None,
            conversation_id=conversation.id,
        )
        db_session.add(message)
        await db_session.commit()

        # Assert
        assert message.id is not None
        assert message.conversation_id == conversation.id
        assert message.room_id is None

    async def test_conversation_room_id_not_null_constraint(self, db_session):
        """Test that conversations require room_id (NOT NULL constraint)."""
        # Act & Assert
        conversation = Conversation(
            room_id=None,  # This violates NOT NULL constraint
            conversation_type=ConversationType.PRIVATE,
            max_participants=2,
        )
        db_session.add(conversation)

        with pytest.raises(IntegrityError, match="NOT NULL|null value"):
            await db_session.commit()

        await db_session.rollback()

    async def test_foreign_key_constraint_invalid_user_id(self, db_session, room_factory):
        """Test foreign key constraint fails with invalid user_id."""
        # Arrange
        room = await room_factory.create(db_session)

        # Act & Assert - invalid sender_user_id (user doesn't exist)
        message = Message(
            sender_user_id=99999,  # Non-existent user
            content="Test message",
            room_id=room.id,
        )
        db_session.add(message)

        with pytest.raises(IntegrityError, match="foreign key constraint|FOREIGN KEY"):
            await db_session.commit()

        await db_session.rollback()

    async def test_foreign_key_constraint_invalid_room_id(self, db_session, user_factory):
        """Test foreign key constraint fails with invalid room_id."""
        # Arrange
        user = await user_factory.create(db_session)

        # Act & Assert - invalid room_id (room doesn't exist)
        message = Message(
            sender_user_id=user.id,
            content="Test message",
            room_id=99999,  # Non-existent room
        )
        db_session.add(message)

        with pytest.raises(IntegrityError, match="foreign key constraint|FOREIGN KEY"):
            await db_session.commit()

        await db_session.rollback()

    async def test_foreign_key_constraint_invalid_conversation_id(self, db_session, user_factory):
        """Test foreign key constraint fails with invalid conversation_id."""
        # Arrange
        user = await user_factory.create(db_session)

        # Act & Assert - invalid conversation_id
        message = Message(
            sender_user_id=user.id,
            content="Test message",
            conversation_id=99999,  # Non-existent conversation
        )
        db_session.add(message)

        with pytest.raises(IntegrityError, match="foreign key constraint|FOREIGN KEY"):
            await db_session.commit()

        await db_session.rollback()

    async def test_cascade_delete_conversation_deletes_messages(
        self, db_session, user_factory, room_factory, conversation_factory, message_factory
    ):
        """Test CASCADE DELETE: deleting conversation deletes its messages."""
        # Arrange
        user = await user_factory.create(db_session)
        room = await room_factory.create(db_session)
        conversation = await conversation_factory.create_private_conversation(db_session, room=room)

        # Create messages using factory to ensure proper commit
        message1 = await message_factory.create_conversation_message(
            db_session, sender=user, conversation=conversation, content="Message 1"
        )
        message2 = await message_factory.create_conversation_message(
            db_session, sender=user, conversation=conversation, content="Message 2"
        )

        message1_id = message1.id
        message2_id = message2.id

        # Act - Delete conversation (CASCADE should delete messages)
        # Use raw SQL to delete conversation to trigger CASCADE
        from sqlalchemy import text

        await db_session.execute(text("DELETE FROM conversations WHERE id = :conv_id"), {"conv_id": conversation.id})
        await db_session.commit()

        # Assert - Messages should be deleted due to CASCADE
        result = await db_session.execute(select(Message).where(Message.id.in_([message1_id, message2_id])))
        deleted_messages = result.scalars().all()
        assert len(deleted_messages) == 0

    async def test_set_null_on_delete_room_message(self, db_session, user_factory, room_factory):
        """Test SET NULL: deleting room sets message.room_id to NULL (but keeps message)."""
        # Arrange
        user = await user_factory.create(db_session)
        room = await room_factory.create(db_session)

        message = Message(
            sender_user_id=user.id,
            content="Room message",
            room_id=room.id,
        )
        db_session.add(message)
        await db_session.commit()

        # Act - Delete room (SET NULL should keep message but set room_id to NULL)
        await db_session.delete(room)

        # This will fail because XOR constraint requires room_id XOR conversation_id
        # When room is deleted and room_id becomes NULL, the message violates XOR
        # This is expected behavior - room messages can't exist without a room
        with pytest.raises(IntegrityError, match="message_xor_room_conversation"):
            await db_session.commit()

        await db_session.rollback()

    async def test_unique_constraint_room_name(self, db_session, room_factory):
        """Test that room names must be unique."""
        # Arrange
        await room_factory.create(db_session, name="Unique Room")

        # Act & Assert - duplicate name should fail
        room2 = Room(
            name="Unique Room",  # Duplicate name
            description="Another room",
            max_users=10,
        )
        db_session.add(room2)

        with pytest.raises(IntegrityError, match="unique constraint|UNIQUE"):
            await db_session.commit()

        await db_session.rollback()

    async def test_unique_constraint_user_email(self, db_session, user_factory):
        """Test that user emails must be unique."""
        # Arrange
        await user_factory.create(db_session, email="unique@example.com")

        # Act & Assert - duplicate email should fail
        from app.core.auth_utils import hash_password

        user2 = User(
            email="unique@example.com",  # Duplicate email
            username="different_username",
            password_hash=hash_password("password123"),
        )
        db_session.add(user2)

        with pytest.raises(IntegrityError, match="unique constraint|UNIQUE"):
            await db_session.commit()

        await db_session.rollback()

    async def test_unique_constraint_user_username(self, db_session, user_factory):
        """Test that usernames must be unique."""
        # Arrange
        await user_factory.create(db_session, username="uniqueuser")

        # Act & Assert - duplicate username should fail
        from app.core.auth_utils import hash_password

        user2 = User(
            email="different@example.com",
            username="uniqueuser",  # Duplicate username
            password_hash=hash_password("password123"),
        )
        db_session.add(user2)

        with pytest.raises(IntegrityError, match="unique constraint|UNIQUE"):
            await db_session.commit()

        await db_session.rollback()

    async def test_not_null_constraint_message_content(self, db_session, user_factory, room_factory):
        """Test that message content cannot be NULL."""
        # Arrange
        user = await user_factory.create(db_session)
        room = await room_factory.create(db_session)

        # Act & Assert
        message = Message(
            sender_user_id=user.id,
            content=None,  # NULL content should fail
            room_id=room.id,
        )
        db_session.add(message)

        with pytest.raises(IntegrityError, match="NOT NULL|null value"):
            await db_session.commit()

        await db_session.rollback()

    async def test_not_null_constraint_user_email(self, db_session):
        """Test that user email cannot be NULL."""
        # Arrange
        from app.core.auth_utils import hash_password

        # Act & Assert
        user = User(
            email=None,  # NULL email should fail
            username="testuser",
            password_hash=hash_password("password123"),
        )
        db_session.add(user)

        with pytest.raises(IntegrityError, match="NOT NULL|null value"):
            await db_session.commit()

        await db_session.rollback()

    async def test_not_null_constraint_room_name(self, db_session):
        """Test that room name cannot be NULL."""
        # Act & Assert
        room = Room(
            name=None,  # NULL name should fail
            max_users=10,
        )
        db_session.add(room)

        with pytest.raises(IntegrityError, match="NOT NULL|null value"):
            await db_session.commit()

        await db_session.rollback()
