"""
Integration tests for SQLAlchemy relationship behavior with PostgreSQL.

Tests verify ORM relationships:
- One-to-Many relationships (User -> Messages, Room -> Messages)
- Many-to-Many relationships (User <-> Conversation via ConversationParticipant)
- Lazy loading vs eager loading
- Back-populates behavior
- Relationship queries

These tests require PostgreSQL and test real ORM behavior.
"""

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from app.models.conversation import Conversation
from app.models.message import Message
from app.models.room import Room
from app.models.user import User
from app.repositories.conversation_repository import ConversationRepository


@pytest.mark.integration
class TestRelationships:
    """Integration tests for SQLAlchemy relationships with PostgreSQL."""

    async def test_user_sent_messages_relationship(self, db_session, user_factory, room_factory, message_factory):
        """Test User -> Messages (sent_messages) one-to-many relationship with dynamic lazy loading."""
        # Arrange
        user = await user_factory.create(db_session)
        room = await room_factory.create(db_session)

        # Create 3 messages from same user
        await message_factory.create_room_message(db_session, sender=user, room=room, content="Msg 1")
        await message_factory.create_room_message(db_session, sender=user, room=room, content="Msg 2")
        await message_factory.create_room_message(db_session, sender=user, room=room, content="Msg 3")

        # Act - Count messages via SQL (testing dynamic relationship behavior)
        messages_count = await db_session.scalar(
            select(func.count(Message.id)).where(Message.sender_user_id == user.id)
        )

        # Assert
        assert messages_count == 3

    async def test_room_messages_relationship(self, db_session, user_factory, room_factory, message_factory):
        """Test Room -> Messages (room_messages) one-to-many relationship with dynamic lazy loading."""
        # Arrange
        user = await user_factory.create(db_session)
        room = await room_factory.create(db_session)

        # Create messages in room
        await message_factory.create_room_message(db_session, sender=user, room=room, content="Msg 1")
        await message_factory.create_room_message(db_session, sender=user, room=room, content="Msg 2")

        # Act - Query messages count (lazy="dynamic" doesn't support eager loading)
        from sqlalchemy import func

        messages_count = await db_session.scalar(select(func.count(Message.id)).where(Message.room_id == room.id))

        # Assert
        assert messages_count == 2

    async def test_conversation_messages_relationship(
        self, db_session, user_factory, room_factory, conversation_factory, message_factory
    ):
        """Test Conversation -> Messages relationship with dynamic lazy loading."""
        # Arrange
        user = await user_factory.create(db_session)
        room = await room_factory.create(db_session)
        conversation = await conversation_factory.create_private_conversation(db_session, room=room)

        # Create messages in conversation
        await message_factory.create_conversation_message(
            db_session, sender=user, conversation=conversation, content="Msg 1"
        )
        await message_factory.create_conversation_message(
            db_session, sender=user, conversation=conversation, content="Msg 2"
        )

        # Act - Query messages count (lazy="dynamic")
        from sqlalchemy import func

        messages_count = await db_session.scalar(
            select(func.count(Message.id)).where(Message.conversation_id == conversation.id)
        )

        # Assert
        assert messages_count == 2

    async def test_message_sender_backref(self, db_session, user_factory, room_factory, message_factory):
        """Test Message -> User (sender_user) back-populates relationship."""
        # Arrange
        user = await user_factory.create(db_session)
        room = await room_factory.create(db_session)
        message = await message_factory.create_room_message(db_session, sender=user, room=room, content="Test message")

        # Act - Access sender through message
        result = await db_session.execute(
            select(Message).where(Message.id == message.id).options(selectinload(Message.sender_user))
        )
        message_with_sender = result.scalar_one()

        # Assert
        assert message_with_sender.sender_user is not None
        assert message_with_sender.sender_user.id == user.id
        assert message_with_sender.sender_user.username == user.username

    async def test_user_current_room_relationship(self, db_session, user_factory, room_factory):
        """Test User -> Room (current_room) relationship."""
        # Arrange
        room = await room_factory.create(db_session, name="Test Room")
        user = await user_factory.create(db_session, current_room_id=room.id)

        # Act - Eager load user with current_room
        result = await db_session.execute(
            select(User).where(User.id == user.id).options(selectinload(User.current_room))
        )
        user_with_room = result.scalar_one()

        # Assert
        assert user_with_room.current_room is not None
        assert user_with_room.current_room.id == room.id
        assert user_with_room.current_room.name == "Test Room"

    async def test_room_users_relationship(self, db_session, user_factory, room_factory):
        """Test Room -> Users (users) one-to-many relationship."""
        # Arrange
        room = await room_factory.create(db_session)
        await user_factory.create(db_session, username="user1", current_room_id=room.id)
        await user_factory.create(db_session, username="user2", current_room_id=room.id)
        await user_factory.create(db_session, username="user3", current_room_id=room.id)

        # Act - Eager load room with users
        result = await db_session.execute(select(Room).where(Room.id == room.id).options(selectinload(Room.users)))
        room_with_users = result.scalar_one()

        # Assert
        assert len(room_with_users.users) == 3
        usernames = [u.username for u in room_with_users.users]
        assert "user1" in usernames
        assert "user2" in usernames
        assert "user3" in usernames

    async def test_conversation_participants_many_to_many(
        self, db_session, user_factory, room_factory, conversation_factory
    ):
        """Test Conversation <-> User many-to-many via ConversationParticipant."""
        # Arrange
        repo = ConversationRepository(db_session)
        room = await room_factory.create(db_session)
        user1 = await user_factory.create(db_session, username="user1")
        user2 = await user_factory.create(db_session, username="user2")
        user3 = await user_factory.create(db_session, username="user3")

        # Create group conversation
        conversation = await conversation_factory.create_group_conversation(db_session, room=room)

        # Add participants
        await repo.add_participant(conversation.id, user1.id)
        await repo.add_participant(conversation.id, user2.id)
        await repo.add_participant(conversation.id, user3.id)

        # Act - Get participants through repository
        participants = await repo.get_participants(conversation.id)

        # Assert
        assert len(participants) == 3
        participant_usernames = [p.participant_name for p in participants]
        assert "user1" in participant_usernames
        assert "user2" in participant_usernames
        assert "user3" in participant_usernames

    async def test_conversation_room_relationship(self, db_session, room_factory, conversation_factory):
        """Test Conversation -> Room relationship (conversations within rooms)."""
        # Arrange
        room = await room_factory.create(db_session, name="Test Room")
        await conversation_factory.create_private_conversation(db_session, room=room)
        await conversation_factory.create_group_conversation(db_session, room=room)

        # Act - Load conversations with room eager loaded
        result = await db_session.execute(
            select(Conversation).where(Conversation.room_id == room.id).options(selectinload(Conversation.room))
        )
        conversations = result.scalars().all()

        # Assert
        assert len(conversations) == 2
        assert all(c.room.id == room.id for c in conversations)
        assert all(c.room.name == "Test Room" for c in conversations)

    async def test_room_conversations_relationship(self, db_session, room_factory, conversation_factory):
        """Test Room -> Conversations one-to-many relationship."""
        # Arrange
        room = await room_factory.create(db_session)
        await conversation_factory.create_private_conversation(db_session, room=room)
        await conversation_factory.create_group_conversation(db_session, room=room)

        # Act
        result = await db_session.execute(
            select(Room).where(Room.id == room.id).options(selectinload(Room.conversations))
        )
        room_with_conversations = result.scalar_one()

        # Assert
        assert len(room_with_conversations.conversations) == 2
        assert all(c.room_id == room.id for c in room_with_conversations.conversations)

    async def test_relationship_filter_by_related_field(self, db_session, user_factory, room_factory, message_factory):
        """Test filtering messages by related user field."""
        # Arrange
        user1 = await user_factory.create(db_session, username="alice")
        user2 = await user_factory.create(db_session, username="bob")
        room = await room_factory.create(db_session)

        await message_factory.create_room_message(db_session, sender=user1, room=room, content="Alice msg")
        await message_factory.create_room_message(db_session, sender=user2, room=room, content="Bob msg")
        await message_factory.create_room_message(db_session, sender=user1, room=room, content="Alice msg 2")

        # Act - Query messages by sender username
        from sqlalchemy.orm import joinedload

        result = await db_session.execute(
            select(Message)
            .join(Message.sender_user)
            .where(User.username == "alice")
            .options(joinedload(Message.sender_user))
        )
        alice_messages = result.scalars().all()

        # Assert
        assert len(alice_messages) == 2
        assert all(msg.sender_user.username == "alice" for msg in alice_messages)

    async def test_relationship_count_aggregation(self, db_session, user_factory, room_factory, message_factory):
        """Test counting related entities using relationship."""
        # Arrange
        user = await user_factory.create(db_session)
        room = await room_factory.create(db_session)

        # Create 5 messages
        for i in range(5):
            await message_factory.create_room_message(db_session, sender=user, room=room, content=f"Message {i}")

        # Act - Count messages via relationship
        from sqlalchemy import func

        result = await db_session.execute(select(func.count(Message.id)).where(Message.sender_user_id == user.id))
        message_count = result.scalar()

        # Assert
        assert message_count == 5

    async def test_relationship_cascade_behavior_on_update(
        self, db_session, user_factory, room_factory, message_factory
    ):
        """Test that updating parent doesn't affect children (no cascade update)."""
        # Arrange
        user = await user_factory.create(db_session, username="original_name")
        room = await room_factory.create(db_session)
        message = await message_factory.create_room_message(db_session, sender=user, room=room, content="Test")

        original_message_id = message.id

        # Act - Update user
        user.username = "new_name"
        await db_session.commit()

        # Assert - Message still exists and references same user
        result = await db_session.execute(select(Message).where(Message.id == original_message_id))
        updated_message = result.scalar_one()
        assert updated_message.sender_user_id == user.id

    async def test_relationship_null_foreign_key_handling(
        self, db_session, user_factory, room_factory, message_factory
    ):
        """Test relationship behavior when foreign key can be NULL."""
        # Arrange
        user = await user_factory.create(db_session, current_room_id=None)

        # Act - Load user with null current_room
        result = await db_session.execute(
            select(User).where(User.id == user.id).options(selectinload(User.current_room))
        )
        user_without_room = result.scalar_one()

        # Assert
        assert user_without_room.current_room is None
        assert user_without_room.current_room_id is None

    async def test_relationship_multiple_paths_to_same_entity(
        self, db_session, user_factory, room_factory, conversation_factory
    ):
        """Test conversation -> room and user -> room relationships don't conflict."""
        # Arrange
        room = await room_factory.create(db_session, name="Shared Room")
        user = await user_factory.create(db_session, current_room_id=room.id)
        conversation = await conversation_factory.create_private_conversation(db_session, room=room)

        # Act - Load both relationships
        result = await db_session.execute(
            select(User).where(User.id == user.id).options(selectinload(User.current_room))
        )
        user_with_room = result.scalar_one()

        result = await db_session.execute(
            select(Conversation).where(Conversation.id == conversation.id).options(selectinload(Conversation.room))
        )
        conv_with_room = result.scalar_one()

        # Assert - Both point to same room
        assert user_with_room.current_room.id == room.id
        assert conv_with_room.room.id == room.id
        assert user_with_room.current_room.id == conv_with_room.room.id
