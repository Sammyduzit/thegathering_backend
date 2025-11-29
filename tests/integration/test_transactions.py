"""
Integration tests for database transactions with PostgreSQL.

Tests verify PostgreSQL transaction behavior:
- Transaction isolation
- Rollback on error
- ACID properties
- Concurrent operations
- Savepoints

These tests require PostgreSQL and will fail with SQLite.
"""

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.models.message import Message
from app.models.room import Room
from app.models.user import User


@pytest.mark.integration
class TestTransactions:
    """Integration tests for transaction behavior with PostgreSQL."""

    async def test_transaction_commit_success(self, db_session, user_factory):
        """Test successful transaction commit."""
        # Arrange
        user = user_factory.build(username="testuser", email="test@example.com")

        # Act
        db_session.add(user)
        await db_session.commit()

        # Assert - verify user persisted
        result = await db_session.execute(select(User).where(User.username == "testuser"))
        persisted_user = result.scalar_one_or_none()
        assert persisted_user is not None
        assert persisted_user.email == "test@example.com"

    async def test_transaction_rollback_on_error(self, db_session, user_factory):
        """Test transaction rollback when error occurs."""
        # Arrange
        user1 = user_factory.build(username="user1", email="user1@example.com")
        db_session.add(user1)
        await db_session.commit()

        # Act - Try to create user with duplicate username (should fail)
        user2 = user_factory.build(username="user1", email="different@example.com")
        db_session.add(user2)

        with pytest.raises(IntegrityError):
            await db_session.commit()

        await db_session.rollback()

        # Assert - second user should NOT exist
        result = await db_session.execute(select(User).where(User.email == "different@example.com"))
        failed_user = result.scalar_one_or_none()
        assert failed_user is None

        # Original user should still exist
        result = await db_session.execute(select(User).where(User.username == "user1"))
        original_user = result.scalar_one_or_none()
        assert original_user is not None

    async def test_transaction_isolation_read_uncommitted(self, integration_engine, user_factory):
        """Test that uncommitted changes are not visible in other sessions."""
        from sqlalchemy.ext.asyncio import AsyncSession

        # Session 1: Create user but don't commit
        async with AsyncSession(integration_engine) as session1:
            user = user_factory.build(username="pending", email="pending@example.com")
            session1.add(user)
            await session1.flush()  # Write to DB but don't commit

            # Session 2: Try to read uncommitted user
            async with AsyncSession(integration_engine) as session2:
                result = await session2.execute(select(User).where(User.username == "pending"))
                uncommitted_user = result.scalar_one_or_none()

                # Should NOT see uncommitted changes (READ COMMITTED isolation)
                assert uncommitted_user is None

            # Rollback session1
            await session1.rollback()

    async def test_transaction_multiple_operations_atomic(
        self, db_session, user_factory, room_factory, message_factory
    ):
        """Test that multiple operations in one transaction are atomic."""
        # Arrange
        user = await user_factory.create(db_session)
        room = await room_factory.create(db_session)

        # Act - Create 3 messages in one transaction
        message1 = Message(sender_user_id=user.id, room_id=room.id, content="Message 1")
        message2 = Message(sender_user_id=user.id, room_id=room.id, content="Message 2")
        message3 = Message(sender_user_id=user.id, room_id=room.id, content="Message 3")

        db_session.add_all([message1, message2, message3])
        await db_session.commit()

        # Assert - All messages should exist
        result = await db_session.execute(select(Message).where(Message.room_id == room.id))
        messages = result.scalars().all()
        assert len(messages) == 3

    async def test_transaction_partial_rollback(self, db_session, user_factory, room_factory):
        """Test rollback only affects uncommitted changes."""
        # Arrange - Create and commit first user
        await user_factory.create(db_session, username="committed_user")

        # Act - Create second user but rollback
        user2 = user_factory.build(username="rolled_back_user")
        db_session.add(user2)
        await db_session.rollback()

        # Assert - First user still exists
        result = await db_session.execute(select(User).where(User.username == "committed_user"))
        committed_user = result.scalar_one_or_none()
        assert committed_user is not None

        # Second user should NOT exist
        result = await db_session.execute(select(User).where(User.username == "rolled_back_user"))
        rolled_back_user = result.scalar_one_or_none()
        assert rolled_back_user is None

    async def test_transaction_flush_vs_commit(self, db_session, user_factory):
        """Test difference between flush() and commit()."""
        # Arrange
        user = user_factory.build(username="flushed", email="flush@example.com")

        # Act - Flush writes to DB but doesn't commit
        db_session.add(user)
        await db_session.flush()

        # User has ID after flush
        assert user.id is not None

        # But rollback will undo it
        await db_session.rollback()

        # Assert - User should NOT persist
        result = await db_session.execute(select(User).where(User.username == "flushed"))
        flushed_user = result.scalar_one_or_none()
        assert flushed_user is None

    async def test_transaction_nested_savepoints(self, db_session, user_factory):
        """Test nested transactions with savepoints."""
        # Create first user
        user1 = user_factory.build(username="user1", email="user1@example.com")
        db_session.add(user1)
        await db_session.commit()

        # Nested transaction with savepoint
        async with db_session.begin_nested() as savepoint:
            user2 = user_factory.build(username="user2", email="user2@example.com")
            db_session.add(user2)
            await db_session.flush()

            # Rollback to savepoint
            await savepoint.rollback()

        # Commit outer transaction
        await db_session.commit()

        # Assert - user1 exists, user2 does not
        result = await db_session.execute(select(User).where(User.username == "user1"))
        assert result.scalar_one_or_none() is not None

        result = await db_session.execute(select(User).where(User.username == "user2"))
        assert result.scalar_one_or_none() is None

    async def test_concurrent_insert_same_unique_field(self, integration_engine, user_factory):
        """Test concurrent inserts with unique constraint."""
        from sqlalchemy.ext.asyncio import AsyncSession

        # Both sessions try to create user with same username
        async def create_user(username: str):
            async with AsyncSession(integration_engine) as session:
                user = user_factory.build(username=username, email=f"{username}@example.com")
                session.add(user)
                try:
                    await session.commit()
                    return True
                except IntegrityError:
                    await session.rollback()
                    return False

        # Act - Try to create same username concurrently
        import asyncio

        results = await asyncio.gather(create_user("concurrent"), create_user("concurrent"), return_exceptions=True)

        # Assert - Only one should succeed
        successes = sum(1 for r in results if r is True)
        assert successes == 1

    async def test_transaction_delete_with_cascade(self, db_session, user_factory, room_factory, conversation_factory):
        """Test transaction with CASCADE delete behavior."""
        # Arrange
        user = await user_factory.create(db_session)
        room = await room_factory.create(db_session)
        conversation = await conversation_factory.create_private_conversation(db_session, room=room)

        # Create messages
        message1 = Message(sender_user_id=user.id, conversation_id=conversation.id, content="Msg 1")
        message2 = Message(sender_user_id=user.id, conversation_id=conversation.id, content="Msg 2")
        db_session.add_all([message1, message2])
        await db_session.commit()

        conversation_id = conversation.id

        # Act - Delete conversation in transaction
        from sqlalchemy import text

        await db_session.execute(text("DELETE FROM conversations WHERE id = :id"), {"id": conversation_id})
        await db_session.commit()

        # Assert - Messages CASCADE deleted
        result = await db_session.execute(select(Message).where(Message.conversation_id == conversation_id))
        messages = result.scalars().all()
        assert len(messages) == 0

    async def test_transaction_isolation_dirty_read_prevented(self, integration_engine, user_factory):
        """Test that dirty reads are prevented (READ COMMITTED isolation)."""
        from sqlalchemy.ext.asyncio import AsyncSession

        # Create initial user
        async with AsyncSession(integration_engine) as setup_session:
            user = user_factory.build(username="dirty_read_test", email="test@example.com")
            setup_session.add(user)
            await setup_session.commit()

        # Session 1: Update user but don't commit
        async with AsyncSession(integration_engine) as session1:
            result = await session1.execute(select(User).where(User.username == "dirty_read_test"))
            user = result.scalar_one()
            original_email = user.email

            user.email = "updated@example.com"
            await session1.flush()  # Write but don't commit

            # Session 2: Read same user
            async with AsyncSession(integration_engine) as session2:
                result = await session2.execute(select(User).where(User.username == "dirty_read_test"))
                user_read = result.scalar_one()

                # Should see original value, not uncommitted change
                assert user_read.email == original_email
                assert user_read.email != "updated@example.com"

            # Rollback session1
            await session1.rollback()

    async def test_transaction_bulk_operations_atomic(self, db_session, user_factory, room_factory):
        """Test bulk operations are atomic within transaction."""
        # Act - Bulk create 10 users
        users = []
        for i in range(10):
            user = user_factory.build(username=f"bulk_user_{i}", email=f"bulk{i}@example.com")
            users.append(user)

        db_session.add_all(users)
        await db_session.commit()

        # Assert - All 10 users should exist
        result = await db_session.execute(select(User).where(User.username.like("bulk_user_%")))
        created_users = result.scalars().all()
        assert len(created_users) == 10

    async def test_transaction_rollback_resets_session_state(self, db_session, user_factory):
        """Test rollback resets session to clean state."""
        # Arrange
        user1 = user_factory.build(username="user1", email="user1@example.com")
        db_session.add(user1)

        # Rollback
        await db_session.rollback()

        # Act - Create new user after rollback
        user2 = user_factory.build(username="user2", email="user2@example.com")
        db_session.add(user2)
        await db_session.commit()

        # Assert - Only user2 exists
        result = await db_session.execute(select(User).where(User.username == "user1"))
        assert result.scalar_one_or_none() is None

        result = await db_session.execute(select(User).where(User.username == "user2"))
        assert result.scalar_one_or_none() is not None
