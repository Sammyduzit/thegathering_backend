"""
Test data factories for creating model instances.

This module provides factory classes for creating test data with sensible
defaults while allowing easy customization. Follows the Builder pattern
for clean, readable test setup.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth_utils import hash_password
from app.models.ai_entity import AIEntity, AIEntityStatus, AIResponseStrategy
from app.models.conversation import Conversation, ConversationType
from app.models.message import Message
from app.models.room import Room
from app.models.user import User, UserStatus


class BaseFactory:
    """Base factory with common functionality."""

    @classmethod
    async def create(cls, session: AsyncSession, **overrides) -> Any:
        """Create and persist instance with given overrides."""
        instance = cls.build(**overrides)
        session.add(instance)
        await session.commit()
        await session.refresh(instance)
        return instance

    @classmethod
    def build(cls, **overrides) -> Any:
        """Build instance without persisting to database."""
        defaults = cls.get_defaults()
        data = {**defaults, **overrides}
        return cls.model_class(**data)

    @classmethod
    def get_defaults(cls) -> Dict[str, Any]:
        """Get default values for this factory."""
        raise NotImplementedError("Subclasses must implement get_defaults")


class UserFactory(BaseFactory):
    """Factory for creating User instances."""

    model_class = User

    @classmethod
    def get_defaults(cls) -> Dict[str, Any]:
        """Default values for User creation."""
        unique_id = str(uuid.uuid4())[:8]
        return {
            "email": f"test_{unique_id}@example.com",
            "username": f"testuser_{unique_id}",
            "password_hash": hash_password("password123"),
            "is_admin": False,
            "is_active": True,
            "status": UserStatus.AVAILABLE,
            "last_active": datetime.now(),
            "preferred_language": "en",
            "avatar_url": None,
            "current_room_id": None,
        }

    @classmethod
    async def create_admin(cls, session: AsyncSession, **overrides) -> User:
        """Create admin user with admin defaults."""
        admin_defaults = {
            "email": "admin@example.com",
            "username": "admin",
            "is_admin": True,
        }
        return await cls.create(session, **admin_defaults, **overrides)

    @classmethod
    async def create_user_with_room(cls, session: AsyncSession, room: Optional[Room] = None, **overrides) -> User:
        """Create user assigned to a room."""
        if room is None:
            room = await RoomFactory.create(session)

        user_defaults = {
            "current_room_id": room.id,
        }
        return await cls.create(session, **user_defaults, **overrides)

    @classmethod
    def build_multiple(cls, count: int, **base_overrides) -> list[User]:
        """Build multiple user instances with unique usernames/emails."""
        users = []
        for i in range(count):
            overrides = {"email": f"user{i}@example.com", "username": f"user{i}", **base_overrides}
            users.append(cls.build(**overrides))
        return users


class RoomFactory(BaseFactory):
    """Factory for creating Room instances."""

    model_class = Room

    @classmethod
    def get_defaults(cls) -> Dict[str, Any]:
        """Default values for Room creation."""
        unique_id = str(uuid.uuid4())[:8]
        return {
            "name": f"Test Room {unique_id}",
            "description": "A test room for testing purposes",
            "max_users": 10,
            "is_active": True,
            "is_translation_enabled": True,
            "created_at": datetime.now(),
        }

    @classmethod
    async def create_with_admin(cls, session: AsyncSession, admin: Optional[User] = None, **overrides) -> Room:
        """Create room with an admin user."""
        if admin is None:
            admin = await UserFactory.create_admin(session)

        return await cls.create(session, admin_id=admin.id, **overrides)

    @classmethod
    async def create_private_room(cls, session: AsyncSession, **overrides) -> Room:
        """Create private room with limited users."""
        private_defaults = {
            "name": "Private Room",
            "description": "A private test room",
            "max_users": 2,
        }
        return await cls.create(session, **private_defaults, **overrides)


class ConversationFactory(BaseFactory):
    """Factory for creating Conversation instances."""

    model_class = Conversation

    @classmethod
    def get_defaults(cls) -> Dict[str, Any]:
        """
        Default values for Conversation creation.

        NOTE: room_id is REQUIRED! Conversations exist within rooms.
        Use helper methods like create_private_conversation() which handle room creation.
        """
        return {
            "conversation_type": ConversationType.PRIVATE,
            "max_participants": 2,
            "is_active": True,
            "created_at": datetime.now(),
            # room_id MUST be provided - conversations are always within a room
        }

    @classmethod
    async def create_room_conversation(
        cls, session: AsyncSession, room: Optional[Room] = None, **overrides
    ) -> Conversation:
        """Create conversation linked to a room."""
        if room is None:
            room = await RoomFactory.create(session)

        room_defaults = {
            "room_id": room.id,
            "conversation_type": ConversationType.PUBLIC_ROOM,
            "max_participants": room.max_users,
        }
        return await cls.create(session, **room_defaults, **overrides)

    @classmethod
    async def create_private_conversation(
        cls, session: AsyncSession, room: Optional[Room] = None, **overrides
    ) -> Conversation:
        """
        Create private conversation within a room.

        Conversations are always bound to a room - users have private chats
        within the context of a room they're both in.
        """
        if room is None:
            room = await RoomFactory.create(session)

        private_defaults = {
            "room_id": room.id,
            "conversation_type": ConversationType.PRIVATE,
            "max_participants": 2,
        }
        return await cls.create(session, **private_defaults, **overrides)

    @classmethod
    async def create_group_conversation(
        cls, session: AsyncSession, room: Optional[Room] = None, max_participants: int = 5, **overrides
    ) -> Conversation:
        """
        Create group conversation within a room.

        Group chats are small circles within a room.
        """
        if room is None:
            room = await RoomFactory.create(session)

        group_defaults = {
            "room_id": room.id,
            "conversation_type": ConversationType.GROUP,
            "max_participants": max_participants,
        }
        return await cls.create(session, **group_defaults, **overrides)


class MessageFactory(BaseFactory):
    """Factory for creating Message instances."""

    model_class = Message

    @classmethod
    def get_defaults(cls) -> Dict[str, Any]:
        """Default values for Message creation."""
        return {
            "content": "Test message content",
            "sent_at": datetime.now(),
        }

    @classmethod
    async def create_room_message(
        cls, session: AsyncSession, sender: Optional[User] = None, room: Optional[Room] = None, **overrides
    ) -> Message:
        """Create message in a room."""
        if sender is None:
            sender = await UserFactory.create(session)
        if room is None:
            room = await RoomFactory.create(session)

        room_defaults = {
            "sender_user_id": sender.id,
            "room_id": room.id,
        }
        return await cls.create(session, **room_defaults, **overrides)

    @classmethod
    async def create_conversation_message(
        cls,
        session: AsyncSession,
        sender: Optional[User] = None,
        conversation: Optional[Conversation] = None,
        **overrides,
    ) -> Message:
        """Create message in a conversation."""
        if sender is None:
            sender = await UserFactory.create(session)
        if conversation is None:
            conversation = await ConversationFactory.create(session)

        conversation_defaults = {
            "sender_user_id": sender.id,
            "conversation_id": conversation.id,
        }
        return await cls.create(session, **conversation_defaults, **overrides)

    @classmethod
    async def create_reply(
        cls, session: AsyncSession, reply_to: Message, sender: Optional[User] = None, **overrides
    ) -> Message:
        """Create reply to existing message."""
        if sender is None:
            sender = await UserFactory.create(session)

        reply_defaults = {
            "sender_id": sender.id,
            "reply_to_id": reply_to.id,
            "content": f"Reply to: {reply_to.content}",
        }

        # Use same room or conversation as original message
        if reply_to.room_id:
            reply_defaults["room_id"] = reply_to.room_id
        elif reply_to.conversation_id:
            reply_defaults["conversation_id"] = reply_to.conversation_id

        return await cls.create(session, **reply_defaults, **overrides)

    @classmethod
    async def create_message_thread(
        cls,
        session: AsyncSession,
        count: int = 3,
        sender: Optional[User] = None,
        room: Optional[Room] = None,
        **overrides,
    ) -> list[Message]:
        """Create a thread of messages."""
        messages = []

        # Create first message
        first_message = await cls.create_room_message(session, sender=sender, room=room, **overrides)
        messages.append(first_message)

        # Create replies
        for i in range(1, count):
            reply = await cls.create_reply(
                session, reply_to=first_message, sender=sender, content=f"Reply {i} to original message"
            )
            messages.append(reply)

        return messages


class AIFactory(BaseFactory):
    """Factory for creating AIEntity instances."""

    model_class = AIEntity

    @classmethod
    def get_defaults(cls) -> Dict[str, Any]:
        """Default values for AI entity creation."""
        unique_id = str(uuid.uuid4())[:8]
        return {
            "username": f"test_ai_{unique_id}",
            "system_prompt": "You are a helpful AI assistant for testing.",
            "model_name": "gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 512,
            "room_response_strategy": AIResponseStrategy.ROOM_MENTION_ONLY,
            "conversation_response_strategy": AIResponseStrategy.CONV_EVERY_MESSAGE,
            "response_probability": 0.5,
            "cooldown_seconds": None,
            "status": AIEntityStatus.ONLINE,
            "is_active": True,
        }

    @classmethod
    async def create_online(cls, session: AsyncSession, **overrides) -> AIEntity:
        """Create an online AI entity."""
        return await cls.create(session, status=AIEntityStatus.ONLINE, **overrides)

    @classmethod
    async def create_offline(cls, session: AsyncSession, **overrides) -> AIEntity:
        """Create an offline AI entity."""
        return await cls.create(session, status=AIEntityStatus.OFFLINE, **overrides)


# Convenience functions for common scenarios
async def create_test_scenario_basic(session: AsyncSession) -> Dict[str, Any]:
    """Create basic test scenario with user, room, and message."""
    admin = await UserFactory.create_admin(session)
    room = await RoomFactory.create_with_admin(session, admin=admin)
    user = await UserFactory.create_user_with_room(session, room=room)
    message = await MessageFactory.create_room_message(session, sender=user, room=room)

    return {
        "admin": admin,
        "user": user,
        "room": room,
        "message": message,
    }


async def create_test_scenario_conversation(session: AsyncSession) -> Dict[str, Any]:
    """Create test scenario with private conversation."""
    user1 = await UserFactory.create(session, username="user1", email="user1@example.com")
    user2 = await UserFactory.create(session, username="user2", email="user2@example.com")
    conversation = await ConversationFactory.create_private_conversation(session)
    message = await MessageFactory.create_conversation_message(session, sender=user1, conversation=conversation)

    return {
        "user1": user1,
        "user2": user2,
        "conversation": conversation,
        "message": message,
    }


async def create_test_scenario_complex(session: AsyncSession) -> Dict[str, Any]:
    """Create complex test scenario with multiple entities."""
    # Users
    admin = await UserFactory.create_admin(session)
    users = []
    for i in range(3):
        user = await UserFactory.create(session, username=f"user{i}", email=f"user{i}@example.com")
        users.append(user)

    # Rooms
    room1 = await RoomFactory.create_with_admin(session, admin=admin, name="Room 1")
    room2 = await RoomFactory.create_with_admin(session, admin=admin, name="Room 2")

    # Conversations
    private_conv = await ConversationFactory.create_private_conversation(session)
    group_conv = await ConversationFactory.create_group_conversation(session)

    # Messages
    room_messages = await MessageFactory.create_message_thread(session, count=3, sender=users[0], room=room1)
    conv_message = await MessageFactory.create_conversation_message(session, sender=users[1], conversation=private_conv)

    return {
        "admin": admin,
        "users": users,
        "rooms": [room1, room2],
        "conversations": [private_conv, group_conv],
        "room_messages": room_messages,
        "conv_message": conv_message,
    }
