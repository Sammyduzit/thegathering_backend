"""
Script to delete all room messages from a specific room.
Usage: python delete_room_messages.py <room_name>
Example: python delete_room_messages.py TranslationTest
"""

import argparse
import asyncio

from sqlalchemy import delete, select

from app.core.database import AsyncSessionLocal
from app.models.message import Message
from app.models.room import Room


async def delete_room_messages(room_name: str):
    """Delete all messages from a specific room by room name."""
    async with AsyncSessionLocal() as session:
        # Find the room by name
        result = await session.execute(select(Room).where(Room.name == room_name))
        room = result.scalar_one_or_none()

        if not room:
            print(f"❌ Room '{room_name}' not found!")
            return

        print(f"✓ Found room: {room.name} (ID: {room.id})")

        # Count messages before deletion
        count_result = await session.execute(select(Message).where(Message.room_id == room.id))
        messages_count = len(count_result.scalars().all())

        if messages_count == 0:
            print(f"ℹ️  No messages found in room '{room_name}'")
            return

        print(f"⚠️  Found {messages_count} message(s) in room '{room_name}'")

        # Delete all messages from this room
        await session.execute(delete(Message).where(Message.room_id == room.id))
        await session.commit()

        print(f"✅ Successfully deleted {messages_count} message(s) from room '{room_name}'")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Delete all messages from a specific room")
    parser.add_argument("room_name", type=str, help="Name of the room to delete messages from")
    args = parser.parse_args()

    print(f"Starting deletion of messages from room: {args.room_name}")
    print("-" * 60)

    await delete_room_messages(args.room_name)

    print("-" * 60)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
