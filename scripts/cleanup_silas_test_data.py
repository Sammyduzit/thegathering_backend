#!/usr/bin/env python3
"""
Script to cleanup all test conversations and memories created by the AI entity "silas".

This script deletes:
- All conversations where silas is a participant
- All AI memories created by silas
- Messages and conversation participants (via CASCADE)

Usage:
    python cleanup_silas_test_data.py

Prerequisites:
- Database connection configured in .env
- AI entity "silas" must exist (will skip if not found)
"""

import asyncio

from sqlalchemy import delete, func, select

from app.core.database import AsyncSessionLocal
from app.models.ai_entity import AIEntity
from app.models.ai_memory import AIMemory
from app.models.conversation import Conversation
from app.models.conversation_participant import ConversationParticipant
from app.models.message import Message


async def cleanup_silas_data():
    """
    Delete all conversations and memories associated with the AI entity "silas".

    This function:
    1. Finds the silas AI entity
    2. Counts existing data (conversations, messages, memories)
    3. Deletes all AI memories from silas
    4. Deletes all conversations with silas as participant (CASCADE deletes messages & participants)
    5. Commits the transaction
    6. Displays statistics
    """
    async with AsyncSessionLocal() as db:
        try:
            print("\n" + "═" * 68)
            print(" " * 18 + "SILAS TEST DATA CLEANUP")
            print("═" * 68 + "\n")

            # ═══════════════════════════════════════════════════════════════
            # STEP 1: Find silas entity
            # ═══════════════════════════════════════════════════════════════
            print("🔍 Searching for AI entity 'silas'...")
            silas = await db.scalar(select(AIEntity).where(AIEntity.username == "silas"))

            if not silas:
                print("⚠️  AI entity 'silas' not found in database.")
                print("   Nothing to clean up. Exiting.\n")
                return

            print(f"✓ Found silas (ID: {silas.id})\n")

            # ═══════════════════════════════════════════════════════════════
            # STEP 2: Count existing data before deletion
            # ═══════════════════════════════════════════════════════════════
            print("📊 Analyzing data to be deleted...")

            # Count conversations with silas
            conversations_subquery = (
                select(ConversationParticipant.conversation_id)
                .where(ConversationParticipant.ai_entity_id == silas.id)
                .distinct()
            )
            conversation_count = await db.scalar(
                select(func.count()).select_from(Conversation).where(Conversation.id.in_(conversations_subquery))
            )

            # Count messages in these conversations
            message_count = await db.scalar(
                select(func.count()).select_from(Message).where(Message.conversation_id.in_(conversations_subquery))
            )

            # Count conversation participants in these conversations
            participant_count = await db.scalar(
                select(func.count())
                .select_from(ConversationParticipant)
                .where(ConversationParticipant.conversation_id.in_(conversations_subquery))
            )

            # Count AI memories
            memory_count = await db.scalar(
                select(func.count()).select_from(AIMemory).where(AIMemory.entity_id == silas.id)
            )

            print(f"   Conversations: {conversation_count}")
            print(f"   Messages: {message_count}")
            print(f"   Participants: {participant_count}")
            print(f"   AI Memories: {memory_count}\n")

            if conversation_count == 0 and memory_count == 0:
                print("✓ No data to delete. Database is already clean.\n")
                return

            # ═══════════════════════════════════════════════════════════════
            # STEP 3: Delete AI Memories
            # ═══════════════════════════════════════════════════════════════
            print("🗑️  Deleting AI memories...")
            delete_memories_stmt = delete(AIMemory).where(AIMemory.entity_id == silas.id)
            result_memories = await db.execute(delete_memories_stmt)
            deleted_memories = result_memories.rowcount
            print(f"   ✓ Deleted {deleted_memories} AI memories\n")

            # ═══════════════════════════════════════════════════════════════
            # STEP 4: Delete Conversations (CASCADE deletes messages & participants)
            # ═══════════════════════════════════════════════════════════════
            print("🗑️  Deleting conversations...")
            delete_conversations_stmt = delete(Conversation).where(Conversation.id.in_(conversations_subquery))
            result_conversations = await db.execute(delete_conversations_stmt)
            deleted_conversations = result_conversations.rowcount
            print(f"   ✓ Deleted {deleted_conversations} conversations")
            print(f"   ✓ CASCADE deleted {message_count} messages")
            print(f"   ✓ CASCADE deleted {participant_count} participants\n")

            # ═══════════════════════════════════════════════════════════════
            # STEP 5: Commit transaction
            # ═══════════════════════════════════════════════════════════════
            await db.commit()

            # ═══════════════════════════════════════════════════════════════
            # STEP 6: Summary
            # ═══════════════════════════════════════════════════════════════
            print("═" * 68)
            print(" " * 25 + "SUMMARY")
            print("═" * 68)
            print("\n  Successfully deleted:")
            print(f"    • {deleted_conversations} conversations")
            print(f"    • {message_count} messages (via CASCADE)")
            print(f"    • {participant_count} participants (via CASCADE)")
            print(f"    • {deleted_memories} AI memories")
            print("\n" + "═" * 68)
            print("  CLEANUP COMPLETE!")
            print("═" * 68 + "\n")
            print("💡 Next step: Run 'python create_test_conversations.py' to recreate test data\n")

        except Exception as e:
            print(f"\n✗ Error during cleanup: {e}")
            print("  Rolling back transaction...\n")
            await db.rollback()
            raise


async def main():
    """Run the cleanup script."""
    print("\n" + "=" * 68)
    print(" " * 15 + "SILAS TEST DATA CLEANUP UTILITY")
    print("=" * 68 + "\n")
    print("⚠️  WARNING: This will permanently delete:")
    print("   • All conversations with silas as participant")
    print("   • All messages in these conversations")
    print("   • All AI memories created by silas\n")

    # Safety confirmation (can be commented out for non-interactive use)
    try:
        response = input("Continue? (yes/no): ").strip().lower()
        if response != "yes":
            print("\n✗ Cleanup cancelled by user.\n")
            return
    except (EOFError, KeyboardInterrupt):
        print("\n\n✗ Cleanup cancelled by user.\n")
        return

    await cleanup_silas_data()


if __name__ == "__main__":
    asyncio.run(main())
