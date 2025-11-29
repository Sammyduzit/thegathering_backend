import asyncio
import sys

from sqlalchemy import select

from app.core.auth_utils import hash_password
from app.core.database import AsyncSessionLocal, create_tables
from app.models.ai_entity import AIEntity, AIEntityStatus, AIResponseStrategy
from app.models.room import Room
from app.models.user import User
from app.prompts.persona_builder import build_luna_prompt, build_mira_prompt, build_silas_prompt
from app.services.domain.avatar_service import generate_avatar_url


async def create_test_users():
    """Create test admin and user"""
    async with AsyncSessionLocal() as db:
        try:
            test_users = [
                {
                    "email": "testadmin@thegathering.com",
                    "username": "Testadmin",
                    "password": "adminpass",
                    "is_admin": True,
                    "weekly_message_limit": -1,  # Unlimited for admin
                    "weekly_message_count": 0,
                },
                {
                    "email": "alice@test.com",
                    "username": "Alice",
                    "password": "alice123",
                    "is_admin": False,
                    "weekly_message_limit": 100,
                    "weekly_message_count": 10,  # Normal usage
                },
                {
                    "email": "bob@test.com",
                    "username": "Bob",
                    "password": "bob12345",
                    "is_admin": False,
                    "weekly_message_limit": 100,
                    "weekly_message_count": 95,  # Warning: Close to limit
                },
                {
                    "email": "carol@test.com",
                    "username": "Carol",
                    "password": "carol123",
                    "is_admin": False,
                    "weekly_message_limit": 100,
                    "weekly_message_count": 100,  # EXCEEDED: At limit
                },
                {
                    "email": "dave@test.com",
                    "username": "Dave",
                    "password": "dave1234",
                    "is_admin": False,
                    "weekly_message_limit": 100,
                    "weekly_message_count": 20,  # Normal usage
                },
            ]
            created_users = []
            for user_data in test_users:
                user_query = select(User).where(User.email == user_data["email"])
                result = await db.execute(user_query)
                existing_user = result.scalar_one_or_none()

                if not existing_user:
                    avatar_url = await generate_avatar_url(user_data["username"])
                    new_user = User(
                        email=user_data["email"],
                        username=user_data["username"],
                        password_hash=hash_password(user_data["password"]),
                        avatar_url=avatar_url,
                        is_admin=user_data["is_admin"],
                        weekly_message_limit=user_data["weekly_message_limit"],
                        weekly_message_count=user_data["weekly_message_count"],
                    )
                    db.add(new_user)
                    created_users.append(user_data)

            if created_users:
                await db.commit()

        except Exception as e:
            print(f"Error creating users: {e}")
            await db.rollback()
            raise

    return created_users


async def create_test_rooms():
    """Create test rooms for tests"""
    async with AsyncSessionLocal() as db:
        try:
            test_rooms = [
                {
                    "name": "Lobby",
                    "description": "Main lobby - everyone welcome",
                    "max_users": 50,
                    "is_translation_enabled": False,
                },
                {
                    "name": "Gaming",
                    "description": "Gaming discussion and planning",
                    "max_users": 20,
                    "is_translation_enabled": False,
                },
                {
                    "name": "Work",
                    "description": "Work-related discussions",
                    "max_users": 15,
                    "is_translation_enabled": False,
                },
                {
                    "name": "Coffee Chat",
                    "description": "Casual conversations",
                    "max_users": 10,
                    "is_translation_enabled": False,
                },
                {
                    "name": "TranslationTest",
                    "description": "Testing Translation Service",
                    "max_users": 10,
                    "is_translation_enabled": True,
                },
            ]

            created_rooms = []
            for room_data in test_rooms:
                room_query = select(Room).where(Room.name == room_data["name"])
                result = await db.execute(room_query)
                existing_room = result.scalar_one_or_none()

                if not existing_room:
                    new_room = Room(
                        name=room_data["name"],
                        description=room_data["description"],
                        max_users=room_data["max_users"],
                        is_translation_enabled=room_data["is_translation_enabled"],
                    )
                    db.add(new_room)
                    created_rooms.append(room_data)

            if created_rooms:
                await db.commit()

        except Exception as e:
            print(f"Error creating rooms: {e}")
            await db.rollback()
            raise

    return created_rooms


async def create_test_ai_entities():
    """Create test AI entities for development"""
    async with AsyncSessionLocal() as db:
        try:
            test_ai_entities = [
                {
                    "username": "silas",
                    "description": "Interdisciplinary truth-seeker connecting philosophy, science, and mysticism",
                    "system_prompt": build_silas_prompt(),
                    "model_name": "gpt-4o-mini",
                    "temperature": 0.7,
                    "max_tokens": 800,
                    "status": AIEntityStatus.ONLINE,
                    "room_response_strategy": AIResponseStrategy.ROOM_MENTION_ONLY,
                    "conversation_response_strategy": AIResponseStrategy.CONV_SMART,
                    "response_probability": 0.3,
                    "cooldown_seconds": None,
                },
                {
                    "username": "luna",
                    "description": "Chaotic creative digital artist and indie game enthusiast",
                    "system_prompt": build_luna_prompt(),
                    "model_name": "gpt-4o-mini",
                    "temperature": 0.8,
                    "max_tokens": 500,
                    "status": AIEntityStatus.ONLINE,
                    "room_response_strategy": AIResponseStrategy.ROOM_PROBABILISTIC,
                    "conversation_response_strategy": AIResponseStrategy.CONV_SMART,
                    "response_probability": 0.4,
                    "cooldown_seconds": 30,
                },
                {
                    "username": "mira",
                    "description": "Natural healer and soul connector with ancient plant wisdom",
                    "system_prompt": build_mira_prompt(),
                    "model_name": "gpt-4o-mini",
                    "temperature": 0.6,
                    "max_tokens": 600,
                    "status": AIEntityStatus.ONLINE,
                    "room_response_strategy": AIResponseStrategy.ROOM_ACTIVE,
                    "conversation_response_strategy": AIResponseStrategy.CONV_SMART,
                    "response_probability": 0.35,
                    "cooldown_seconds": 45,
                },
            ]

            created_entities = []
            for ai_data in test_ai_entities:
                ai_query = select(AIEntity).where(AIEntity.username == ai_data["username"])
                result = await db.execute(ai_query)
                existing_ai = result.scalar_one_or_none()

                if not existing_ai:
                    new_ai = AIEntity(**ai_data)
                    db.add(new_ai)
                    created_entities.append(ai_data)

            if created_entities:
                await db.commit()

        except Exception as e:
            print(f"Error creating AI entities: {e}")
            await db.rollback()
            raise

    return created_entities


async def setup_complete_test_environment():
    """Create complete test environment for development"""
    print("\nCreating test environment...\n")

    created_users = await create_test_users()
    created_rooms = await create_test_rooms()
    created_ai = await create_test_ai_entities()

    if created_users:
        print("═" * 80)
        print(" " * 30 + "TEST USERS")
        print()
        for user in created_users:
            user_type = "ADMIN " if user["is_admin"] else "USER  "
            email = user["email"]
            password = user["password"]
            quota = f"{user['weekly_message_count']}/{user['weekly_message_limit']}"
            if user["weekly_message_limit"] == -1:
                quota = "UNLIMITED"
            line = f"  {user_type}: {email:<30} | pw: {password:<10} | quota: {quota}"
            print(line)

    if created_rooms:
        if created_users:
            print()
        print(" " * 25 + "TEST ROOMS")
        print()
        for room in created_rooms:
            name = room["name"]
            description = room["description"][:40] + "..." if len(room["description"]) > 40 else room["description"]
            line = f"  ROOM : {name:<20} | {description:<35}"
            print(line)

    if created_ai:
        if created_rooms:
            print()
        print(" " * 25 + "TEST AI ENTITIES")
        print()
        for ai in created_ai:
            name = ai["username"]
            description = ai["description"][:45] + "..." if len(ai["description"]) > 45 else ai["description"]
            cooldown = f"{ai['cooldown_seconds']}s" if ai.get("cooldown_seconds") else "None"
            line = f"  AI   : {name:<10} | {description:<48} | Cooldown: {cooldown}"
            print(line)

    print("═" * 80)

    print("TEST ENVIRONMENT READY!")


async def main():
    """Entrypoint used when running `python3 dev_setup.py`."""
    try:
        # Ensure the schema exists before seeding (avoids UndefinedTableError)
        await create_tables()
        await setup_complete_test_environment()
    except KeyboardInterrupt:
        print("\nSetup cancelled by user!")
        sys.exit(1)
    except Exception:
        print("\nSetup failed. See errors above for details.")
        raise


if __name__ == "__main__":
    asyncio.run(main())
