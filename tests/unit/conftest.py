"""
Unit test fixtures with SQLite and mocked dependencies.

Modernized for pytest-asyncio 1.2.0 (October 2025):
- No event_loop fixture (removed in 1.x)
- Uses loop_scope="function" for all fixtures
- Clean separation: SQLite for speed, mocks for isolation

Unit tests should be:
- Fast (< 5s total)
- Isolated (no external dependencies)
- Deterministic (no flakiness)
"""

import os
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import StaticPool

from app.core.database import Base
from app.interfaces.translator import TranslatorInterface
from app.models.ai_entity import AIEntity, AIEntityStatus
from app.models.user import User

# Force unit test environment
os.environ["TEST_TYPE"] = "unit"


# ============================================================================
# AI-Specific Fixtures for Unit Tests
# ============================================================================


@pytest.fixture
def sample_ai_entity(sample_ai_entity_data):
    """Create sample AI entity for unit tests."""
    return AIEntity(
        id=1,
        **sample_ai_entity_data,
        status=AIEntityStatus.ONLINE,
        temperature=0.7,
        max_tokens=1024,
    )


@pytest.fixture
def sample_user(sample_user_data):
    """Create sample user for unit tests."""
    return User(
        id=2,
        username=sample_user_data["username"],
        email=sample_user_data["email"],
    )


@pytest.fixture
def mock_ai_provider():
    """Create mock AI provider for testing."""
    return AsyncMock()


@pytest.fixture
def mock_context_service():
    """Create mock AI context service for testing."""
    return AsyncMock()


@pytest.fixture
def mock_message_repo():
    """Create mock message repository for testing."""
    return AsyncMock()


@pytest.fixture
def mock_memory_repo():
    """Create mock AI memory repository for testing."""
    return AsyncMock()


# ============================================================================
# Database Fixtures (SQLite in-memory)
# ============================================================================


@pytest_asyncio.fixture(loop_scope="function")
async def unit_engine():
    """
    SQLite in-memory engine for unit tests.

    Function-scoped for maximum isolation.
    Uses StaticPool to maintain in-memory database during test.
    """
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        echo=False,
        connect_args={"check_same_thread": False},
    )

    # Enable foreign key constraints for SQLite
    @event.listens_for(engine.sync_engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    # Create schema
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Cleanup
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(unit_engine):
    """
    Isolated database session for each unit test.

    Each test gets a fresh session. Factories handle their own commits.
    Session is closed after test completes.
    """
    async with AsyncSession(unit_engine, expire_on_commit=False) as session:
        yield session
        # Session cleanup happens automatically at context exit


# ============================================================================
# Mock Fixtures
# ============================================================================


@pytest_asyncio.fixture
async def mock_translator():
    """Mock translator implementing TranslatorInterface."""
    return AsyncMock(spec=TranslatorInterface)


# ============================================================================
# Quick Test Data Fixtures (for simple tests)
# ============================================================================


@pytest_asyncio.fixture
async def test_user(db_session, user_factory):
    """Quick access to a test user for simple unit tests."""
    return await user_factory.create(db_session)


@pytest_asyncio.fixture
async def test_admin(db_session, user_factory):
    """Quick access to an admin user for simple unit tests."""
    return await user_factory.create_admin(db_session)


@pytest_asyncio.fixture
async def test_room(db_session, room_factory):
    """Quick access to a test room for simple unit tests."""
    return await room_factory.create(db_session)


@pytest_asyncio.fixture
async def test_message(db_session, message_factory, test_user, test_room):
    """Quick access to a test message for simple unit tests."""
    return await message_factory.create_room_message(db_session, sender=test_user, room=test_room)


# ============================================================================
# AI Cooldown Test Fixtures
# ============================================================================


@pytest_asyncio.fixture
async def async_db_session(unit_engine):
    """Alias for db_session for AI cooldown tests."""
    async with AsyncSession(unit_engine, expire_on_commit=False) as session:
        yield session


@pytest_asyncio.fixture
async def created_ai_entity(async_db_session, sample_ai_entity_data):
    """Create AI entity for cooldown tests."""
    from sqlalchemy import select

    ai_entity = AIEntity(
        **sample_ai_entity_data,
        status=AIEntityStatus.ONLINE,
        temperature=0.7,
        max_tokens=1024,
    )
    async_db_session.add(ai_entity)
    await async_db_session.commit()

    ai_entity = await async_db_session.scalar(select(AIEntity).where(AIEntity.id == ai_entity.id))
    async_db_session.expunge(ai_entity)
    return ai_entity


@pytest_asyncio.fixture
async def created_room(async_db_session, sample_room_data):
    """Create room for cooldown tests."""
    from sqlalchemy import select

    from app.models.room import Room

    room = Room(**sample_room_data)
    async_db_session.add(room)
    await async_db_session.commit()

    room = await async_db_session.scalar(select(Room).where(Room.id == room.id))
    async_db_session.expunge(room)
    return room


@pytest_asyncio.fixture
async def created_conversation(async_db_session, created_room, test_user):
    """Create conversation for cooldown tests."""
    from sqlalchemy import select

    from app.models.conversation import Conversation, ConversationType

    conversation = Conversation(
        room_id=created_room.id,
        conversation_type=ConversationType.PRIVATE,
        max_participants=2,
    )
    async_db_session.add(conversation)
    await async_db_session.commit()

    conversation = await async_db_session.scalar(select(Conversation).where(Conversation.id == conversation.id))
    async_db_session.expunge(conversation)
    return conversation
