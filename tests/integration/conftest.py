"""
Integration test fixtures with PostgreSQL and real services.

Modernized for pytest-asyncio 1.2.0 (October 2025):
- No event_loop fixture (removed in 1.x)
- Uses loop_scope="function" for all fixtures
- NullPool for PostgreSQL (prevents asyncpg event loop binding issues)

Integration tests verify:
- Real database operations with PostgreSQL
- Database constraints and transactions
- Real service interactions
- NO HTTP layer (use E2E tests for that)

Scope Strategy:
- All fixtures: function-scoped (maximum isolation)
- Engine: NullPool (no connection pooling, prevents asyncpg issues)
- Session: Transaction rollback (test isolation)
"""

import os
from concurrent.futures import ThreadPoolExecutor

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.core.database import Base
from app.implementations.deepl_translator import DeepLTranslator
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.message_repository import MessageRepository
from app.repositories.message_translation_repository import MessageTranslationRepository
from app.repositories.room_repository import RoomRepository
from app.repositories.user_repository import UserRepository
from app.services.domain.background_service import BackgroundService
from app.services.domain.conversation_service import ConversationService
from app.services.domain.room_service import RoomService
from app.services.domain.translation_service import TranslationService

# Force integration test environment
os.environ["TEST_TYPE"] = "integration"


# ============================================================================
# Database Fixtures (PostgreSQL with NullPool)
# ============================================================================


@pytest_asyncio.fixture(loop_scope="function")
async def integration_engine():
    """
    PostgreSQL engine for integration tests.

    Uses NullPool to prevent asyncpg event loop binding issues.
    Schema is created/dropped per test for complete isolation.
    """
    # Validate PostgreSQL is configured
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        pytest.skip(
            "Integration tests require DATABASE_URL environment variable.\n"
            "Set DATABASE_URL in .env.test or environment:\n"
            "  DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/the_gathering_test"
        )

    if not DATABASE_URL.startswith(("postgresql://", "postgresql+asyncpg://")):
        pytest.skip(
            f"Integration tests require PostgreSQL, got: {DATABASE_URL}\n"
            "Unit tests use SQLite, Integration tests need PostgreSQL for production parity."
        )

    # Convert postgresql:// to postgresql+asyncpg:// if needed
    if DATABASE_URL.startswith("postgresql://") and not DATABASE_URL.startswith("postgresql+asyncpg://"):
        DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

    engine = create_async_engine(
        DATABASE_URL,
        poolclass=NullPool,  # Essential for pytest + asyncpg
        echo=False,
    )

    # Create schema
    async with engine.begin() as conn:
        # Enable pgvector extension before creating tables
        from sqlalchemy import text

        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(integration_engine):
    """
    Isolated database session for each integration test.

    Each test gets a fresh session. Factories handle their own commits.
    Session is closed after test completes.
    """
    async with AsyncSession(integration_engine, expire_on_commit=False) as session:
        yield session
        # Session cleanup happens automatically at context exit


# ============================================================================
# Repository Fixtures (Real Implementations)
# ============================================================================


@pytest_asyncio.fixture
async def user_repo(db_session):
    """Real UserRepository with PostgreSQL."""
    return UserRepository(db_session)


@pytest_asyncio.fixture
async def room_repo(db_session):
    """Real RoomRepository with PostgreSQL."""
    return RoomRepository(db_session)


@pytest_asyncio.fixture
async def message_repo(db_session):
    """Real MessageRepository with PostgreSQL."""
    return MessageRepository(db_session)


@pytest_asyncio.fixture
async def conversation_repo(db_session):
    """Real ConversationRepository with PostgreSQL."""
    return ConversationRepository(db_session)


@pytest_asyncio.fixture
async def message_translation_repo(db_session):
    """Real MessageTranslationRepository with PostgreSQL."""
    return MessageTranslationRepository(db_session)


# ============================================================================
# Service Fixtures (Real Implementations)
# ============================================================================


@pytest_asyncio.fixture
async def deepl_translator():
    """
    Real DeepL translator (skip tests if no API key available).

    Integration tests that require translation will be skipped
    if DEEPL_API_KEY environment variable is not set.
    """
    if not settings.deepl_api_key:
        pytest.skip("DeepL API key not available for integration tests")

    executor = ThreadPoolExecutor(max_workers=2)
    translator = DeepLTranslator(api_key=settings.deepl_api_key, executor=executor)

    yield translator

    await translator.dispose()


@pytest_asyncio.fixture
async def translation_service(deepl_translator, message_repo, message_translation_repo):
    """Real TranslationService with real DeepL API."""
    return TranslationService(
        translator=deepl_translator,
        message_repo=message_repo,
        translation_repo=message_translation_repo,
    )


@pytest_asyncio.fixture
async def room_service(
    room_repo,
    user_repo,
    message_repo,
    conversation_repo,
    message_translation_repo,
    translation_service,
):
    """Real RoomService with all real dependencies."""
    return RoomService(
        room_repo=room_repo,
        user_repo=user_repo,
        message_repo=message_repo,
        conversation_repo=conversation_repo,
        message_translation_repo=message_translation_repo,
        translation_service=translation_service,
    )


@pytest_asyncio.fixture
async def conversation_service(conversation_repo, message_repo, user_repo, room_repo, translation_service):
    """Real ConversationService with all real dependencies."""
    return ConversationService(
        conversation_repo=conversation_repo,
        message_repo=message_repo,
        user_repo=user_repo,
        room_repo=room_repo,
        translation_service=translation_service,
    )


@pytest_asyncio.fixture
async def background_service(translation_service, message_translation_repo):
    """Real BackgroundService with real dependencies."""
    return BackgroundService(
        translation_service=translation_service,
        message_translation_repo=message_translation_repo,
    )
