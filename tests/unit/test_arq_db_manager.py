"""Tests for ARQ Database Session Manager."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.arq_db_manager import ARQDatabaseManager, db_session_context


@pytest.mark.unit
@pytest.mark.asyncio
async def test_arq_db_manager_connect():
    """Test ARQ database manager connects successfully."""
    manager = ARQDatabaseManager()

    await manager.connect()

    assert manager.session_factory is not None
    assert manager.scoped_session is not None

    await manager.disconnect()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_arq_db_manager_get_session():
    """Test getting a job-scoped session."""
    manager = ARQDatabaseManager()
    await manager.connect()

    # Set context for job isolation
    db_session_context.set("test-job-123")

    async for session in manager.get_session():
        assert isinstance(session, AsyncSession)
        break

    await manager.disconnect()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_arq_db_manager_job_isolation():
    """Test that different job contexts get isolated sessions."""
    manager = ARQDatabaseManager()
    await manager.connect()

    # Job 1
    db_session_context.set("job-1")
    async for session1 in manager.get_session():
        session1_id = id(session1)
        break

    # Job 2
    db_session_context.set("job-2")
    async for session2 in manager.get_session():
        session2_id = id(session2)
        break

    # Different contexts should yield different sessions
    assert session1_id != session2_id

    await manager.disconnect()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_arq_db_manager_raises_without_connect():
    """Test that get_session raises if not connected."""
    manager = ARQDatabaseManager()

    with pytest.raises(RuntimeError, match="ARQDatabaseManager not connected"):
        async for _ in manager.get_session():
            pass
