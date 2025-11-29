"""ARQ Database Session Manager with contextvars for job isolation."""

from contextvars import ContextVar
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_scoped_session, async_sessionmaker

from app.core.database import engine

# Context variable for ARQ job-isolated sessions
db_session_context: ContextVar[str | None] = ContextVar("db_session_context", default=None)


class ARQDatabaseManager:
    """Manages database sessions for ARQ workers with job isolation."""

    def __init__(self):
        self.session_factory: async_sessionmaker[AsyncSession] | None = None
        self.scoped_session: async_scoped_session[AsyncSession] | None = None

    async def connect(self) -> None:
        """Initialize session factory and scoped session on worker startup."""
        self.session_factory = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )

        self.scoped_session = async_scoped_session(
            session_factory=self.session_factory,
            scopefunc=lambda: db_session_context.get(),
        )

    async def disconnect(self) -> None:
        """Clean up sessions and close engine on worker shutdown."""
        if self.scoped_session:
            await self.scoped_session.remove()

        await engine.dispose()

    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a job-scoped database session for ARQ tasks."""
        if not self.scoped_session:
            raise RuntimeError("ARQDatabaseManager not connected. Call connect() first.")

        session = self.scoped_session()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
