from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

from app.core.config import settings

engine = create_async_engine(
    settings.database_url.replace("postgresql://", "postgresql+asyncpg://"),
    pool_size=10,
    max_overflow=5,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=settings.debug,
)

AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()


async def get_db():
    """Async database session dependency"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def create_tables():
    """Create all database tables"""
    async with engine.begin() as conn:
        # Enable pgvector extension before creating tables
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    print("All tables created")


async def drop_tables():
    """Drop all database tables"""
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all, checkfirst=True)
    except (IntegrityError, OperationalError) as e:
        print(f"FK constraint issue, using reflect method: {e}")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.reflect)
            await conn.run_sync(Base.metadata.drop_all)

    print("All tables dropped")
