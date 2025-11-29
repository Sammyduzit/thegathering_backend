from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from sqlalchemy import exists, select
from sqlalchemy.ext.asyncio import AsyncSession

T = TypeVar("T")


class BaseRepository(ABC, Generic[T]):
    """Abstract base repository providing common CRUD operations."""

    def __init__(self, db: AsyncSession):
        """
        Initialize repository with async database session.
        :param db: SQLAlchemy async database session
        """
        self.db = db

    async def _check_exists_where(self, *where_clauses) -> bool:
        """
        Helper: Check existence with given WHERE clauses using SELECT EXISTS.

        :param where_clauses: SQLAlchemy WHERE clause expressions
        :return: True if entity exists, False otherwise
        """
        exists_query = select(exists().where(*where_clauses))
        exists_result = await self.db.scalar(exists_query)
        return exists_result or False

    @abstractmethod
    async def get_by_id(self, id: int) -> T | None:
        """
        Get entity by ID.
        :param id: Entity ID
        :return: Entity or None if not found
        """
        pass

    @abstractmethod
    async def get_all(self, limit: int = 100, offset: int = 0) -> list[T]:
        """
        Get all entities with pagination.
        :param limit: Maximum number of entities to return
        :param offset: Number of entities to skip
        :return: List of entities
        """
        pass

    async def create(self, entity: T) -> T:
        """
        Create new entity.

        Default implementation for standard CRUD.
        Can be overridden in subclasses for custom logic.

        :param entity: Entity to create
        :return: Created entity with generated ID
        """
        self.db.add(entity)
        await self.db.commit()
        await self.db.refresh(entity)
        return entity

    async def update(self, entity: T) -> T:
        """
        Update existing entity.

        Default implementation for standard CRUD.
        Can be overridden in subclasses for custom logic.

        :param entity: Entity to update
        :return: Updated entity
        """
        await self.db.commit()
        await self.db.refresh(entity)
        return entity

    @abstractmethod
    async def delete(self, id: int) -> bool:
        """
        Delete entity by ID.
        :param id: Entity ID to delete
        :return: True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def exists(self, id: int) -> bool:
        """
        Check if entity exists by ID.
        :param id: Entity ID to check
        :return: True if exists, False otherwise
        """
        pass
