from abc import abstractmethod

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User

from .base_repository import BaseRepository


class IUserRepository(BaseRepository[User]):
    """Abstract interface for User repository."""

    @abstractmethod
    async def get_by_email(self, email: str) -> User | None:
        """Get user by email address."""
        pass

    @abstractmethod
    async def get_by_username(self, username: str) -> User | None:
        """Get user by username."""
        pass

    @abstractmethod
    async def get_active_users(self) -> list[User]:
        """Get all active users."""
        pass

    @abstractmethod
    async def get_users_in_room(self, room_id: int) -> list[User]:
        """Get all users currently in a specific room."""
        pass

    @abstractmethod
    async def email_exists(self, email: str) -> bool:
        """Check if email already exists."""
        pass

    @abstractmethod
    async def username_exists(self, username: str) -> bool:
        """Check if username already exists."""
        pass

    @abstractmethod
    async def get_quota_exceeded_users(self) -> list[User]:
        """Get all users who have exceeded their weekly message quota."""
        pass


class UserRepository(IUserRepository):
    """SQLAlchemy implementation of User repository."""

    def __init__(self, db: AsyncSession):
        """
        Initialize with async database session.
        :param db: SQLAlchemy async database session
        """
        super().__init__(db)

    async def get_by_id(self, id: int) -> User | None:
        """Get user by ID."""
        query = select(User).where(User.id == id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> User | None:
        """Get user by email address."""
        query = select(User).where(User.email == email)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_by_username(self, username: str) -> User | None:
        """Get user by username."""
        query = select(User).where(User.username == username)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_all(self, limit: int = 100, offset: int = 0) -> list[User]:
        """Get all users with pagination."""
        query = select(User).limit(limit).offset(offset)
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_active_users(self) -> list[User]:
        """Get all active users."""
        query = select(User).where(User.is_active.is_(True))
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_users_in_room(self, room_id: int) -> list[User]:
        """Get all users currently in a specific room."""
        query = select(User).where(and_(User.current_room_id == room_id, User.is_active.is_(True)))
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def delete(self, id: int) -> bool:
        """Delete user by ID (soft delete - set inactive)."""
        user = await self.get_by_id(id)
        if user:
            user.is_active = False
            await self.db.commit()
            return True
        return False

    async def exists(self, id: int) -> bool:
        """Check if user exists by ID."""
        return await self._check_exists_where(User.id == id)

    async def email_exists(self, email: str) -> bool:
        """Check if email already exists."""
        return await self._check_exists_where(User.email == email)

    async def username_exists(self, username: str) -> bool:
        """Check if username already exists."""
        return await self._check_exists_where(User.username == username)

    async def get_quota_exceeded_users(self) -> list[User]:
        """Get all users who have exceeded their weekly message quota."""
        query = select(User).where(
            and_(
                User.weekly_message_count >= User.weekly_message_limit,
                User.weekly_message_limit != -1,  # Exclude unlimited users
                User.is_active.is_(True),  # Only active users
            )
        )
        result = await self.db.execute(query)
        return list(result.scalars().all())
