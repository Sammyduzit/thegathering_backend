from abc import abstractmethod

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.room import Room
from app.models.user import User
from app.repositories.base_repository import BaseRepository


class IRoomRepository(BaseRepository[Room]):
    """Abstract interface for Room repository."""

    @abstractmethod
    async def get_active_rooms(self) -> list[Room]:
        """Get all active rooms."""
        pass

    @abstractmethod
    async def get_by_name(self, name: str) -> Room | None:
        """Get room by name."""
        pass

    @abstractmethod
    async def name_exists(self, name: str, exclude_room_id: int | None = None) -> bool:
        """Check if room name already exists."""
        pass

    @abstractmethod
    async def get_user_count(self, room_id: int) -> int:
        """Get count of users currently in room."""
        pass

    @abstractmethod
    async def get_users_in_room(self, room_id: int) -> list[User]:
        """Get all users currently in a specific room."""
        pass

    @abstractmethod
    async def soft_delete(self, room_id: int) -> bool:
        """Soft delete room (set inactive)."""
        pass


class RoomRepository(IRoomRepository):
    """SQLAlchemy implementation of Room repository."""

    def __init__(self, db: AsyncSession):
        """
        Initialize with async database session.
        :param db: SQLAlchemy async database session
        """
        super().__init__(db)

    async def get_by_id(self, id: int) -> Room | None:
        """Get room by ID."""
        query = select(Room).where(and_(Room.id == id, Room.is_active.is_(True)))
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_active_rooms(self) -> list[Room]:
        """Get all active rooms."""
        query = select(Room).where(Room.is_active.is_(True))
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_by_name(self, name: str) -> Room | None:
        """Get room by name."""
        query = select(Room).where(and_(Room.name == name, Room.is_active.is_(True)))
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def name_exists(self, name: str, exclude_room_id: int | None = None) -> bool:
        """Check if room name already exists."""
        query = select(Room).where(and_(Room.name == name, Room.is_active.is_(True)))

        if exclude_room_id:
            query = query.where(Room.id != exclude_room_id)

        result = await self.db.execute(query)
        existing_room = result.scalar_one_or_none()
        return existing_room is not None

    async def get_user_count(self, room_id: int) -> int:
        """Get count of users currently in room."""
        user_count_query = select(func.count(User.id)).where(User.current_room_id == room_id)
        result = await self.db.execute(user_count_query)
        return result.scalar() or 0

    async def get_users_in_room(self, room_id: int) -> list[User]:
        """Get all users currently in a specific room."""
        query = (
            select(User).where(and_(User.current_room_id == room_id, User.is_active.is_(True))).order_by(User.username)
        )

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_all(self, limit: int = 100, offset: int = 0) -> list[Room]:
        """Get all rooms with pagination."""
        query = select(Room).limit(limit).offset(offset)
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def delete(self, id: int) -> bool:
        """Hard delete room by ID."""
        room = await self.get_by_id(id)
        if room:
            self.db.delete(room)
            await self.db.commit()
            return True
        return False

    async def soft_delete(self, room_id: int) -> bool:
        """Soft delete room (set inactive)."""
        room = await self.get_by_id(room_id)
        if room:
            room.is_active = False
            await self.db.commit()
            return True
        return False

    async def exists(self, id: int) -> bool:
        """Check if room exists by ID (active rooms only)."""
        return await self._check_exists_where(and_(Room.id == id, Room.is_active.is_(True)))
