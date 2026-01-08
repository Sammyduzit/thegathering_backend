"""Unit tests for BackgroundService."""

import pytest

from app.services.domain.background_service import BackgroundService


@pytest.mark.unit
class TestBackgroundService:
    """Verifies activity logging and notification helpers."""

    @pytest.fixture
    def service(self):
        return BackgroundService()

    @pytest.mark.asyncio
    async def test_log_user_activity_background_handles_details(self, service):
        # Should not raise even with custom details
        await service.log_user_activity_background(user_id=1, activity_type="test", details={"foo": "bar"})

    @pytest.mark.asyncio
    async def test_notify_room_users_background_runs_without_error(self, service):
        await service.notify_room_users_background(room_id=1, message="hi", exclude_user_ids=[1, 2])
