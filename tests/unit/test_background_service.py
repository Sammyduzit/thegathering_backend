"""Unit tests for BackgroundService."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.models.message import Message
from app.models.message_translation import MessageTranslation
from app.services.domain.background_service import BackgroundService


@pytest.mark.unit
class TestBackgroundService:
    """Verifies translation reuse, creation, cleanup and logging helpers."""

    @pytest.fixture
    def service(self):
        translation_service = AsyncMock()
        message_translation_repo = AsyncMock()

        svc = BackgroundService(
            translation_service=translation_service,
            message_translation_repo=message_translation_repo,
        )
        return svc, translation_service, message_translation_repo

    @pytest.mark.asyncio
    async def test_translation_skips_when_disabled(self, service):
        svc, translation_service, repo = service
        message = SimpleNamespace(id=1, content="Hello")

        result = await svc.process_message_translation_background(
            message=message,
            target_languages=["DE", "FR"],
            room_translation_enabled=False,
        )

        assert result == {}
        translation_service.translate_message_content.assert_not_called()
        repo.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_translation_reuses_existing_entries(self, service):
        svc, translation_service, repo = service
        message = SimpleNamespace(id=1, content="Hello")

        repo.get_by_message_and_language.return_value = MessageTranslation(
            message_id=1,
            target_language="DE",
            content="Hallo",
        )

        result = await svc.process_message_translation_background(
            message=message,
            target_languages=["DE"],
        )

        assert result == {"DE": "Hallo"}
        translation_service.translate_message_content.assert_not_called()
        repo.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_translation_creates_new_entries(self, service):
        svc, translation_service, repo = service
        message = SimpleNamespace(id=1, content="Hello")

        repo.get_by_message_and_language.return_value = None
        translation_service.translate_message_content.return_value = {"DE": "Hallo"}

        result = await svc.process_message_translation_background(
            message=message,
            target_languages=["DE"],
        )

        translation_service.translate_message_content.assert_awaited_once()
        repo.create.assert_awaited_once()
        assert result == {"DE": "Hallo"}

    @pytest.mark.asyncio
    async def test_translation_continues_on_errors(self, service):
        svc, translation_service, repo = service
        message = SimpleNamespace(id=1, content="Hello")

        repo.get_by_message_and_language.return_value = None
        from app.interfaces.translator import TranslationError

        translation_service.translate_message_content.side_effect = TranslationError("boom")

        result = await svc.process_message_translation_background(
            message=message,
            target_languages=["DE"],
        )

        assert result == {}
        repo.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleanup_old_translations_delegates_to_repo(self, service):
        svc, _, repo = service
        repo.cleanup_old_translations.return_value = 5

        cleaned = await svc.cleanup_old_translations_background(days_old=10)
        assert cleaned == 5
        repo.cleanup_old_translations.assert_awaited_once_with(10)

    @pytest.mark.asyncio
    async def test_log_user_activity_background_handles_details(self, service):
        svc, *_ = service
        # Should not raise even with custom details
        await svc.log_user_activity_background(user_id=1, activity_type="test", details={"foo": "bar"})

    @pytest.mark.asyncio
    async def test_notify_room_users_background_runs_without_error(self, service):
        svc, *_ = service
        await svc.notify_room_users_background(room_id=1, message="hi", exclude_user_ids=[1, 2])
