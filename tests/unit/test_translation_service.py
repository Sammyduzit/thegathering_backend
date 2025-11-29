"""Unit tests for TranslationService."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.models.message_translation import MessageTranslation
from app.services.domain.translation_service import TranslationService


@pytest.mark.unit
class TestTranslationService:
    """Covers translation workflow and persistence helpers."""

    @pytest.fixture
    def service(self):
        translator = AsyncMock()
        message_repo = AsyncMock()
        translation_repo = AsyncMock()

        svc = TranslationService(
            translator=translator,
            message_repo=message_repo,
            translation_repo=translation_repo,
        )
        return svc, translator, message_repo, translation_repo

    @pytest.mark.asyncio
    async def test_translate_message_content_requires_targets(self, service):
        svc, translator, *_ = service

        result = await svc.translate_message_content(content="hello", target_languages=[])
        assert result == {}
        translator.translate_to_multiple_languages.assert_not_called()

    @pytest.mark.asyncio
    async def test_translate_message_content_invokes_translator(self, service):
        svc, translator, *_ = service
        translator.translate_to_multiple_languages.return_value = {"DE": "Hallo"}

        result = await svc.translate_message_content(
            content="Hello",
            target_languages=["DE"],
            source_language="EN",
        )

        assert result == {"DE": "Hallo"}
        translator.translate_to_multiple_languages.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_translate_message_content_handles_exception(self, service):
        svc, translator, *_ = service
        translator.translate_to_multiple_languages.side_effect = RuntimeError("boom")

        result = await svc.translate_message_content(content="Hello", target_languages=["DE"])
        assert result == {}

    @pytest.mark.asyncio
    async def test_create_message_translations_returns_empty_when_no_translations(self, service):
        svc, *_ = service

        result = await svc.create_message_translations(message_id=1, translations={})
        assert result == []

    @pytest.mark.asyncio
    async def test_create_message_translations_persists_bulk(self, service):
        svc, *_, translation_repo = service
        translation_repo.bulk_create_translations.return_value = [
            MessageTranslation(message_id=1, target_language="DE", content="Hallo")
        ]

        result = await svc.create_message_translations(
            message_id=1,
            translations={"DE": "Hallo"},
        )

        translation_repo.bulk_create_translations.assert_awaited_once()
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_translate_and_store_message_full_flow(self, service):
        svc, translator, _, translation_repo = service
        translator.translate_to_multiple_languages.return_value = {"DE": "Hallo"}
        translation_repo.bulk_create_translations.return_value = [
            MessageTranslation(message_id=1, target_language="DE", content="Hallo")
        ]

        count = await svc.translate_and_store_message(
            message_id=1,
            content="Hello",
            target_languages=["DE"],
        )

        assert count == 1
        translator.translate_to_multiple_languages.assert_awaited_once()
        translation_repo.bulk_create_translations.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_translate_and_store_message_handles_failure(self, service):
        svc, translator, *_ = service
        translator.translate_to_multiple_languages.side_effect = RuntimeError("fail")

        count = await svc.translate_and_store_message(
            message_id=1,
            content="Hello",
            target_languages=["DE"],
        )

        assert count == 0

    @pytest.mark.asyncio
    async def test_get_message_translation_returns_content(self, service):
        svc, *_, translation_repo = service
        translation_repo.get_by_message_and_language.return_value = SimpleNamespace(content="Hola")

        result = await svc.get_message_translation(message_id=1, target_language="es")

        translation_repo.get_by_message_and_language.assert_awaited_once()
        assert result == "Hola"

    @pytest.mark.asyncio
    async def test_get_all_message_translations_returns_dict(self, service):
        svc, *_, translation_repo = service
        translation_repo.get_by_message_id.return_value = [
            MessageTranslation(message_id=1, target_language="FR", content="Salut")
        ]

        result = await svc.get_all_message_translations(1)
        assert result == {"FR": "Salut"}

    @pytest.mark.asyncio
    async def test_delete_message_translations_delegates(self, service):
        svc, *_, translation_repo = service
        translation_repo.delete_by_message_id.return_value = 3

        deleted = await svc.delete_message_translations(1)
        assert deleted == 3
        translation_repo.delete_by_message_id.assert_awaited_once_with(1)
