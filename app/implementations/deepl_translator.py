"""
DeepL translator implementation.

This module provides a concrete implementation of the TranslatorInterface
using the DeepL translation service API.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import deepl
import structlog

from app.interfaces.translator import TranslationError, TranslatorInterface

logger = structlog.get_logger(__name__)


class DeepLTranslator(TranslatorInterface):
    """DeepL API implementation of the translator interface."""

    def __init__(self, api_key: str, executor: ThreadPoolExecutor | None = None):
        """
        Initialize DeepL translator.

        :param api_key: DeepL API key
        :param executor: Thread pool executor for async operations (optional)
        """
        if not api_key:
            raise ValueError("DeepL API key is required")

        self.client = deepl.Translator(api_key)
        self.executor = executor
        self._supported_languages: list[str] | None = None

    async def translate_text(self, text: str, target_language: str, source_language: str | None = None) -> str:
        """Translate text using DeepL API."""
        if not text.strip():
            return ""

        try:
            # Run DeepL API call in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor, self._sync_translate_text, text, target_language, source_language
            )
            return result.text
        except Exception as e:
            logger.error(f"DeepL translation failed: {e}")
            raise TranslationError(f"Translation failed: {str(e)}", e)

    def _sync_translate_text(
        self, text: str, target_language: str, source_language: str | None = None
    ) -> deepl.TextResult:
        """Synchronous wrapper for DeepL translate_text call."""
        return self.client.translate_text(
            text, target_lang=target_language.upper(), source_lang=source_language.upper() if source_language else None
        )

    async def translate_to_multiple_languages(
        self, text: str, target_languages: list[str], source_language: str | None = None
    ) -> dict[str, str]:
        """Translate text to multiple languages concurrently."""
        if not text.strip():
            return {}

        if not target_languages:
            return {}

        try:
            # Create translation tasks for all target languages
            tasks = []
            for lang in target_languages:
                task = self.translate_text(text, lang, source_language)
                tasks.append(task)

            # Execute all translations concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Build result dictionary, handling any exceptions
            translations = {}
            for i, result in enumerate(results):
                lang = target_languages[i]
                if isinstance(result, Exception):
                    logger.warning(f"Translation to {lang} failed: {result}")
                    # Skip failed translations rather than failing entirely
                    continue
                translations[lang] = result

            return translations

        except Exception as e:
            logger.error(f"Multiple language translation failed: {e}")
            raise TranslationError(f"Multiple translation failed: {str(e)}", e)

    async def detect_language(self, text: str) -> str:
        """Detect language using DeepL API."""
        if not text.strip():
            raise TranslationError("Cannot detect language of empty text")

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.executor, self._sync_detect_language, text)
            return result.lower()
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            raise TranslationError(f"Language detection failed: {str(e)}", e)

    def _sync_detect_language(self, text: str) -> str:
        """Synchronous wrapper for language detection."""
        # DeepL doesn't have dedicated detection endpoint
        # We can use a translation request and check the detected source language
        result = self.client.translate_text(text, target_lang="EN")
        return result.detected_source_lang

    def get_supported_languages(self) -> list[str]:
        """Get supported languages from DeepL API."""
        if self._supported_languages is None:
            try:
                # Get source and target languages
                source_langs = self.client.get_source_languages()
                target_langs = self.client.get_target_languages()

                # Combine and deduplicate language codes
                all_langs = set()
                for lang in source_langs:
                    all_langs.add(lang.code.lower())
                for lang in target_langs:
                    all_langs.add(lang.code.lower())

                self._supported_languages = sorted(all_langs)
            except Exception as e:
                logger.error(f"Failed to get supported languages: {e}")
                # Return common languages as fallback
                self._supported_languages = ["en", "de", "fr", "es", "it", "pt", "ru", "ja", "zh"]

        return self._supported_languages.copy()

    async def check_availability(self) -> bool:
        """Check if DeepL service is available."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, self._sync_check_availability)
            return True
        except Exception as e:
            logger.warning(f"DeepL availability check failed: {e}")
            return False

    def _sync_check_availability(self) -> None:
        """Synchronous availability check."""
        # Try to get account usage info as a health check
        self.client.get_usage()

    def dispose(self) -> None:
        """Clean up resources."""
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
