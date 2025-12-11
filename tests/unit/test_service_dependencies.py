"""Unit tests for service dependency factories."""

import builtins

import pytest

from app.core import config
from app.providers.google_provider import GoogleProvider
from app.providers.openai_provider import OpenAIProvider
from app.services import service_dependencies


@pytest.mark.unit
def test_get_ltm_ai_provider_google_guard(monkeypatch):
    """Expect RuntimeError if ltm_provider=google but GOOGLE_API_KEY is missing."""
    # Backup
    orig_provider = config.settings.ltm_provider
    orig_google_key = config.settings.google_api_key
    try:
        config.settings.ltm_provider = "google"
        config.settings.google_api_key = None

        with pytest.raises(RuntimeError):
            service_dependencies.get_ltm_ai_provider()
    finally:
        config.settings.ltm_provider = orig_provider
        config.settings.google_api_key = orig_google_key


@pytest.mark.unit
def test_get_ltm_ai_provider_google_success(monkeypatch):
    """Factory returns GoogleProvider when configured."""
    orig_provider = config.settings.ltm_provider
    orig_google_key = config.settings.google_api_key
    try:
        config.settings.ltm_provider = "google"
        config.settings.google_api_key = "dummy-key"

        provider = service_dependencies.get_ltm_ai_provider()
        assert isinstance(provider, GoogleProvider)
    finally:
        config.settings.ltm_provider = orig_provider
        config.settings.google_api_key = orig_google_key


@pytest.mark.unit
def test_get_ltm_ai_provider_openai_guard(monkeypatch):
    """Expect RuntimeError if ltm_provider=openai but OPENAI_API_KEY is missing."""
    orig_provider = config.settings.ltm_provider
    orig_openai_key = config.settings.openai_api_key
    try:
        config.settings.ltm_provider = "openai"
        config.settings.openai_api_key = None

        with pytest.raises(RuntimeError):
            service_dependencies.get_ltm_ai_provider()
    finally:
        config.settings.ltm_provider = orig_provider
        config.settings.openai_api_key = orig_openai_key


@pytest.mark.unit
def test_get_ltm_ai_provider_openai_success(monkeypatch):
    """Factory returns OpenAIProvider when configured."""
    orig_provider = config.settings.ltm_provider
    orig_openai_key = config.settings.openai_api_key
    try:
        config.settings.ltm_provider = "openai"
        config.settings.openai_api_key = "dummy-key"

        provider = service_dependencies.get_ltm_ai_provider()
        assert isinstance(provider, OpenAIProvider)
    finally:
        config.settings.ltm_provider = orig_provider
        config.settings.openai_api_key = orig_openai_key
