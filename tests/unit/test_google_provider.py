"""Unit tests for GoogleProvider."""

from types import SimpleNamespace

import pytest

from app.providers import google_provider
from app.providers.google_provider import GoogleProvider


class DummyLLM:
    def __init__(self, response_content: str):
        self.response_content = response_content

    async def ainvoke(self, messages):
        return SimpleNamespace(content=self.response_content)


@pytest.mark.unit
async def test_google_provider_generate_response(monkeypatch):
    """GoogleProvider.generate_response should return LLM content and build messages."""
    dummy_llm = DummyLLM("hello")

    # Patch ChatGoogleGenerativeAI to return our dummy
    monkeypatch.setattr(google_provider, "ChatGoogleGenerativeAI", lambda **kwargs: dummy_llm)

    provider = GoogleProvider(api_key="dummy", model_name="gemini-2.5-flash-lite")

    result = await provider.generate_response(
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.1,
    )

    assert result == "hello"
