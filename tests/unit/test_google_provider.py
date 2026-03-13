"""Unit tests for GoogleProvider."""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.providers import google_provider
from app.providers.google_provider import GoogleProvider


class DummyLLM:
    def __init__(self, response_content: str):
        self.response_content = response_content
        self.last_messages = None

    async def ainvoke(self, messages):
        self.last_messages = messages
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


@pytest.mark.unit
async def test_google_provider_maps_assistant_role_to_ai_message(monkeypatch):
    dummy_llm = DummyLLM("ok")
    monkeypatch.setattr(google_provider, "ChatGoogleGenerativeAI", lambda **kwargs: dummy_llm)

    provider = GoogleProvider(api_key="dummy", model_name="gemini-2.5-flash-lite")

    await provider.generate_response(
        messages=[
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Continue"},
        ]
    )

    assert isinstance(dummy_llm.last_messages[0], HumanMessage)
    assert isinstance(dummy_llm.last_messages[1], AIMessage)
    assert isinstance(dummy_llm.last_messages[2], HumanMessage)


@pytest.mark.unit
async def test_google_provider_get_chat_model(monkeypatch):
    dummy_llm = DummyLLM("ok")
    monkeypatch.setattr(google_provider, "ChatGoogleGenerativeAI", lambda **kwargs: dummy_llm)
    provider = GoogleProvider(api_key="dummy", model_name="gemini-2.5-flash-lite")

    assert provider.get_chat_model() is dummy_llm
