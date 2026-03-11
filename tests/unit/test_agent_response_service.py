"""Unit tests for AgentResponseService."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage

from app.services.ai.agent_response_service import AgentResponseService


class FakeChatModel:
    def __init__(self, responses):
        self._responses = list(responses)
        self.bound_tools = None
        self.bind_kwargs = None

    def bind(self, **kwargs):
        self.bind_kwargs = kwargs
        return self

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    async def ainvoke(self, messages):
        return self._responses.pop(0)


class FakeProvider:
    def __init__(self, model):
        self.model = model
        self.last_overrides = None

    def get_chat_model(self, temperature=None, max_tokens=None):
        self.last_overrides = {"temperature": temperature, "max_tokens": max_tokens}
        return self.model


@pytest.mark.unit
@pytest.mark.asyncio
async def test_agent_response_service_manual_loop_without_tool_calls():
    model = FakeChatModel([AIMessage(content="Direct agent answer")])
    provider = FakeProvider(model=model)

    service = AgentResponseService(
        ai_provider=provider,
        memory_retriever=None,
        message_repo=AsyncMock(),
        user_repo=AsyncMock(),
    )

    result = await service.generate_conversation_response(
        messages=[{"role": "user", "content": "Hi"}],
        system_prompt="Be helpful",
        temperature=0.3,
        max_tokens=100,
        ai_entity_id=1,
        conversation_id=2,
        user_id=3,
    )

    assert result == "Direct agent answer"
    assert provider.last_overrides == {"temperature": 0.3, "max_tokens": 100}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_agent_response_service_executes_tool_call_then_returns_final_answer():
    model = FakeChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "call_1", "name": "search_message_history", "args": {"query": "rust", "limit": 2}}
                ],
            ),
            AIMessage(content="Final answer with tool context"),
        ]
    )
    provider = FakeProvider(model=model)

    message_repo = AsyncMock()
    message_repo.search_messages.return_value = [
        SimpleNamespace(
            sent_at=None,
            sender_username="sam",
            content="Rust message",
        )
    ]

    service = AgentResponseService(
        ai_provider=provider,
        memory_retriever=None,
        message_repo=message_repo,
        user_repo=AsyncMock(),
    )

    result = await service.generate_conversation_response(
        messages=[{"role": "user", "content": "Find rust context"}],
        system_prompt="Use tools when useful",
        temperature=0.2,
        max_tokens=120,
        ai_entity_id=1,
        conversation_id=99,
        user_id=55,
    )

    assert result == "Final answer with tool context"
    assert provider.last_overrides == {"temperature": 0.2, "max_tokens": 120}
    message_repo.search_messages.assert_called_once_with(
        query="rust",
        conversation_id=99,
        limit=2,
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_agent_response_service_create_agent_reads_messages_output():
    model = FakeChatModel([])
    provider = FakeProvider(model=model)

    service = AgentResponseService(
        ai_provider=provider,
        memory_retriever=None,
        message_repo=AsyncMock(),
        user_repo=AsyncMock(),
    )

    class FakeAgent:
        async def ainvoke(self, _payload):
            return {"messages": [AIMessage(content="Agent output")]}

    def fake_create_agent(*_args, **_kwargs):
        return FakeAgent()

    service._create_agent_callable = fake_create_agent

    result = await service.generate_conversation_response(
        messages=[{"role": "user", "content": "Hi"}],
        system_prompt="Be helpful",
        temperature=0.4,
        max_tokens=80,
        ai_entity_id=1,
        conversation_id=2,
        user_id=3,
    )

    assert result == "Agent output"
    assert provider.last_overrides == {"temperature": 0.4, "max_tokens": 80}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_agent_response_service_create_agent_falls_back_on_bad_output():
    model = FakeChatModel([AIMessage(content="Fallback answer")])
    provider = FakeProvider(model=model)

    service = AgentResponseService(
        ai_provider=provider,
        memory_retriever=None,
        message_repo=AsyncMock(),
        user_repo=AsyncMock(),
    )

    class FakeAgent:
        async def ainvoke(self, _payload):
            return {"unexpected": "shape"}

    def fake_create_agent(*_args, **_kwargs):
        return FakeAgent()

    service._create_agent_callable = fake_create_agent

    result = await service.generate_conversation_response(
        messages=[{"role": "user", "content": "Hi"}],
        system_prompt="Be helpful",
        temperature=0.1,
        max_tokens=50,
        ai_entity_id=1,
        conversation_id=2,
        user_id=3,
    )

    assert result == "Fallback answer"
