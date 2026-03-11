"""Unit tests for agent tool factory."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.ai.agent_tools import create_agent_tools


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_agent_tools_memory_and_history_and_user_lookup():
    memory_retriever = AsyncMock()
    memory_retriever.retrieve_tiered.return_value = [
        SimpleNamespace(
            summary="User likes Rust",
            importance_score=8.0,
            memory_metadata={"type": "long_term"},
        )
    ]

    message_repo = AsyncMock()
    message_repo.search_messages.return_value = [
        SimpleNamespace(
            sent_at=datetime(2026, 2, 28, tzinfo=timezone.utc),
            sender_username="sam",
            content="Rust memory pipeline sounds great",
        )
    ]

    user_repo = AsyncMock()
    user_repo.get_by_username.return_value = SimpleNamespace(
        username="sam",
        preferred_language="de",
        status=SimpleNamespace(value="available"),
        is_active=True,
        email="sam@example.com",
    )

    tools = create_agent_tools(
        memory_retriever=memory_retriever,
        message_repo=message_repo,
        user_repo=user_repo,
        ai_entity_id=11,
        conversation_id=22,
        user_id=33,
    )
    tool_map = {tool.name: tool for tool in tools}

    memory_result = await tool_map["search_memory"].ainvoke({"query": "Rust"})
    history_result = await tool_map["search_message_history"].ainvoke({"query": "Rust", "limit": 3})
    user_result = await tool_map["lookup_user"].ainvoke({"username": "sam"})

    assert "User likes Rust" in memory_result
    assert "Rust memory pipeline sounds great" in history_result
    assert "username: sam" in user_result
    assert "preferred_language: de" in user_result
    assert "email" not in user_result.lower()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_search_memory_reports_unavailable_when_retriever_missing():
    message_repo = AsyncMock()
    user_repo = AsyncMock()
    tools = create_agent_tools(
        memory_retriever=None,
        message_repo=message_repo,
        user_repo=user_repo,
        ai_entity_id=1,
        conversation_id=2,
        user_id=3,
    )
    tool_map = {tool.name: tool for tool in tools}

    result = await tool_map["search_memory"].ainvoke({"query": "anything"})
    assert "unavailable" in result.lower()
