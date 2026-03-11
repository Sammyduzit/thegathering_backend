"""Factory for conversation-scoped, read-only agent tools."""

from __future__ import annotations

from time import perf_counter

import structlog
from langchain_core.tools import BaseTool, tool

from app.interfaces.memory_retriever import IMemoryRetriever
from app.repositories.message_repository import IMessageRepository
from app.repositories.user_repository import IUserRepository

logger = structlog.get_logger(__name__)


def _format_memory_results(memories) -> str:
    if not memories:
        return "No relevant memories found."

    lines = ["Relevant memories:"]
    for memory in memories[:5]:
        memory_type = "unknown"
        if memory.memory_metadata:
            memory_type = memory.memory_metadata.get("type", "unknown")

        lines.append(f"- [{memory_type}] {memory.summary} (importance={memory.importance_score:.1f})")

    return "\n".join(lines)


def _format_history_results(messages) -> str:
    if not messages:
        return "No matching message history found in this conversation."

    lines = ["Matching messages (newest first):"]
    for message in messages:
        timestamp = message.sent_at.isoformat() if message.sent_at else "unknown-time"
        sender = message.sender_username or "Unknown"
        content = " ".join(message.content.split())
        lines.append(f"- ({timestamp}) {sender}: {content}")

    return "\n".join(lines)


def create_agent_tools(
    *,
    memory_retriever: IMemoryRetriever | None,
    message_repo: IMessageRepository,
    user_repo: IUserRepository,
    ai_entity_id: int,
    conversation_id: int,
    user_id: int,
) -> list[BaseTool]:
    """Create conversation-scoped tools for a single response run."""

    @tool("search_memory")
    async def search_memory(query: str) -> str:
        """Search the AI's tiered memory for facts relevant to the query."""
        started = perf_counter()
        try:
            if not query.strip():
                return "Memory query was empty."
            if memory_retriever is None:
                return "Memory retrieval is currently unavailable."

            memories = await memory_retriever.retrieve_tiered(
                entity_id=ai_entity_id,
                user_id=user_id,
                conversation_id=conversation_id,
                query=query,
            )
            return _format_memory_results(memories)
        except Exception as exc:
            logger.warning("agent_tool_failed", tool_name="search_memory", error=str(exc))
            return "Memory lookup failed."
        finally:
            logger.info(
                "agent_tool_call",
                tool_name="search_memory",
                latency_ms=int((perf_counter() - started) * 1000),
            )

    @tool("search_message_history")
    async def search_message_history(query: str, limit: int = 5) -> str:
        """Search recent messages in the current conversation only."""
        started = perf_counter()
        safe_limit = min(max(limit, 1), 10)
        try:
            if not query.strip():
                return "History query was empty."

            messages = await message_repo.search_messages(
                query=query,
                conversation_id=conversation_id,
                limit=safe_limit,
            )
            return _format_history_results(messages)
        except Exception as exc:
            logger.warning("agent_tool_failed", tool_name="search_message_history", error=str(exc))
            return "Message history lookup failed."
        finally:
            logger.info(
                "agent_tool_call",
                tool_name="search_message_history",
                latency_ms=int((perf_counter() - started) * 1000),
                limit=safe_limit,
            )

    @tool("lookup_user")
    async def lookup_user(username: str) -> str:
        """Lookup a user by username and return non-sensitive profile fields."""
        started = perf_counter()
        try:
            normalized_username = username.strip()
            if not normalized_username:
                return "Username was empty."

            user = await user_repo.get_by_username(normalized_username)
            if not user:
                return "User not found."

            status = user.status.value if hasattr(user.status, "value") else str(user.status)
            preferred_language = user.preferred_language or "unknown"

            return (
                f"username: {user.username}\n"
                f"preferred_language: {preferred_language}\n"
                f"status: {status}\n"
                f"is_active: {user.is_active}"
            )
        except Exception as exc:
            logger.warning("agent_tool_failed", tool_name="lookup_user", error=str(exc))
            return "User lookup failed."
        finally:
            logger.info(
                "agent_tool_call",
                tool_name="lookup_user",
                latency_ms=int((perf_counter() - started) * 1000),
            )

    return [search_memory, search_message_history, lookup_user]
