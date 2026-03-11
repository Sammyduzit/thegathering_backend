"""Agent-based response generation with tool-calling support."""

from __future__ import annotations

from uuid import uuid4

import structlog
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from app.interfaces.ai_provider import IAIProvider
from app.interfaces.memory_retriever import IMemoryRetriever
from app.repositories.message_repository import IMessageRepository
from app.repositories.user_repository import IUserRepository
from app.services.ai.agent_tools import create_agent_tools

logger = structlog.get_logger(__name__)


class AgentResponseService:
    """Generates responses via tool-calling agent runtime."""

    def __init__(
        self,
        ai_provider: IAIProvider,
        memory_retriever: IMemoryRetriever | None,
        message_repo: IMessageRepository,
        user_repo: IUserRepository,
        max_tool_iterations: int = 3,
    ):
        self.ai_provider = ai_provider
        self.memory_retriever = memory_retriever
        self.message_repo = message_repo
        self.user_repo = user_repo
        self.max_tool_iterations = max_tool_iterations
        self._create_agent_callable = self._resolve_create_agent()

    async def generate_conversation_response(
        self,
        *,
        messages: list[dict[str, str]],
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        ai_entity_id: int,
        conversation_id: int,
        user_id: int,
    ) -> str:
        """Generate a response for a conversation using tools."""
        tools = create_agent_tools(
            memory_retriever=self.memory_retriever,
            message_repo=self.message_repo,
            user_repo=self.user_repo,
            ai_entity_id=ai_entity_id,
            conversation_id=conversation_id,
            user_id=user_id,
        )

        try:
            if self._create_agent_callable:
                return await self._generate_with_create_agent(
                    tools=tools,
                    messages=messages,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
        except Exception as exc:
            logger.warning("create_agent_runtime_failed_fallback", error=str(exc))

        return await self._generate_with_manual_tool_loop(
            tools=tools,
            messages=messages,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _resolve_create_agent(self):
        try:
            from langchain.agents import create_agent

            return create_agent
        except Exception:
            return None

    async def _generate_with_create_agent(
        self,
        *,
        tools: list[BaseTool],
        messages: list[dict[str, str]],
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        create_agent = self._create_agent_callable
        if create_agent is None:
            return await self._generate_with_manual_tool_loop(
                tools=tools,
                messages=messages,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        model = self.ai_provider.get_chat_model(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        agent = create_agent(
            model=model,
            tools=tools,
            system_prompt=system_prompt,
        )

        result = await agent.ainvoke({"messages": self._to_langchain_messages(messages)})
        return self._extract_agent_output(result)

    async def _generate_with_manual_tool_loop(
        self,
        *,
        tools: list[BaseTool],
        messages: list[dict[str, str]],
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        # Compatibility path: portable tool-calling loop without AgentExecutor.
        model = self.ai_provider.get_chat_model(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        bound_model = model.bind_tools(tools)

        tool_map = {tool.name: tool for tool in tools}
        conversation: list[BaseMessage] = [SystemMessage(content=system_prompt), *self._to_langchain_messages(messages)]
        last_content = ""

        for _ in range(self.max_tool_iterations):
            ai_message = await bound_model.ainvoke(conversation)
            if not isinstance(ai_message, AIMessage):
                ai_message = AIMessage(content=str(getattr(ai_message, "content", "")))

            conversation.append(ai_message)
            last_content = self._to_plain_text(ai_message.content)

            tool_calls = getattr(ai_message, "tool_calls", None) or []
            if not tool_calls:
                return last_content.strip()

            for tool_call in tool_calls:
                conversation.append(await self._execute_tool_call(tool_call=tool_call, tool_map=tool_map))

        logger.warning("agent_tool_iteration_limit_reached", max_tool_iterations=self.max_tool_iterations)
        return last_content.strip()

    async def _execute_tool_call(self, *, tool_call: dict, tool_map: dict[str, BaseTool]) -> ToolMessage:
        tool_name = tool_call.get("name", "")
        tool_call_id = tool_call.get("id") or str(uuid4())
        args = tool_call.get("args", {})

        tool = tool_map.get(tool_name)
        if not tool:
            return ToolMessage(
                content=f"Tool '{tool_name}' is unavailable.",
                tool_call_id=tool_call_id,
                name=tool_name,
            )

        try:
            tool_result = await tool.ainvoke(args)
            return ToolMessage(
                content=self._to_plain_text(tool_result),
                tool_call_id=tool_call_id,
                name=tool_name,
            )
        except Exception as exc:
            logger.warning("agent_tool_execution_failed", tool_name=tool_name, error=str(exc))
            return ToolMessage(
                content=f"Tool '{tool_name}' failed.",
                tool_call_id=tool_call_id,
                name=tool_name,
            )

    def _to_langchain_messages(self, messages: list[dict[str, str]]) -> list[BaseMessage]:
        converted: list[BaseMessage] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "assistant":
                converted.append(AIMessage(content=content))
            elif role == "system":
                converted.append(SystemMessage(content=content))
            else:
                converted.append(HumanMessage(content=content))

        return converted

    def _extract_agent_output(self, result) -> str:
        if not isinstance(result, dict):
            logger.warning(
                "agent_output_unexpected_type",
                result_type=type(result).__name__,
            )
            raise ValueError("create_agent returned non-dict result")

        messages = result.get("messages")
        if not isinstance(messages, list) or not messages:
            logger.warning(
                "agent_output_missing_messages",
                result_keys=sorted(result.keys()),
            )
            raise ValueError("create_agent returned result without messages")

        content = self._extract_last_message_content(messages)
        if not content:
            logger.warning(
                "agent_output_empty_message",
                last_message_type=type(messages[-1]).__name__,
            )
            raise ValueError("create_agent returned empty message content")

        return content

    def _extract_last_message_content(self, messages: list[object]) -> str | None:
        for message in reversed(messages):
            if isinstance(message, BaseMessage):
                raw_content = message.content
            elif isinstance(message, dict):
                raw_content = message.get("content")
            else:
                raw_content = getattr(message, "content", None)

            if raw_content is None:
                continue

            content = self._to_plain_text(raw_content).strip()
            if content:
                return content

        return None

    def _to_plain_text(self, content) -> str:
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(str(text))
                else:
                    parts.append(str(item))
            return "\n".join(parts)

        return str(content)
