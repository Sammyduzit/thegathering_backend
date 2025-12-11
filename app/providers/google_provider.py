"""
Google Gemini provider implementation using LangChain 0.3.x.

Provides chat completion functionality via Google's Gemini API with modern async patterns.
"""

from typing import AsyncIterator

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.constants import (
    DEFAULT_PROVIDER_MAX_TOKENS,
    DEFAULT_PROVIDER_TEMPERATURE,
)
from app.interfaces.ai_provider import AIProviderError, IAIProvider

logger = structlog.get_logger(__name__)


class GoogleProvider(IAIProvider):
    """Google Gemini LLM provider implementation using LangChain."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-1.5-flash",
        default_temperature: float = DEFAULT_PROVIDER_TEMPERATURE,
        default_max_tokens: int = DEFAULT_PROVIDER_MAX_TOKENS,
    ):
        """
        Initialize Google Gemini provider.

        :param api_key: Google API key
        :param model_name: Model to use (e.g., 'gemini-1.5-flash', 'gemini-1.5-pro')
        :param default_temperature: Default temperature for responses
        :param default_max_tokens: Default max tokens for responses
        """
        self.api_key = api_key
        self.model_name = model_name
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens

        # Initialize LangChain ChatGoogleGenerativeAI
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model_name,
            temperature=default_temperature,
            max_output_tokens=default_max_tokens,
        )

    def _build_messages(
        self, messages: list[dict[str, str]], system_prompt: str | None = None
    ) -> list[SystemMessage | HumanMessage]:
        """
        Build LangChain message list from message dicts.

        :param messages: List of message dicts with 'role' and 'content' keys
        :param system_prompt: Optional system prompt to prepend
        :return: List of LangChain message objects
        """
        lc_messages = []

        # Add system prompt if provided
        if system_prompt:
            lc_messages.append(SystemMessage(content=system_prompt))

        # Convert messages to LangChain format
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            else:  # user or assistant
                lc_messages.append(HumanMessage(content=content))

        return lc_messages

    def _get_llm_override(
        self, temperature: float | None = None, max_tokens: int | None = None, streaming: bool = False, **kwargs
    ) -> ChatGoogleGenerativeAI:
        """
        Get LLM instance with parameter overrides if needed.

        :param temperature: Override default temperature
        :param max_tokens: Override default max_tokens
        :param streaming: Enable streaming mode
        :param kwargs: Additional Google-specific parameters
        :return: ChatGoogleGenerativeAI instance (either default or with overrides)
        """
        if temperature is None and max_tokens is None and not streaming:
            return self.llm

        return ChatGoogleGenerativeAI(
            google_api_key=self.api_key,
            model=self.model_name,
            temperature=temperature if temperature is not None else self.default_temperature,
            max_output_tokens=max_tokens if max_tokens is not None else self.default_max_tokens,
            streaming=streaming,
            **kwargs,
        )

    async def generate_response(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> str:
        """
        Generate chat completion response from Google Gemini.

        :param messages: List of message dicts with 'role' and 'content' keys
        :param system_prompt: Optional system prompt to prepend
        :param temperature: Override default temperature
        :param max_tokens: Override default max_tokens
        :param kwargs: Additional Google-specific parameters
        :return: Generated response text
        :raises AIProviderError: If Google API call fails
        """
        try:
            # Build message list and get LLM instance
            lc_messages = self._build_messages(messages, system_prompt)
            llm = self._get_llm_override(temperature, max_tokens, **kwargs)

            # Generate response
            response = await llm.ainvoke(lc_messages)
            return response.content

        except Exception as e:
            logger.error(f"Google Gemini API call failed: {e}")
            raise AIProviderError(f"Failed to generate response: {str(e)}", original_error=e)

    async def generate_streaming_response(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Generate streaming chat completion response from Google Gemini.

        :param messages: List of message dicts with 'role' and 'content' keys
        :param system_prompt: Optional system prompt to prepend
        :param temperature: Override default temperature
        :param max_tokens: Override default max_tokens
        :param kwargs: Additional Google-specific parameters
        :return: Response chunks as they arrive
        :raises AIProviderError: If Google API call fails
        """
        try:
            # Build message list and get LLM instance with streaming enabled
            lc_messages = self._build_messages(messages, system_prompt)
            llm = self._get_llm_override(temperature, max_tokens, streaming=True, **kwargs)

            # Stream response
            async for chunk in llm.astream(lc_messages):
                if hasattr(chunk, "content") and chunk.content:
                    yield chunk.content

        except Exception as e:
            logger.error(f"Google Gemini streaming API call failed: {e}")
            raise AIProviderError(f"Failed to generate streaming response: {str(e)}", original_error=e)

    async def check_availability(self) -> bool:
        """
        Check if Google Gemini provider is available and configured.

        :return: True if provider is available, False otherwise
        """
        if not self.api_key:
            return False

        try:
            # Simple test call to verify API key
            test_messages = [HumanMessage(content="test")]
            await self.llm.ainvoke(test_messages)
            return True
        except Exception as e:
            logger.warning(f"Google Gemini availability check failed: {e}")
            return False

    def get_model_name(self) -> str:
        """
        Get the current Google Gemini model name.

        :return: Model identifier (e.g., 'gemini-1.5-flash')
        """
        return self.model_name
