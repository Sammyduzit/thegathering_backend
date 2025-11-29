"""
AI Provider interface for LLM chat completion services.

This interface abstracts LLM functionality, allowing different providers
(OpenAI, Anthropic, etc.) to be used interchangeably through dependency injection.

Based on LangChain 0.3.x architecture with modern async patterns.
"""

from abc import ABC, abstractmethod

from app.core.constants import DEFAULT_PROVIDER_MAX_TOKENS, DEFAULT_PROVIDER_TEMPERATURE


class IAIProvider(ABC):
    """Abstract interface for AI/LLM chat completion services."""

    @abstractmethod
    async def generate_response(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        temperature: float = DEFAULT_PROVIDER_TEMPERATURE,
        max_tokens: int = DEFAULT_PROVIDER_MAX_TOKENS,
        **kwargs,
    ) -> str:
        """
        Generate chat completion response from LLM.

        :param messages: List of message dicts with 'role' and 'content' keys (e.g., [{"role": "user", "content": "Hello"}])
        :param system_prompt: Optional system prompt to prepend
        :param temperature: LLM temperature (0.0-2.0)
        :param max_tokens: Maximum tokens in response
        :param kwargs: Additional provider-specific parameters
        :return: Generated response text
        :raises AIProviderError: If LLM call fails
        """
        pass

    @abstractmethod
    async def generate_streaming_response(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        temperature: float = DEFAULT_PROVIDER_TEMPERATURE,
        max_tokens: int = DEFAULT_PROVIDER_MAX_TOKENS,
        **kwargs,
    ):
        """
        Generate streaming chat completion response from LLM.

        :param messages: List of message dicts with 'role' and 'content' keys
        :param system_prompt: Optional system prompt to prepend
        :param temperature: LLM temperature (0.0-2.0)
        :param max_tokens: Maximum tokens in response
        :param kwargs: Additional provider-specific parameters
        :return: Response chunks as they arrive
        :raises AIProviderError: If LLM call fails
        """
        pass

    @abstractmethod
    async def check_availability(self) -> bool:
        """
        Check if AI provider is available and configured.

        :return: True if provider is available, False otherwise
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Get the current model name being used.

        :return: Model identifier (e.g., 'gpt-4', 'claude-3-opus')
        """
        pass


class AIProviderError(Exception):
    """Exception raised when AI provider operations fail."""

    def __init__(self, message: str, original_error: Exception | None = None):
        super().__init__(message)
        self.original_error = original_error
