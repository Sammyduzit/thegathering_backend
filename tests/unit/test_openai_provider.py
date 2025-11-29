"""Unit tests for OpenAIProvider."""

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from app.interfaces.ai_provider import AIProviderError
from app.providers.openai_provider import OpenAIProvider


@pytest.mark.unit
class TestOpenAIProvider:
    """Unit tests for OpenAI LLM provider."""

    @pytest.fixture
    def provider(self):
        """Create OpenAI provider instance."""
        with patch("app.providers.openai_provider.ChatOpenAI") as mock_chat:
            # Mock the ChatOpenAI instance
            mock_llm = AsyncMock()
            mock_chat.return_value = mock_llm
            provider = OpenAIProvider(api_key="test-api-key", model_name="gpt-4")
            provider.llm = mock_llm  # Replace with mock
            return provider

    async def test_generate_response_basic(self, provider):
        """Test basic response generation."""
        # Arrange
        messages = [{"role": "user", "content": "Hello"}]
        provider.llm.ainvoke.return_value = AIMessage(content="Hi there!")

        # Act
        result = await provider.generate_response(messages=messages)

        # Assert
        assert result == "Hi there!"
        provider.llm.ainvoke.assert_called_once()

        # Verify message structure
        call_args = provider.llm.ainvoke.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0].content == "Hello"

    async def test_generate_response_with_system_prompt(self, provider):
        """Test response generation with system prompt."""
        # Arrange
        messages = [{"role": "user", "content": "Tell me a joke"}]
        system_prompt = "You are a funny comedian"
        provider.llm.ainvoke.return_value = AIMessage(content="Why did the chicken...")

        # Act
        result = await provider.generate_response(messages=messages, system_prompt=system_prompt)

        # Assert
        assert result == "Why did the chicken..."

        # Verify system message was added first
        call_args = provider.llm.ainvoke.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0].content == system_prompt
        assert call_args[1].content == "Tell me a joke"

    async def test_generate_response_with_temperature(self, provider):
        """Test response generation with custom temperature."""
        # Arrange
        messages = [{"role": "user", "content": "Be creative"}]

        # Act
        with patch("app.providers.openai_provider.ChatOpenAI") as mock_chat:
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = AIMessage(content="Creative response!")
            mock_chat.return_value = mock_llm

            result = await provider.generate_response(messages=messages, temperature=0.9)

        # Assert
        assert result == "Creative response!"

    async def test_generate_response_with_max_tokens(self, provider):
        """Test response generation with custom max_tokens."""
        # Arrange
        messages = [{"role": "user", "content": "Short answer"}]

        # Act
        with patch("app.providers.openai_provider.ChatOpenAI") as mock_chat:
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = AIMessage(content="OK")
            mock_chat.return_value = mock_llm

            result = await provider.generate_response(messages=messages, max_tokens=50)

        # Assert
        assert result == "OK"

    async def test_generate_response_multiple_messages(self, provider):
        """Test response generation with multiple messages."""
        # Arrange
        messages = [
            {"role": "user", "content": "Alice: Hi"},
            {"role": "user", "content": "You: Hello Alice!"},
            {"role": "user", "content": "Alice: How are you?"},
        ]
        provider.llm.ainvoke.return_value = AIMessage(content="I'm doing well, thanks!")

        # Act
        result = await provider.generate_response(messages=messages)

        # Assert
        assert result == "I'm doing well, thanks!"

        # Verify all messages were passed
        call_args = provider.llm.ainvoke.call_args[0][0]
        assert len(call_args) == 3

    async def test_generate_response_with_system_in_messages(self, provider):
        """Test handling system messages in message list."""
        # Arrange
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Help me"},
        ]
        provider.llm.ainvoke.return_value = AIMessage(content="How can I help?")

        # Act
        result = await provider.generate_response(messages=messages)

        # Assert
        assert result == "How can I help?"

    async def test_generate_response_provider_error(self, provider):
        """Test handling LLM provider errors."""
        # Arrange
        messages = [{"role": "user", "content": "Test"}]
        provider.llm.ainvoke.side_effect = Exception("API connection failed")

        # Act & Assert
        with pytest.raises(AIProviderError, match="Failed to generate response"):
            await provider.generate_response(messages=messages)

    async def test_generate_response_empty_response(self, provider):
        """Test handling empty response from LLM."""
        # Arrange
        messages = [{"role": "user", "content": "Test"}]
        provider.llm.ainvoke.return_value = AIMessage(content="")

        # Act
        result = await provider.generate_response(messages=messages)

        # Assert
        assert result == ""

    async def test_check_availability_success(self, provider):
        """Test provider availability check when configured."""
        # Arrange
        provider.llm.ainvoke.return_value = AIMessage(content="test")

        # Act
        result = await provider.check_availability()

        # Assert
        assert result is True

    async def test_check_availability_failure(self, provider):
        """Test provider availability check when API fails."""
        # Arrange
        provider.llm.ainvoke.side_effect = Exception("API unavailable")

        # Act
        result = await provider.check_availability()

        # Assert
        assert result is False
