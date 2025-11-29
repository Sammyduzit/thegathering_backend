"""Unit tests for AIResponseService."""

from unittest.mock import AsyncMock

import pytest

from app.interfaces.ai_provider import AIProviderError
from app.models.message import Message
from app.services.ai.ai_response_service import AIResponseService


@pytest.mark.unit
class TestAIResponseService:
    """Unit tests for AI response service."""

    @pytest.fixture
    def mock_cooldown_repo(self):
        """Create mock cooldown repository with sensible defaults."""
        repo = AsyncMock()
        repo.is_on_cooldown = AsyncMock(return_value=False)
        repo.upsert_cooldown = AsyncMock()
        return repo

    @pytest.fixture
    def service(self, mock_ai_provider, mock_context_service, mock_message_repo, mock_cooldown_repo):
        """Create service instance with mocked dependencies from conftest."""
        return AIResponseService(
            ai_provider=mock_ai_provider,
            context_service=mock_context_service,
            message_repo=mock_message_repo,
            cooldown_repo=mock_cooldown_repo,
        )

    async def test_generate_conversation_response_success(
        self, service, mock_ai_provider, mock_context_service, mock_message_repo, sample_ai_entity
    ):
        """Test generating AI response for conversation successfully."""
        # Arrange
        messages = [{"role": "user", "content": "testuser: Hello"}]
        memory_context = "# Previous Memories:\n!! User likes AI"

        mock_context_service.build_full_context.return_value = (messages, memory_context)
        mock_ai_provider.generate_response.return_value = "Hi testuser! How can I help?"

        created_message = Message(id=1, content="Hi testuser! How can I help?", conversation_id=1, sender_ai_id=1)
        mock_message_repo.create_conversation_message.return_value = created_message

        # Act
        result = await service.generate_conversation_response(
            conversation_id=1,
            ai_entity=sample_ai_entity,
            user_id=42,
            include_memories=True,
        )

        # Assert
        assert result.content == "Hi testuser! How can I help?"
        assert result.sender_ai_id == 1

        mock_context_service.build_full_context.assert_called_once_with(
            conversation_id=1,
            room_id=None,
            ai_entity=sample_ai_entity,
            user_id=42,
            include_memories=True,
        )

        # Verify system prompt includes memories
        call_args = mock_ai_provider.generate_response.call_args
        assert call_args.kwargs["messages"] == messages
        assert "# Previous Memories:" in call_args.kwargs["system_prompt"]
        assert sample_ai_entity.system_prompt in call_args.kwargs["system_prompt"]
        assert abs(call_args.kwargs["temperature"] - 0.7) < 0.001
        assert call_args.kwargs["max_tokens"] == 1024

        mock_message_repo.create_conversation_message.assert_called_once_with(
            conversation_id=1,
            content="Hi testuser! How can I help?",
            sender_ai_id=1,
            in_reply_to_message_id=None,
        )

    async def test_generate_conversation_response_no_memories(
        self, service, mock_ai_provider, mock_context_service, mock_message_repo, sample_ai_entity
    ):
        """Test generating response without memories."""
        # Arrange
        messages = [{"role": "user", "content": "testuser: Hello"}]
        mock_context_service.build_full_context.return_value = (messages, None)
        mock_ai_provider.generate_response.return_value = "Hi!"

        created_message = Message(id=1, content="Hi!", conversation_id=1, sender_ai_id=1)
        mock_message_repo.create_conversation_message.return_value = created_message

        # Act
        result = await service.generate_conversation_response(
            conversation_id=1,
            ai_entity=sample_ai_entity,
            user_id=42,
            include_memories=False,
        )

        # Assert
        assert result.content == "Hi!"

        # Verify system prompt includes anti-parroting instruction but NOT memories
        call_args = mock_ai_provider.generate_response.call_args
        assert sample_ai_entity.system_prompt in call_args.kwargs["system_prompt"]
        assert "IMPORTANT: You respond directly as part of the conversation." in call_args.kwargs["system_prompt"]
        # Anti-parroting instruction
        assert "# Previous Memories:" not in call_args.kwargs["system_prompt"]

    async def test_generate_conversation_response_cleans_parroting(
        self, service, mock_ai_provider, mock_context_service, mock_message_repo, sample_ai_entity
    ):
        """Test that post-processing removes name prefixes from AI responses."""
        # Arrange
        messages = [{"role": "user", "content": "testuser: Hello"}]
        mock_context_service.build_full_context.return_value = (messages, None)
        # AI provider returns response WITH prefix (parroting)
        mock_ai_provider.generate_response.return_value = "test_ai: Hi there!"

        # Note: _clean_parroting will clean it to "Hi there!"
        created_message = Message(id=1, content="Hi there!", conversation_id=1, sender_ai_id=1)
        mock_message_repo.create_conversation_message.return_value = created_message

        # Act
        await service.generate_conversation_response(
            conversation_id=1,
            ai_entity=sample_ai_entity,
            user_id=42,
        )

        # Assert - message should be saved WITHOUT prefix
        mock_message_repo.create_conversation_message.assert_called_once_with(
            conversation_id=1,
            content="Hi there!",  # Prefix removed
            sender_ai_id=1,
            in_reply_to_message_id=None,
        )

    async def test_generate_conversation_response_provider_error(
        self, service, mock_ai_provider, mock_context_service, sample_ai_entity
    ):
        """Test handling LLM provider errors."""
        # Arrange
        mock_context_service.build_full_context.return_value = ([], None)
        mock_ai_provider.generate_response.side_effect = Exception("LLM API error")

        # Act & Assert
        with pytest.raises(AIProviderError, match="AI response generation failed"):
            await service.generate_conversation_response(
                conversation_id=1,
                ai_entity=sample_ai_entity,
                user_id=42,
            )

    async def test_generate_room_response_success(
        self, service, mock_ai_provider, mock_context_service, mock_message_repo, sample_ai_entity
    ):
        """Test generating AI response for room successfully."""
        # Arrange
        messages = [{"role": "user", "content": "testuser: Hello room"}]
        memory_context = "# Previous Memories:\n! Room discussion"

        mock_context_service.build_full_context.return_value = (messages, memory_context)
        mock_ai_provider.generate_response.return_value = "Hello everyone!"

        created_message = Message(id=1, content="Hello everyone!", room_id=1, sender_ai_id=1)
        mock_message_repo.create_room_message.return_value = created_message

        # Act
        result = await service.generate_room_response(
            room_id=1,
            ai_entity=sample_ai_entity,
            user_id=24,
            include_memories=True,
        )

        # Assert
        assert result.content == "Hello everyone!"
        assert result.sender_ai_id == 1

        mock_context_service.build_full_context.assert_called_once_with(
            conversation_id=None,
            room_id=1,
            ai_entity=sample_ai_entity,
            user_id=24,
            include_memories=True,
        )

        mock_message_repo.create_room_message.assert_called_once_with(
            room_id=1,
            content="Hello everyone!",
            sender_ai_id=1,
            in_reply_to_message_id=None,
        )

    async def test_should_ai_respond_mentioned_by_name(self, service, sample_ai_entity):
        """Test AI responds when mentioned by name."""
        from app.models.ai_entity import AIResponseStrategy

        # Arrange
        sample_ai_entity.conversation_response_strategy = AIResponseStrategy.CONV_SMART
        message = Message(id=1, content="Hey test_ai, how are you?", sender_user_id=2)

        # Act
        result = await service.should_ai_respond(
            ai_entity=sample_ai_entity,
            latest_message=message,
            conversation_id=1,
        )

        # Assert
        assert result is True

    async def test_should_ai_respond_mentioned_case_insensitive(self, service, sample_ai_entity):
        """Test AI responds when mentioned by username regardless of case."""
        from app.models.ai_entity import AIResponseStrategy

        # Arrange
        sample_ai_entity.conversation_response_strategy = AIResponseStrategy.CONV_SMART
        mentioned_name = sample_ai_entity.username.upper()
        message = Message(id=1, content=f"{mentioned_name}, can you help?", sender_user_id=2)

        # Act
        result = await service.should_ai_respond(
            ai_entity=sample_ai_entity,
            latest_message=message,
            conversation_id=1,
        )

        # Assert
        assert result is True

    async def test_should_ai_respond_question_in_conversation(self, service, sample_ai_entity):
        """Test AI responds to questions in conversations."""
        from app.models.ai_entity import AIResponseStrategy

        # Arrange
        sample_ai_entity.conversation_response_strategy = AIResponseStrategy.CONV_SMART
        message = Message(id=1, content="What is the weather like?", sender_user_id=2)

        # Act
        result = await service.should_ai_respond(
            ai_entity=sample_ai_entity,
            latest_message=message,
            conversation_id=1,
        )

        # Assert
        assert result is True

    async def test_should_ai_respond_question_in_room(self, service, sample_ai_entity):
        """Test AI does NOT respond to random questions in rooms (avoid spam)."""
        # Arrange
        message = Message(id=1, content="What is the weather like?", sender_user_id=2)

        # Act
        result = await service.should_ai_respond(
            ai_entity=sample_ai_entity,
            latest_message=message,
            room_id=1,
        )

        # Assert
        assert result is False  # More selective in rooms

    async def test_should_ai_respond_own_message(self, service, sample_ai_entity):
        """Test AI does NOT respond to its own messages."""
        # Arrange
        message = Message(id=1, content="I just said something", sender_ai_id=1)

        # Act
        result = await service.should_ai_respond(
            ai_entity=sample_ai_entity,
            latest_message=message,
            conversation_id=1,
        )

        # Assert
        assert result is False

    async def test_should_ai_respond_regular_message(self, service, sample_ai_entity):
        """Test AI does NOT respond to regular messages without trigger."""
        # Arrange
        message = Message(id=1, content="Just chatting here", sender_user_id=2)

        # Act
        result = await service.should_ai_respond(
            ai_entity=sample_ai_entity,
            latest_message=message,
            conversation_id=1,
        )

        # Assert
        assert result is False

    async def test_check_provider_availability_success(self, service, mock_ai_provider):
        """Test checking provider availability when configured."""
        # Arrange
        mock_ai_provider.check_availability.return_value = True

        # Act
        result = await service.check_provider_availability()

        # Assert
        assert result is True
        mock_ai_provider.check_availability.assert_called_once()

    async def test_check_provider_availability_error(self, service, mock_ai_provider):
        """Test checking provider availability when check fails."""
        # Arrange
        mock_ai_provider.check_availability.side_effect = Exception("API error")

        # Act
        result = await service.check_provider_availability()

        # Assert
        assert result is False

    # ===== Checkpoint 3 Tests: Response Strategy =====

    async def test_should_respond_room_mention_only(self, service, sample_ai_entity):
        """Test ROOM_MENTION_ONLY strategy."""
        from app.models.ai_entity import AIResponseStrategy

        # Arrange
        sample_ai_entity.room_response_strategy = AIResponseStrategy.ROOM_MENTION_ONLY
        message_with_mention = Message(
            id=1, content=f"Hey {sample_ai_entity.username}, can you help?", sender_user_id=1
        )
        message_without_mention = Message(id=2, content="This is a random message", sender_user_id=1)

        # Act & Assert
        assert await service.should_ai_respond(sample_ai_entity, message_with_mention, room_id=1) is True
        assert await service.should_ai_respond(sample_ai_entity, message_without_mention, room_id=1) is False

    async def test_should_respond_room_probabilistic(self, service, sample_ai_entity):
        """Test ROOM_PROBABILISTIC strategy."""
        from app.models.ai_entity import AIResponseStrategy

        # Arrange
        sample_ai_entity.room_response_strategy = AIResponseStrategy.ROOM_PROBABILISTIC
        sample_ai_entity.response_probability = 0.0  # Never respond (unless mentioned)
        message = Message(id=1, content="Random message", sender_user_id=1)
        message_with_mention = Message(id=2, content="Hey test_ai", sender_user_id=1)

        # Act & Assert
        assert await service.should_ai_respond(sample_ai_entity, message, room_id=1) is False
        assert await service.should_ai_respond(sample_ai_entity, message_with_mention, room_id=1) is True

    async def test_should_respond_room_active(self, service, sample_ai_entity):
        """Test ROOM_ACTIVE strategy."""
        from app.models.ai_entity import AIResponseStrategy

        # Arrange
        sample_ai_entity.room_response_strategy = AIResponseStrategy.ROOM_ACTIVE
        normal_message = Message(id=1, content="This is a normal message", sender_user_id=1)
        short_message = Message(id=2, content="ok", sender_user_id=1)

        # Act & Assert
        assert await service.should_ai_respond(sample_ai_entity, normal_message, room_id=1) is True
        assert await service.should_ai_respond(sample_ai_entity, short_message, room_id=1) is False

    async def test_should_respond_conv_every_message(self, service, sample_ai_entity):
        """Test CONV_EVERY_MESSAGE strategy."""
        from app.models.ai_entity import AIResponseStrategy

        # Arrange
        sample_ai_entity.conversation_response_strategy = AIResponseStrategy.CONV_EVERY_MESSAGE
        message = Message(id=1, content="Any message", sender_user_id=1)

        # Act & Assert
        assert await service.should_ai_respond(sample_ai_entity, message, conversation_id=1) is True

    async def test_should_respond_conv_on_questions(self, service, sample_ai_entity):
        """Test CONV_ON_QUESTIONS strategy."""
        from app.models.ai_entity import AIResponseStrategy

        # Arrange
        sample_ai_entity.conversation_response_strategy = AIResponseStrategy.CONV_ON_QUESTIONS
        question = Message(id=1, content="What is the weather today?", sender_user_id=1)
        statement = Message(id=2, content="It's a nice day", sender_user_id=1)

        # Act & Assert
        assert await service.should_ai_respond(sample_ai_entity, question, conversation_id=1) is True
        assert await service.should_ai_respond(sample_ai_entity, statement, conversation_id=1) is False

    async def test_should_respond_conv_smart(self, service, sample_ai_entity):
        """Test CONV_SMART strategy."""
        from app.models.ai_entity import AIResponseStrategy

        # Arrange
        sample_ai_entity.conversation_response_strategy = AIResponseStrategy.CONV_SMART
        question = Message(id=1, content="What time is it?", sender_user_id=1)
        mention = Message(id=2, content="Test AI, are you there?", sender_user_id=1)
        normal = Message(id=3, content="Just chatting here", sender_user_id=1)

        # Act & Assert
        assert await service.should_ai_respond(sample_ai_entity, question, conversation_id=1) is True
        assert await service.should_ai_respond(sample_ai_entity, mention, conversation_id=1) is True
        assert await service.should_ai_respond(sample_ai_entity, normal, conversation_id=1) is False

    async def test_should_respond_no_response_strategy(self, service, sample_ai_entity):
        """Test NO_RESPONSE strategy blocks all responses."""
        from app.models.ai_entity import AIResponseStrategy

        # Arrange
        sample_ai_entity.room_response_strategy = AIResponseStrategy.NO_RESPONSE
        sample_ai_entity.conversation_response_strategy = AIResponseStrategy.NO_RESPONSE
        message = Message(id=1, content="Hey test_ai, can you help?", sender_user_id=1)

        # Act & Assert
        assert await service.should_ai_respond(sample_ai_entity, message, room_id=1) is False
        assert await service.should_ai_respond(sample_ai_entity, message, conversation_id=1) is False

    async def test_should_not_respond_to_own_messages(self, service, sample_ai_entity):
        """Test AI doesn't respond to its own messages."""
        # Arrange
        own_message = Message(id=1, content="I am responding", sender_ai_id=sample_ai_entity.id)

        # Act & Assert
        assert await service.should_ai_respond(sample_ai_entity, own_message, room_id=1) is False
        assert await service.should_ai_respond(sample_ai_entity, own_message, conversation_id=1) is False

    # ===== Anti-Parroting Tests =====

    def test_clean_parroting_removes_ai_username_prefix(self, service):
        """Test removing AI's own username prefix."""
        # Arrange
        text = "Sokrates: This is my answer"
        ai_username = "Sokrates"

        # Act
        result = service._clean_parroting(text, ai_username)

        # Assert
        assert result == "This is my answer"

    def test_clean_parroting_removes_you_prefix(self, service):
        """Test removing 'You:' prefix."""
        # Arrange
        text = "You: How are you?"
        ai_username = "Sokrates"

        # Act
        result = service._clean_parroting(text, ai_username)

        # Assert
        assert result == "How are you?"

    def test_clean_parroting_removes_ai_prefix(self, service):
        """Test removing 'AI:' prefix."""
        # Arrange
        text = "AI: I am here to help"
        ai_username = "Sokrates"

        # Act
        result = service._clean_parroting(text, ai_username)

        # Assert
        assert result == "I am here to help"

    def test_clean_parroting_removes_assistant_prefix(self, service):
        """Test removing 'Assistant:' prefix."""
        # Arrange
        text = "Assistant: Let me assist you"
        ai_username = "Sokrates"

        # Act
        result = service._clean_parroting(text, ai_username)

        # Assert
        assert result == "Let me assist you"

    def test_clean_parroting_removes_bracket_prefix(self, service):
        """Test removing '[Name]:' bracket prefix."""
        # Arrange
        text = "[Sokrates]: Here is my response"
        ai_username = "Sokrates"

        # Act
        result = service._clean_parroting(text, ai_username)

        # Assert
        assert result == "Here is my response"

    def test_clean_parroting_removes_generic_username_prefix(self, service):
        """Test removing generic 'username:' prefix."""
        # Arrange
        text = "assistant_alpha: This is my answer"
        ai_username = "assistant_alpha"

        # Act
        result = service._clean_parroting(text, ai_username)

        # Assert
        assert result == "This is my answer"

    def test_clean_parroting_case_insensitive(self, service):
        """Test prefix removal is case-insensitive."""
        # Arrange
        text = "SOKRATES: This is my answer"
        ai_username = "Sokrates"

        # Act
        result = service._clean_parroting(text, ai_username)

        # Assert
        assert result == "This is my answer"

    def test_clean_parroting_preserves_normal_text(self, service):
        """Test normal text without prefix is preserved."""
        # Arrange
        text = "This is a normal response without any prefix"
        ai_username = "Sokrates"

        # Act
        result = service._clean_parroting(text, ai_username)

        # Assert
        assert result == "This is a normal response without any prefix"

    def test_clean_parroting_preserves_colon_in_middle(self, service):
        """Test colons in the middle of text are preserved."""
        # Arrange
        text = "The time is 3:45 PM and the ratio is 1:2"
        ai_username = "Sokrates"

        # Act
        result = service._clean_parroting(text, ai_username)

        # Assert
        assert result == "The time is 3:45 PM and the ratio is 1:2"

    def test_clean_parroting_handles_whitespace(self, service):
        """Test proper whitespace handling after prefix removal."""
        # Arrange
        text = "Sokrates:    Extra spaces should be trimmed"
        ai_username = "Sokrates"

        # Act
        result = service._clean_parroting(text, ai_username)

        # Assert
        assert result == "Extra spaces should be trimmed"

    def test_clean_parroting_empty_string(self, service):
        """Test handling of empty string."""
        # Arrange
        text = ""
        ai_username = "Sokrates"

        # Act
        result = service._clean_parroting(text, ai_username)

        # Assert
        assert result == ""
