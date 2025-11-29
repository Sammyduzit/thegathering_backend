"""Unit tests for AIEntityService."""

from unittest.mock import AsyncMock

import pytest

from app.core.exceptions import (
    AIEntityNotFoundException,
    AIEntityOfflineException,
    ConversationNotFoundException,
    DuplicateResourceException,
    InvalidOperationException,
)
from app.models.ai_entity import AIEntity, AIEntityStatus
from app.models.conversation import Conversation, ConversationType
from app.services.ai.ai_entity_service import AIEntityService


@pytest.mark.unit
class TestAIEntityService:
    """Unit tests for AI entity service business logic."""

    @pytest.fixture
    def mock_ai_repo(self):
        """Create mock AI entity repository."""
        return AsyncMock()

    @pytest.fixture
    def mock_conversation_repo(self):
        """Create mock conversation repository."""
        return AsyncMock()

    @pytest.fixture
    def mock_cooldown_repo(self):
        """Create mock cooldown repository."""
        return AsyncMock()

    @pytest.fixture
    def mock_room_repo(self):
        """Create mock room repository."""
        return AsyncMock()

    @pytest.fixture
    def mock_message_repo(self):
        """Create mock message repository."""
        return AsyncMock()

    @pytest.fixture
    def mock_conversation_service(self):
        """Create mock conversation service."""
        service = AsyncMock()
        service._enqueue_long_term_memory_for_ai = AsyncMock()
        return service

    @pytest.fixture
    def service(
        self,
        mock_ai_repo,
        mock_conversation_repo,
        mock_cooldown_repo,
        mock_room_repo,
        mock_message_repo,
        mock_conversation_service,
    ):
        """Create service instance with mocked dependencies."""
        return AIEntityService(
            ai_entity_repo=mock_ai_repo,
            conversation_repo=mock_conversation_repo,
            cooldown_repo=mock_cooldown_repo,
            room_repo=mock_room_repo,
            message_repo=mock_message_repo,
            conversation_service=mock_conversation_service,
        )

    async def test_get_all_entities(self, service, mock_ai_repo):
        """Test getting all AI entities."""
        # Arrange
        mock_entities = [
            AIEntity(id=1, username="ai1", system_prompt="Test", model_name="gpt-4"),
            AIEntity(id=2, username="ai2", system_prompt="Test", model_name="gpt-4"),
        ]
        mock_ai_repo.get_all.return_value = mock_entities

        # Act
        result = await service.get_all_entities()

        # Assert
        assert len(result) == 2
        mock_ai_repo.get_all.assert_called_once()

    async def test_get_available_entities(self, service, mock_ai_repo):
        """Test getting available AI entities (online and not deleted)."""
        # Arrange
        mock_entities = [
            AIEntity(
                id=1,
                username="ai1",
                system_prompt="Test",
                model_name="gpt-4",
                status=AIEntityStatus.ONLINE,
            )
        ]
        mock_ai_repo.get_available_entities.return_value = mock_entities

        # Act
        result = await service.get_available_entities()

        # Assert
        assert len(result) == 1
        assert result[0].status == AIEntityStatus.ONLINE
        mock_ai_repo.get_available_entities.assert_called_once()

    async def test_get_entity_by_id_success(self, service, mock_ai_repo):
        """Test getting AI entity by ID successfully."""
        # Arrange
        mock_entity = AIEntity(id=1, username="ai1", system_prompt="Test", model_name="gpt-4")
        mock_ai_repo.get_by_id.return_value = mock_entity

        # Act
        result = await service.get_entity_by_id(1)

        # Assert
        assert result.id == 1
        mock_ai_repo.get_by_id.assert_called_once_with(1)

    async def test_get_entity_by_id_not_found(self, service, mock_ai_repo):
        """Test getting AI entity by ID raises AIEntityNotFoundException when not found."""
        # Arrange
        mock_ai_repo.get_by_id.return_value = None

        # Act & Assert
        with pytest.raises(AIEntityNotFoundException) as exc_info:
            await service.get_entity_by_id(999)

        assert "999" in str(exc_info.value)
        assert exc_info.value.error_code == "AI_ENTITY_NOT_FOUND"

    async def test_create_entity_success(self, service, mock_ai_repo):
        """Test creating AI entity successfully."""
        # Arrange
        mock_ai_repo.username_exists.return_value = False
        mock_entity = AIEntity(id=1, username="new_ai", system_prompt="Test", model_name="gpt-4")
        mock_ai_repo.create.return_value = mock_entity

        # Act
        result = await service.create_entity(
            username="new_ai",
            system_prompt="Test",
            model_name="gpt-4",
        )

        # Assert
        assert result.username == "new_ai"
        mock_ai_repo.username_exists.assert_called_once_with("new_ai")
        mock_ai_repo.create.assert_called_once()

    async def test_create_entity_duplicate_name(self, service, mock_ai_repo):
        """Test creating AI entity with duplicate name raises DuplicateResourceException."""
        # Arrange
        mock_ai_repo.username_exists.return_value = True

        # Act & Assert
        with pytest.raises(DuplicateResourceException) as exc_info:
            await service.create_entity(
                username="existing_ai",
                system_prompt="Test",
                model_name="gpt-4",
            )

        assert exc_info.value.error_code == "DUPLICATE_RESOURCE"
        assert "existing_ai" in str(exc_info.value)

    async def test_update_entity_success(self, service, mock_ai_repo):
        """Test updating AI entity successfully."""
        # Arrange
        mock_entity = AIEntity(id=1, username="ai1", system_prompt="Test", model_name="gpt-4")
        mock_ai_repo.get_by_id.return_value = mock_entity
        mock_ai_repo.update.return_value = mock_entity

        # Act
        result = await service.update_entity(entity_id=1, username="updated_name")

        # Assert
        assert result.username == "updated_name"
        mock_ai_repo.update.assert_called_once()

    async def test_delete_entity_success(self, service, mock_ai_repo):
        """Test deleting AI entity successfully."""
        # Arrange
        mock_entity = AIEntity(id=1, username="ai1", system_prompt="Test", model_name="gpt-4")
        mock_ai_repo.get_by_id.return_value = mock_entity
        mock_ai_repo.delete.return_value = True

        # Act
        result = await service.delete_entity(1)

        # Assert
        assert "deleted" in result["message"]
        assert result["entity_id"] == 1
        mock_ai_repo.delete.assert_called_once_with(1)

    async def test_get_available_in_room(self, service, mock_ai_repo):
        """Test getting available AI entities in a room."""
        # Arrange
        mock_entities = [
            AIEntity(
                id=1,
                username="ai1",
                system_prompt="Test",
                model_name="gpt-4",
                status=AIEntityStatus.ONLINE,
                current_room_id=1,
            )
        ]
        mock_ai_repo.get_available_in_room.return_value = mock_entities

        # Act
        result = await service.get_available_in_room(1)

        # Assert
        assert len(result) == 1
        assert result[0].current_room_id == 1
        mock_ai_repo.get_available_in_room.assert_called_once_with(1)

    async def test_invite_to_conversation_success(self, service, mock_ai_repo, mock_conversation_repo):
        """Test inviting AI to conversation successfully."""
        # Arrange
        mock_entity = AIEntity(
            id=1,
            username="ai1",
            system_prompt="Test",
            model_name="gpt-4",
            status=AIEntityStatus.ONLINE,
        )
        mock_conversation = Conversation(
            id=1, room_id=1, conversation_type=ConversationType.PRIVATE, max_participants=2
        )

        mock_ai_repo.get_by_id.return_value = mock_entity
        mock_conversation_repo.get_by_id.return_value = mock_conversation
        mock_ai_repo.get_ai_in_conversation.return_value = None
        mock_conversation_repo.add_participant.return_value = None

        # Act
        result = await service.invite_to_conversation(conversation_id=1, ai_entity_id=1)

        # Assert
        assert "invited" in result["message"]
        assert result["conversation_id"] == 1
        assert result["ai_entity_id"] == 1
        mock_conversation_repo.add_participant.assert_called_once_with(1, ai_entity_id=1)

    async def test_invite_to_conversation_ai_offline(self, service, mock_ai_repo):
        """Test inviting offline AI to conversation raises 400."""
        # Arrange
        mock_entity = AIEntity(
            id=1,
            username="ai1",
            system_prompt="Test",
            model_name="gpt-4",
            status=AIEntityStatus.OFFLINE,
        )
        mock_ai_repo.get_by_id.return_value = mock_entity

        # Act & Assert
        with pytest.raises(AIEntityOfflineException) as exc_info:
            await service.invite_to_conversation(conversation_id=1, ai_entity_id=1)

        assert exc_info.value.error_code == "AI_ENTITY_OFFLINE"
        assert "ai1" in str(exc_info.value)

    async def test_invite_to_conversation_not_found(self, service, mock_ai_repo, mock_conversation_repo):
        """Test inviting AI to non-existent conversation raises 404."""
        # Arrange
        mock_entity = AIEntity(
            id=1,
            username="ai1",
            system_prompt="Test",
            model_name="gpt-4",
            status=AIEntityStatus.ONLINE,
        )
        mock_ai_repo.get_by_id.return_value = mock_entity
        mock_conversation_repo.get_by_id.return_value = None

        # Act & Assert
        with pytest.raises(ConversationNotFoundException) as exc_info:
            await service.invite_to_conversation(conversation_id=999, ai_entity_id=1)

        assert exc_info.value.error_code == "CONVERSATION_NOT_FOUND"
        assert "999" in str(exc_info.value)

    async def test_invite_to_conversation_ai_already_present(self, service, mock_ai_repo, mock_conversation_repo):
        """Test inviting AI to conversation where AI already exists raises 409."""
        # Arrange
        mock_entity = AIEntity(
            id=1,
            username="ai1",
            system_prompt="Test",
            model_name="gpt-4",
            status=AIEntityStatus.ONLINE,
        )
        mock_conversation = Conversation(
            id=1, room_id=1, conversation_type=ConversationType.PRIVATE, max_participants=2
        )
        existing_ai = AIEntity(
            id=2,
            username="ai2",
            system_prompt="Test",
            model_name="gpt-4",
            status=AIEntityStatus.ONLINE,
        )

        mock_ai_repo.get_by_id.return_value = mock_entity
        mock_conversation_repo.get_by_id.return_value = mock_conversation
        mock_ai_repo.get_ai_in_conversation.return_value = existing_ai

        # Act & Assert
        with pytest.raises(InvalidOperationException) as exc_info:
            await service.invite_to_conversation(conversation_id=1, ai_entity_id=1)

        assert exc_info.value.error_code == "INVALID_OPERATION"
        assert "already in this conversation" in str(exc_info.value)

    async def test_remove_from_conversation_success(self, service, mock_ai_repo, mock_conversation_repo):
        """Test removing AI from conversation successfully."""
        # Arrange
        mock_entity = AIEntity(id=1, username="ai1", system_prompt="Test", model_name="gpt-4")
        mock_conversation = Conversation(
            id=1, room_id=1, conversation_type=ConversationType.PRIVATE, max_participants=2
        )

        mock_ai_repo.get_by_id.return_value = mock_entity
        mock_conversation_repo.get_by_id.return_value = mock_conversation
        mock_conversation_repo.remove_participant.return_value = None

        # Act
        result = await service.remove_from_conversation(conversation_id=1, ai_entity_id=1)

        # Assert
        assert "removed" in result["message"]
        assert result["conversation_id"] == 1
        assert result["ai_entity_id"] == 1
        mock_conversation_repo.remove_participant.assert_called_once_with(1, ai_entity_id=1)

    async def test_update_cooldown_room_context(self, service, mock_cooldown_repo):
        """Test updating cooldown for room context."""
        # Arrange
        mock_cooldown_repo.upsert_cooldown.return_value = None

        # Act
        await service.update_cooldown(ai_entity_id=1, room_id=1)

        # Assert
        mock_cooldown_repo.upsert_cooldown.assert_called_once_with(
            ai_entity_id=1,
            room_id=1,
            conversation_id=None,
        )

    async def test_update_cooldown_conversation_context(self, service, mock_cooldown_repo):
        """Test updating cooldown for conversation context."""
        # Arrange
        mock_cooldown_repo.upsert_cooldown.return_value = None

        # Act
        await service.update_cooldown(ai_entity_id=1, conversation_id=1)

        # Assert
        mock_cooldown_repo.upsert_cooldown.assert_called_once_with(
            ai_entity_id=1,
            room_id=None,
            conversation_id=1,
        )

    # ===== Checkpoint 2 Tests: AI Room Assignment =====

    async def test_assign_ai_to_room_success(self, service, mock_ai_repo, mock_room_repo):
        """Test assigning AI to room successfully."""
        from app.models.room import Room

        # Arrange
        mock_entity = AIEntity(
            id=1,
            username="ai1",
            system_prompt="Test",
            model_name="gpt-4",
            status=AIEntityStatus.ONLINE,
        )
        mock_room = Room(id=1, name="Test Room", has_ai=False)

        mock_ai_repo.get_by_id.return_value = mock_entity
        mock_room_repo.get_by_id.return_value = mock_room
        mock_ai_repo.update.return_value = mock_entity

        # Act
        result = await service.update_entity(entity_id=1, current_room_id=1)

        # Assert
        assert result.current_room_id == 1
        assert mock_room.has_ai is True
        mock_room_repo.get_by_id.assert_called_once_with(1)
        mock_ai_repo.update.assert_called_once()

    async def test_assign_ai_to_room_already_has_ai(self, service, mock_ai_repo, mock_room_repo):
        """Test assigning AI to room that already has AI raises error."""
        from app.models.room import Room

        # Arrange
        mock_entity = AIEntity(
            id=1,
            username="ai1",
            system_prompt="Test",
            model_name="gpt-4",
            status=AIEntityStatus.ONLINE,
        )
        mock_room = Room(id=1, name="Test Room", has_ai=True)  # Already has AI

        mock_ai_repo.get_by_id.return_value = mock_entity
        mock_room_repo.get_by_id.return_value = mock_room

        # Act & Assert
        with pytest.raises(InvalidOperationException) as exc_info:
            await service.update_entity(entity_id=1, current_room_id=1)

        assert "already has an AI entity" in str(exc_info.value)

    async def test_assign_ai_offline_cannot_join(self, service, mock_ai_repo, mock_room_repo):
        """Test offline AI cannot join a room."""
        from app.models.room import Room

        # Arrange
        mock_entity = AIEntity(
            id=1,
            username="ai1",
            system_prompt="Test",
            model_name="gpt-4",
            status=AIEntityStatus.OFFLINE,  # Offline
        )
        mock_room = Room(id=1, name="Test Room", has_ai=False)

        mock_ai_repo.get_by_id.return_value = mock_entity
        mock_room_repo.get_by_id.return_value = mock_room

        # Act & Assert
        with pytest.raises(InvalidOperationException) as exc_info:
            await service.update_entity(entity_id=1, current_room_id=1)

        assert "must be ONLINE" in str(exc_info.value)

    async def test_remove_ai_from_room(self, service, mock_ai_repo, mock_room_repo):
        """Test removing AI from room."""
        from app.models.room import Room

        # Arrange
        mock_room = Room(id=1, name="Test Room", has_ai=True)
        mock_entity = AIEntity(
            id=1,
            username="ai1",
            system_prompt="Test",
            model_name="gpt-4",
            status=AIEntityStatus.ONLINE,
            current_room_id=1,
        )

        mock_ai_repo.get_by_id.return_value = mock_entity
        mock_room_repo.get_by_id.return_value = mock_room
        mock_ai_repo.update.return_value = mock_entity

        # Act
        result = await service.update_entity(entity_id=1, current_room_id=None)

        # Assert
        assert result.current_room_id is None
        assert mock_room.has_ai is False
        mock_ai_repo.update.assert_called_once()

    async def test_update_status_offline_auto_leaves_room(self, service, mock_ai_repo, mock_room_repo):
        """Test setting AI status to OFFLINE automatically removes from room."""
        from app.models.room import Room

        # Arrange
        mock_room = Room(id=1, name="Test Room", has_ai=True)
        mock_entity = AIEntity(
            id=1,
            username="ai1",
            system_prompt="Test",
            model_name="gpt-4",
            status=AIEntityStatus.ONLINE,
            current_room_id=1,
        )

        mock_ai_repo.get_by_id.return_value = mock_entity
        mock_room_repo.get_by_id.return_value = mock_room
        mock_ai_repo.update.return_value = mock_entity

        # Act
        result = await service.update_entity(entity_id=1, status=AIEntityStatus.OFFLINE)

        # Assert
        assert result.status == AIEntityStatus.OFFLINE
        assert result.current_room_id is None
        assert mock_room.has_ai is False
        mock_ai_repo.update.assert_called_once()
