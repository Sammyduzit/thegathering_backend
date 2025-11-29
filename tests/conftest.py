"""
Global test configuration and fixtures.

This module contains ONLY global, test-agnostic fixtures that are shared
across ALL test types (unit/integration/e2e).

Test-specific fixtures are located in their respective conftest.py files:
- tests/unit/conftest.py       - Unit test fixtures (SQLite, mocks)
- tests/integration/conftest.py - Integration test fixtures (PostgreSQL, real services)
- tests/e2e/conftest.py         - E2E test fixtures (PostgreSQL + FastAPI HTTP client)
"""

import pytest

from tests.fixtures import AIFactory, ConversationFactory, MessageFactory, RoomFactory, UserFactory

# ============================================================================
# Sample Data Fixtures (No Database Dependencies)
# ============================================================================


@pytest.fixture
def sample_user_data():
    """Standard user registration data for all test types."""
    return {
        "email": "user@example.com",
        "username": "testuser",
        "password": "password123",
    }


@pytest.fixture
def sample_admin_data():
    """Standard admin registration data for all test types."""
    return {
        "email": "admin@example.com",
        "username": "testadmin",
        "password": "adminpass123",
    }


@pytest.fixture
def sample_room_data():
    """Standard room creation data for all test types."""
    return {"name": "Test Room", "description": "A test room for testing purposes", "max_users": 10}


@pytest.fixture
def sample_message_data():
    """Standard message data for all test types."""
    return {"content": "Test message content for testing purposes"}


@pytest.fixture
def sample_conversation_data():
    """Standard conversation data for all test types."""
    return {"conversation_type": "PRIVATE", "max_participants": 2}


@pytest.fixture
def sample_ai_entity_data():
    """Standard AI entity data for all test types."""
    return {
        "username": "test_ai",
        "system_prompt": "You are a test AI assistant",
        "model_name": "gpt-4",
    }


@pytest.fixture
def sample_ai_memory_data():
    """Standard AI memory data for all test types."""
    return {
        "content": "Test memory content",
        "importance": 2,
        "memory_type": "conversation",
    }


# ============================================================================
# Shared Factory Fixtures (available to all test types)
# ============================================================================


@pytest.fixture
def user_factory():
    """User factory for creating test users."""
    return UserFactory


@pytest.fixture
def room_factory():
    """Room factory for creating test rooms."""
    return RoomFactory


@pytest.fixture
def message_factory():
    """Message factory for creating test messages."""
    return MessageFactory


@pytest.fixture
def conversation_factory():
    """Conversation factory for creating test conversations."""
    return ConversationFactory


@pytest.fixture
def ai_entity_factory():
    """AI entity factory for creating test AI entities."""
    return AIFactory


# ============================================================================
# Global Test Configuration
# ============================================================================


@pytest.fixture(scope="session")
def test_config():
    """Global test configuration settings."""
    return {
        "test_database_prefix": "test_",
        "max_test_duration": 300,  # 5 minutes
        "cleanup_on_failure": True,
        "log_level": "INFO",
    }


# ============================================================================
# Pytest Hooks
# ============================================================================


def pytest_configure(config):
    """
    Configure pytest with custom markers.

    This hook is called after command line options have been parsed
    and all plugins and initial conftest files have been loaded.
    """
    # Markers are already defined in pytest.ini, but we register them
    # here programmatically for IDE support and documentation
    config.addinivalue_line("markers", "unit: Unit tests with mocked dependencies (fast)")
    config.addinivalue_line("markers", "integration: Integration tests with real database (medium)")
    config.addinivalue_line("markers", "e2e: End-to-end tests with full API (slow)")
    config.addinivalue_line("markers", "slow: Tests that take longer to run")
    config.addinivalue_line("markers", "ci: Tests for CI environment")


def pytest_collection_modifyitems(config, items):
    """
    Auto-mark tests based on their file location.

    This ensures tests are properly categorized even if developers
    forget to add the @pytest.mark.xxx decorator.
    """
    for item in items:
        test_path = str(item.fspath)

        # Auto-mark based on directory
        if "/unit/" in test_path:
            item.add_marker(pytest.mark.unit)
        elif "/integration/" in test_path:
            item.add_marker(pytest.mark.integration)
        elif "/e2e/" in test_path:
            item.add_marker(pytest.mark.e2e)

        # Auto-mark async tests
        if "async" in item.name or hasattr(item.obj, "__wrapped__"):
            item.add_marker(pytest.mark.asyncio)
