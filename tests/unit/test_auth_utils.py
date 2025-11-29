"""
Unit tests for auth_utils module.

Tests focus on password hashing and verification using bcrypt.
"""

import pytest

from app.core.auth_utils import hash_password, verify_password


@pytest.mark.unit
class TestAuthUtils:
    """Unit tests for authentication utilities."""

    def test_hash_password_success(self):
        """Test successful password hashing."""
        # Arrange
        password = "mysecretpassword"

        # Act
        hashed = hash_password(password)

        # Assert
        assert hashed is not None
        assert isinstance(hashed, str)
        assert hashed != password
        assert hashed.startswith("$2b$")  # bcrypt hash format

    def test_hash_password_different_hashes(self):
        """Test that same password produces different hashes (salt)."""
        # Arrange
        password = "samepassword"

        # Act
        hash1 = hash_password(password)
        hash2 = hash_password(password)

        # Assert
        assert hash1 != hash2  # Different salts

    def test_hash_password_too_long(self):
        """Test that password exceeding 72 bytes raises ValueError."""
        # Arrange
        password = "a" * 73  # 73 bytes (bcrypt limit is 72)

        # Act & Assert
        with pytest.raises(ValueError, match="Password too long"):
            hash_password(password)

    def test_verify_password_correct(self):
        """Test password verification with correct password."""
        # Arrange
        password = "correctpassword"
        hashed = hash_password(password)

        # Act
        result = verify_password(password, hashed)

        # Assert
        assert result is True

    def test_verify_password_incorrect(self):
        """Test password verification with incorrect password."""
        # Arrange
        password = "correctpassword"
        wrong_password = "wrongpassword"
        hashed = hash_password(password)

        # Act
        result = verify_password(wrong_password, hashed)

        # Assert
        assert result is False

    def test_verify_password_empty_string(self):
        """Test password verification with empty string."""
        # Arrange
        password = "nonemptypassword"
        hashed = hash_password(password)

        # Act
        result = verify_password("", hashed)

        # Assert
        assert result is False

    def test_hash_and_verify_workflow(self):
        """Test complete hash and verify workflow."""
        # Arrange
        password = "testpassword123"

        # Act
        hashed = hash_password(password)
        is_valid = verify_password(password, hashed)

        # Assert
        assert is_valid is True

    def test_hash_special_characters(self):
        """Test password hashing with special characters."""
        # Arrange
        password = "p@ssw0rd!#$%^&*()"

        # Act
        hashed = hash_password(password)
        is_valid = verify_password(password, hashed)

        # Assert
        assert is_valid is True

    def test_hash_unicode_characters(self):
        """Test password hashing with unicode characters."""
        # Arrange
        password = "pässwörd123"

        # Act
        hashed = hash_password(password)
        is_valid = verify_password(password, hashed)

        # Assert
        assert is_valid is True
