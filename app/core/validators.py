from typing import Annotated

from markupsafe import escape
from pydantic import AfterValidator

from app.core.constants import SUPPORTED_LANGUAGES


def validate_language_code(language_code: str) -> bool:
    """Validate if language code is supported by DeepL API"""
    return language_code.lower() in SUPPORTED_LANGUAGES


def get_language_name(language_code: str) -> str:
    """Get full language name"""
    return SUPPORTED_LANGUAGES.get(language_code.lower(), "Unknown")


def sanitize_html_content(content: str | None) -> str | None:
    """
    Sanitize HTML content by escaping special characters.
    :param content: Raw user input
    :return: Sanitized content
    """
    if content is None:
        return content
    return str(escape(content)).strip()


def sanitize_username(name: str) -> str:
    """
    Sanitize username
    :param name: Raw user input
    :return: Sanitized username
    """
    return str(escape(name)).strip()


def sanitize_room_text(text: str | None) -> str | None:
    """
    Sanitize room text fields (name, description)
    :param text: Raw text input for room infos
    :return: Sanitized text
    """
    if text is None:
        return text
    return str(escape(text)).strip()


SanitizedString = Annotated[str, AfterValidator(sanitize_html_content)]
SanitizedOptionalString = Annotated[str | None, AfterValidator(sanitize_html_content)]
SanitizedUsername = Annotated[str, AfterValidator(sanitize_username)]
SanitizedRoomText = Annotated[str | None, AfterValidator(sanitize_room_text)]
