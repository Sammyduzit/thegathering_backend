"""
Concrete implementations of service interfaces.

This module provides production-ready implementations of the defined
interfaces, such as DeepL translator, external API clients, etc.
"""

from .deepl_translator import DeepLTranslator

__all__ = [
    "DeepLTranslator",
]
