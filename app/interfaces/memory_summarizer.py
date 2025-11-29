"""
Memory Summarizer interface for generating conversation summaries.

This interface abstracts summary generation functionality, allowing different
summarizers (heuristic, LLM-based, etc.) to be used interchangeably through dependency injection.
"""

from abc import ABC, abstractmethod

from app.models.ai_entity import AIEntity
from app.models.message import Message


class IMemorySummarizer(ABC):
    """Abstract interface for conversation memory summarization services."""

    @abstractmethod
    async def summarize(
        self,
        messages: list[Message],
        ai_entity: AIEntity | None = None,
    ) -> str:
        """
        Generate summary from conversation messages.

        :param messages: List of messages to summarize (chronological order)
        :param ai_entity: AI entity object (for contextualized summaries using username)
        :return: Generated summary text (1-2 sentences), e.g., 'Assistant Alpha discussed FastAPI setup and database configuration with Alice'
        :raises MemorySummarizationError: If summarization fails
        """
        pass


class MemorySummarizationError(Exception):
    """Exception raised when memory summarization operations fail."""

    def __init__(self, message: str, original_error: Exception | None = None):
        super().__init__(message)
        self.original_error = original_error
