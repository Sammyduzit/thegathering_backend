"""
AI Context Service for building conversation context.

Retrieves message history and AI memory to provide context for LLM responses.
"""

from datetime import datetime, timezone

import structlog

from app.core.constants import MAX_CONTEXT_MESSAGES
from app.interfaces.memory_retriever import IMemoryRetriever
from app.models.ai_entity import AIEntity
from app.models.ai_memory import AIMemory
from app.repositories.ai_memory_repository import IAIMemoryRepository
from app.repositories.message_repository import IMessageRepository

logger = structlog.get_logger(__name__)


class AIContextService:
    """Service for building AI conversation context."""

    def __init__(
        self,
        message_repo: IMessageRepository,
        memory_repo: IAIMemoryRepository,
        memory_retriever: IMemoryRetriever | None = None,
    ):
        """
        Initialize AI context service.

        :param message_repo: Message repository for fetching conversation history
        :param memory_repo: AI memory repository for persisting memory updates
        :param memory_retriever: Memory retriever implementation (optional, for enhanced retrieval)
        """
        self.message_repo = message_repo
        self.memory_repo = memory_repo
        self.memory_retriever = memory_retriever

    async def build_conversation_context(
        self,
        conversation_id: int,
        ai_entity: AIEntity,
        max_messages: int = MAX_CONTEXT_MESSAGES,
    ) -> list[dict[str, str]]:
        """
        Build conversation context for AI response generation.

        :param conversation_id: Conversation ID to get messages from
        :param ai_entity: AI entity that will respond
        :param max_messages: Maximum number of recent messages to include
        :return: List of message dicts with 'role' and 'content' keys (all use 'user' role)
        """
        # Get recent messages from conversation
        messages, _ = await self.message_repo.get_conversation_messages(
            conversation_id=conversation_id,
            page=1,
            page_size=max_messages,
        )

        # Convert to LLM message format with proper role-tags
        # AI's own messages: "assistant" role (for better LLM in-context learning)
        # Others: "user" role with name prefix
        context_messages = []
        for msg in reversed(messages):  # Reverse to get chronological order
            if msg.sender_ai_id == ai_entity.id:
                # AI's own previous messages: "assistant" role, NO name prefix
                context_messages.append({"role": "assistant", "content": msg.content})
            else:
                # Others: "user" role, WITH name prefix
                if msg.sender_user_id:
                    sender_name = msg.sender_user.username
                elif msg.sender_ai_id:
                    sender_name = f"{msg.sender_ai.username} (AI)"
                else:
                    sender_name = "Unknown"

                content = f"{sender_name}: {msg.content}"
                context_messages.append({"role": "user", "content": content})

        logger.info(
            "conversation_context_built",
            ai_username=ai_entity.username,
            conversation_id=conversation_id,
            message_count=len(context_messages),
        )

        return context_messages

    async def build_room_context(
        self,
        room_id: int,
        ai_entity: AIEntity,
        max_messages: int = MAX_CONTEXT_MESSAGES,
    ) -> list[dict[str, str]]:
        """
        Build room message context for AI response generation.

        :param room_id: Room ID to get messages from
        :param ai_entity: AI entity that will respond
        :param max_messages: Maximum number of recent messages to include
        :return: List of message dicts with 'role' and 'content' keys
        """
        # Get recent messages from room
        messages, _ = await self.message_repo.get_room_messages(
            room_id=room_id,
            page=1,
            page_size=max_messages,
        )

        # Convert to LLM message format with proper role-tags
        # AI's own messages: "assistant" role (for better LLM in-context learning)
        # Others: "user" role with name prefix
        context_messages = []
        for msg in reversed(messages):  # Reverse to get chronological order
            if msg.sender_ai_id == ai_entity.id:
                # AI's own previous messages: "assistant" role, NO name prefix
                context_messages.append({"role": "assistant", "content": msg.content})
            else:
                # Others: "user" role, WITH name prefix
                if msg.sender_user_id:
                    sender_name = msg.sender_user.username
                elif msg.sender_ai_id:
                    sender_name = f"{msg.sender_ai.username} (AI)"
                else:
                    sender_name = "Unknown"

                content = f"{sender_name}: {msg.content}"
                context_messages.append({"role": "user", "content": content})

        logger.info(
            "room_context_built",
            ai_username=ai_entity.username,
            room_id=room_id,
            message_count=len(context_messages),
        )

        return context_messages

    async def get_ai_memories(
        self,
        ai_entity_id: int,
        user_id: int,
        conversation_id: int | None,
        query: str,
        keywords: list[str] | None = None,
    ) -> str:
        """
        Retrieve AI entity's memories using tiered retrieval.

        Uses 3-layer retrieval with cross-layer RRF fusion:
        - Short-term: Recent conversation context
        - Long-term: Past conversations with user
        - Personality: Global knowledge base

        :param ai_entity_id: AI entity ID
        :param user_id: User ID for personalized memories
        :param conversation_id: Current conversation ID
        :param query: Query text for semantic search
        :param keywords: Optional keywords (deprecated, query used instead)
        :return: Formatted memory context string

        Note:
            Memory limits are controlled by settings (short_term_candidates, long_term_candidates, etc.)
        """
        if not self.memory_retriever:
            return ""

        # Tiered retrieval with cross-layer RRF
        memories = await self.memory_retriever.retrieve_tiered(
            entity_id=ai_entity_id,
            user_id=user_id,
            conversation_id=conversation_id,
            query=query,
        )

        if not memories:
            return ""

        # Update access tracking
        await self._update_access_tracking(memories)

        # Format tiered context
        memory_context = self._format_tiered_context(memories)

        logger.info(
            "tiered_memories_retrieved",
            memory_count=len(memories),
            ai_entity_id=ai_entity_id,
            user_id=user_id,
            conversation_id=conversation_id,
        )

        return memory_context

    def _format_tiered_context(self, memories: list[AIMemory]) -> str:
        """
        Format mixed-layer memories with usage instructions, grouped by type for clarity.

        :param memories: Mixed memories from all layers
        :return: Formatted memory context string with usage guidelines
        """
        # Group by type
        short_term = [m for m in memories if m.memory_metadata and m.memory_metadata.get("type") == "short_term"]
        long_term = [m for m in memories if m.memory_metadata and m.memory_metadata.get("type") == "long_term"]
        personality = [m for m in memories if m.memory_metadata and m.memory_metadata.get("type") == "personality"]

        lines = []

        # Header with usage instructions
        if short_term or long_term or personality:
            lines.append("# YOUR MEMORY LAYERS")
            lines.append("Use these memories to personalize your responses and maintain conversation continuity:")
            lines.append("")

        # Short-term (chunked memories)
        if short_term:
            lines.append("## Recent Context (this conversation):")
            lines.append("Use this for immediate conversation flow and continuity. Reference recent topics naturally.")
            lines.append("")

            for mem in short_term:
                # Show chunk metadata
                chunk_info = mem.memory_metadata.get("message_range", "unknown")
                lines.append(f"### Chunk {mem.memory_metadata.get('chunk_index', '?')} (Messages {chunk_info}):")

                # Show FULL messages from this chunk (no truncation!)
                if mem.memory_content and "messages" in mem.memory_content:
                    messages = mem.memory_content["messages"]
                    for msg in messages:  # All messages, full content
                        sender_name = msg.get("sender_name", "Unknown")
                        content = msg.get("content", "")  # FULL content, no truncation!
                        lines.append(f"  - {sender_name}: {content}")
                else:
                    # Fallback to summary if messages not available
                    lines.append(f"  - {mem.summary}")

                lines.append("")

        # Long-term (learned facts)
        if long_term:
            lines.append("## Past Interactions (Learned Facts):")
            lines.append("Draw on these facts when relevant. Markers: [CRITICAL] / [RELEVANT]")
            lines.append("")

            for mem in long_term:
                # Defensive: Check if fact exists (guard against corrupted/old LTM entries)
                fact = mem.memory_content.get("fact")
                if not fact or not fact.get("text"):
                    continue

                theme = fact.get("theme", mem.summary)
                text = fact["text"]
                importance = fact.get("importance", 0.5)

                # ASCII Importance Marker (terminal-friendly)
                if importance >= 0.8:
                    lines.append(f"[CRITICAL] [{theme}] {text}")
                elif importance >= 0.5:
                    lines.append(f"[RELEVANT] [{theme}] {text}")
                else:
                    lines.append(f"[{theme}] {text}")

            lines.append("")

        # Personality
        if personality:
            lines.append("## Your Core Knowledge & Perspective:")
            lines.append(
                "These define your foundational understanding and worldview. "
                "Let these inform your responses naturally without explicitly citing them."
            )
            for mem in personality:
                lines.append(f"- {mem.summary}")

        return "\n".join(lines)

    async def _update_access_tracking(self, memories: list[AIMemory]) -> None:
        """
        Update access tracking for selected memories.

        Increments access_count and updates last_accessed timestamp.

        :param memories: List of memories to update
        """
        now = datetime.now(timezone.utc)

        for memory in memories:
            memory.access_count = (memory.access_count or 0) + 1
            memory.last_accessed = now

            # Update in database (repository expects full AIMemory object)
            await self.memory_repo.update(memory)

        logger.debug("access_tracking_updated", memory_count=len(memories))

    def build_retrieval_query(
        self, messages: list[dict[str, str]], use_last_n: int = 3
    ) -> str:
        """
        Build enhanced RAG query from recent messages for better context.

        Combines last N messages, prioritizing questions, and removing name prefixes.
        This helps with follow-up questions like "What about its performance?"
        which need context from previous messages.

        :param messages: Message history (in chronological order)
        :param use_last_n: Number of recent messages to include in query (default: 3)
        :return: Combined query string for RAG retrieval
        """
        if not messages:
            return ""

        recent_messages = messages[-use_last_n:]
        query_parts = []

        for msg in recent_messages:
            content = msg.get("content", "")

            # Remove name prefixes ("Alice: ..." → "...")
            if ": " in content:
                content = content.split(": ", 1)[1]

            # Prioritize questions (prepend to query)
            if "?" in content:
                query_parts.insert(0, content)
            else:
                query_parts.append(content)

        return "\n".join(query_parts)

    async def build_full_context(
        self,
        conversation_id: int | None,
        room_id: int | None,
        ai_entity: AIEntity,
        user_id: int | None = None,
        include_memories: bool = True,
    ) -> tuple[list[dict[str, str]], str | None]:
        """
        Build complete context including messages and memories.

        :param conversation_id: Conversation ID (for private/group chats)
        :param room_id: Room ID (for public room messages)
        :param ai_entity: AI entity that will respond
        :param user_id: User ID for personalized memories (optional, NULL for rooms = only personality memories)
        :param include_memories: Whether to include AI memories in system prompt
        :return: Tuple of (message_context, memory_context) where memory_context may be None
        :raises ValueError: If both conversation_id and room_id are None
        """
        # Get message context
        if conversation_id:
            messages = await self.build_conversation_context(conversation_id, ai_entity)
        elif room_id:
            messages = await self.build_room_context(room_id, ai_entity)
        else:
            raise ValueError("Either conversation_id or room_id must be provided")

        # Get memory context if enabled
        memory_context = None
        if include_memories and messages and user_id:
            # Build enhanced query from last 3 messages for better RAG context
            query = self.build_retrieval_query(messages, use_last_n=3)
            memory_context = await self.get_ai_memories(
                ai_entity_id=ai_entity.id,
                user_id=user_id,
                conversation_id=conversation_id,
                query=query,
            )

        return messages, memory_context
