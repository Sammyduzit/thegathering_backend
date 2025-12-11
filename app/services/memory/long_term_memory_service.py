import asyncio
import hashlib
import json
import re
import structlog

from app.core.config import settings
from app.interfaces.embedding_service import IEmbeddingService
from app.interfaces.keyword_extractor import IKeywordExtractor
from app.models.ai_memory import AIMemory
from app.repositories.ai_memory_repository import AIMemoryRepository
from app.services.memory.base_memory_service import BaseMemoryService

logger = structlog.get_logger()


class LongTermMemoryService(BaseMemoryService):
    """Service for LLM-based fact extraction from short-term memory chunks."""

    def __init__(
        self,
        memory_repo: AIMemoryRepository,
        embedding_service: IEmbeddingService,
        keyword_extractor: IKeywordExtractor,
        ai_provider,
    ):
        """
        Initialize long-term memory service.

        :param memory_repo: AI memory repository
        :param embedding_service: Embedding service (Google/OpenAI)
        :param keyword_extractor: Keyword extractor (YAKE by default)
        :param ai_provider: AI provider for fact extraction (Google Gemini or OpenAI)
        """
        super().__init__(keyword_extractor)
        self.memory_repo = memory_repo
        self.embedding_service = embedding_service
        self.ai_provider = ai_provider

    async def create_long_term_from_chunks(
        self,
        entity_id: int,
        user_ids: list[int],
        conversation_id: int,
        stm_chunks: list[AIMemory],
    ) -> list[AIMemory]:
        """
        Create long-term memory from short-term memory chunks via LLM fact extraction.

        Core principle: 1 Fact = 1 LTM Entry = 1 Embedding

        :param entity_id: AI entity ID
        :param user_ids: List of user IDs (participants)
        :param conversation_id: Conversation ID
        :param stm_chunks: Short-term memory chunks
        :return: List of created LTM AIMemory instances
        """
        if not stm_chunks:
            return []

        ltm_memories = []

        for stm_chunk in stm_chunks:
            chunk_messages = stm_chunk.memory_content.get("messages", [])
            chunk_index = stm_chunk.memory_metadata.get("chunk_index", 0)
            message_range = stm_chunk.memory_metadata.get("message_range", "unknown")

            if not chunk_messages:
                continue

            # Extract message IDs for audit trail
            message_ids = [msg.get("message_id") for msg in chunk_messages if msg.get("message_id")]

            # Extract facts from this chunk via LLM
            facts = await self._extract_facts_from_chunk(chunk_messages)

            # Create LTM entry per fact
            for fact in facts:
                ltm_entry = await self._create_ltm_entry(
                    entity_id=entity_id,
                    user_ids=user_ids,
                    conversation_id=conversation_id,
                    chunk_index=chunk_index,
                    message_range=message_range,
                    fact=fact,
                    message_ids=message_ids,
                )

                if ltm_entry:
                    ltm_memories.append(ltm_entry)

        return ltm_memories

    async def _extract_facts_from_chunk(self, chunk_messages: list[dict]) -> list[dict]:
        """
        Extract facts from chunk messages via LLM with retry logic.

        Uses exponential backoff for retries before falling back to heuristic extraction.

        :param chunk_messages: Messages with sender_name and content
        :return: List of facts [{"text": str, "importance": float, "participants": list, "theme": str}]
        """
        # Format messages for LLM (use .get() for safety)
        formatted_messages = "\n".join([
            f"{msg.get('sender_name', 'Unknown')}: {msg.get('content', '')}"
            for msg in chunk_messages
            if msg.get('content')  # Skip empty messages
        ])

        if not formatted_messages:
            return []

        # Build prompt
        prompt = self._build_fact_extraction_prompt(formatted_messages)

        # Retry loop with exponential backoff
        last_exception = None
        for attempt in range(settings.ltm_extraction_max_retries):
            try:
                # Call LLM for fact extraction
                llm_response = await self.ai_provider.generate_response(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=settings.ltm_extraction_temperature,
                )

                # Parse facts from response
                facts = self._parse_facts_response(llm_response)

                # Filter by importance threshold and max facts
                filtered_facts = [
                    f for f in facts
                    if f.get("importance", 0.0) >= settings.ltm_min_importance_threshold
                    and f.get("text")  # Must have text
                ]
                filtered_facts = filtered_facts[:settings.ltm_max_facts_per_chunk]

                # Fallback to heuristic if LLM returned no usable facts
                if not filtered_facts:
                    logger.info("ltm_llm_returned_no_facts_using_heuristic")
                    return self._extract_facts_heuristic(chunk_messages)

                # Success - return filtered facts
                if attempt > 0:
                    logger.info("ltm_fact_extraction_succeeded_after_retry", attempt=attempt + 1)
                return filtered_facts

            except Exception as e:
                last_exception = e
                logger.warning(
                    "ltm_fact_extraction_attempt_failed",
                    attempt=attempt + 1,
                    max_retries=settings.ltm_extraction_max_retries,
                    error=str(e),
                )

                # If not last attempt, wait with exponential backoff
                if attempt < settings.ltm_extraction_max_retries - 1:
                    retry_delay = settings.ltm_extraction_retry_delay * (2**attempt)
                    logger.debug("ltm_retrying_after_delay", delay_seconds=retry_delay)
                    await asyncio.sleep(retry_delay)

        # All retries exhausted - fall back to heuristic
        logger.warning(
            "ltm_all_retries_exhausted_using_heuristic",
            retries=settings.ltm_extraction_max_retries,
            last_error=str(last_exception),
        )
        return self._extract_facts_heuristic(chunk_messages)

    async def _create_ltm_entry(
        self,
        entity_id: int,
        user_ids: list[int],
        conversation_id: int,
        chunk_index: int,
        message_range: str,
        fact: dict,
        message_ids: list[int],
    ) -> AIMemory | None:
        """
        Create single LTM entry for a fact.

        :param entity_id: AI entity ID
        :param user_ids: User IDs
        :param conversation_id: Conversation ID
        :param chunk_index: STM chunk index
        :param message_range: Message range (e.g., "72-95")
        :param fact: {"text": str, "importance": float, "participants": list, "theme": str}
        :param message_ids: Original message IDs for audit trail
        :return: Created AIMemory or None if duplicate or invalid
        """
        # Validate fact has required fields
        fact_text = fact.get("text")
        if not fact_text:
            logger.warning("ltm_fact_missing_text_skipping", fact=fact)
            return None

        fact_importance = fact.get("importance", 0.5)  # Default to medium
        fact_participants = fact.get("participants", [])
        fact_theme = fact.get("theme", "General")

        # Calculate fact hash for idempotence
        fact_hash = self._get_fact_hash(fact_text)

        # Idempotence check
        existing_fact = await self.memory_repo.get_ltm_fact(
            entity_id=entity_id,
            conversation_id=conversation_id,
            chunk_index=chunk_index,
            fact_hash=fact_hash,
        )

        if existing_fact:
            logger.debug("ltm_fact_already_exists_skipping", fact_hash=fact_hash)
            return None

        # Extract keywords from fact text
        keywords = await self._extract_keywords(fact_text)

        # Generate embedding with participant context
        participants_str = ", ".join(fact_participants) if fact_participants else ""
        embedding_text = f"{fact_text} | Participants: {participants_str}" if participants_str else fact_text
        embedding = await self.embedding_service.embed_text(embedding_text)

        # Create LTM memory
        ltm_memory = AIMemory(
            entity_id=entity_id,
            user_ids=user_ids,
            conversation_id=conversation_id,
            summary=fact_theme,
            memory_content={
                "fact": {
                    "text": fact_text,
                    "importance": fact_importance,
                    "participants": fact_participants,
                    "theme": fact_theme,
                },
                "conversation_id": conversation_id,
                "chunk_index": chunk_index,
                "message_range": message_range,
                "message_ids": message_ids,
            },
            memory_metadata={
                "type": "long_term",
                "from_short_term": True,
                "conversation_id": conversation_id,
                "chunk_index": chunk_index,
                "fact_hash": fact_hash,
            },
            importance_score=fact_importance,
            keywords=keywords,
            embedding=embedding,
        )

        created_memory = await self.memory_repo.create(ltm_memory)
        logger.info(
            "ltm_fact_created",
            entity_id=entity_id,
            conversation_id=conversation_id,
            chunk_index=chunk_index,
            fact_hash=fact_hash,
            importance=fact_importance,
        )

        return created_memory

    def _build_fact_extraction_prompt(self, formatted_messages: str) -> str:
        """
        Build German prompt for LLM fact extraction.

        :param formatted_messages: Formatted messages (sender_name: content)
        :return: Prompt string
        """
        return f"""Analysiere diese Conversation Messages und extrahiere gelernte Fakten.

TEILNEHMER:
Die Namen aller Teilnehmer (User und KI-Assistenten) sind in den Messages erkennbar.

AUFGABE:
Extrahiere 0-{settings.ltm_max_facts_per_chunk} bedeutsame Fakten über:
1. Alle Teilnehmer (User UND KI-Assistenten)
2. Diskutierte Themen
3. Wichtige Aussagen/Meinungen
4. Lernfortschritte

AUSGABE-FORMAT (JSON):
{{
  "facts": [
    {{
      "text": "Bob ist Python-Experte mit 10 Jahren Erfahrung",
      "importance": 0.8,
      "participants": ["Bob", "Silas"],
      "theme": "Python Expertise"
    }}
  ]
}}

REGELN:
1. Min 0, Max {settings.ltm_max_facts_per_chunk} Facts
2. Keine Smalltalk-Floskeln ("Hi", "Danke", "Tschüss")
3. Kurz & präzise (1-2 Sätze, max 200 Zeichen)
4. Jeder Fact ist atomisch/eigenständig
5. Theme: 2-4 Wörter, beschreibt spezifischen Kontext
6. Importance: 0.0-1.0
   - 0.8-1.0: Kritisch (Beruf, Expertise, wichtige Präferenzen)
   - 0.5-0.7: Relevant (Meinungen, Lernfortschritt)
   - 0.3-0.4: Nebensächlich (casual preferences)
   - <{settings.ltm_min_importance_threshold}: Ignorieren
7. Participants: Namen aller beteiligten
8. Facts über KI-Assistenten sind genauso wichtig wie Facts über User

CONVERSATION MESSAGES:
{formatted_messages}

Antworte NUR mit JSON, keine zusätzlichen Erklärungen."""

    def _parse_facts_response(self, llm_response: str) -> list[dict]:
        """
        Parse LLM response into fact list with validation.

        :param llm_response: LLM response (may contain markdown code blocks)
        :return: List of validated facts
        """
        try:
            # Strip markdown code blocks if present
            llm_response = llm_response.strip()
            if llm_response.startswith("```json"):
                llm_response = llm_response[7:]
            if llm_response.startswith("```"):
                llm_response = llm_response[3:]
            if llm_response.endswith("```"):
                llm_response = llm_response[:-3]

            llm_response = llm_response.strip()

            # Parse JSON
            parsed = json.loads(llm_response)
            raw_facts = parsed.get("facts", [])

            # Validate and normalize each fact
            validated_facts = []
            for fact in raw_facts:
                if not isinstance(fact, dict):
                    continue

                # Must have text
                if not fact.get("text"):
                    continue

                # Normalize structure with defaults
                normalized_fact = {
                    "text": fact["text"],
                    "importance": fact.get("importance", 0.5),  # Default medium
                    "participants": fact.get("participants", []),
                    "theme": fact.get("theme", "General"),
                }

                # Ensure participants is a list
                if not isinstance(normalized_fact["participants"], list):
                    normalized_fact["participants"] = []

                # Ensure importance is a number in valid range
                try:
                    normalized_fact["importance"] = float(normalized_fact["importance"])
                    normalized_fact["importance"] = max(0.0, min(1.0, normalized_fact["importance"]))
                except (ValueError, TypeError):
                    normalized_fact["importance"] = 0.5

                validated_facts.append(normalized_fact)

            return validated_facts

        except json.JSONDecodeError as e:
            logger.warning("ltm_json_parse_failed", error=str(e), response=llm_response[:200])
            return []
        except Exception as e:
            logger.warning("ltm_parse_unexpected_error", error=str(e))
            return []

    def _extract_facts_heuristic(self, chunk_messages: list[dict]) -> list[dict]:
        """
        Fallback heuristic for fact extraction when LLM fails.

        Phase 1: Basic factual sentence extraction.

        :param chunk_messages: Messages with sender_name and content
        :return: List of facts (basic structure)
        """
        facts = []
        combined_text = " ".join([msg.get("content", "") for msg in chunk_messages])

        if not combined_text.strip():
            return []

        # Extract factual sentences using non-capturing groups to get full sentences
        # Pattern: sentence containing factual verbs
        sentences = re.split(r'[.!?]+', combined_text)

        factual_verbs = r'\b(?:ist|hat|kann|arbeitet|lernt|mag|bevorzugt|studiert|kennt|weiß)\b'

        factual_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            # Check if sentence contains factual verbs and is substantial
            if re.search(factual_verbs, sentence, re.IGNORECASE) and len(sentence) > 20:
                factual_sentences.append(sentence)

        # Take top 3 factual sentences
        for sentence in factual_sentences[:3]:
            # Skip very long sentences (likely concatenated)
            if len(sentence) > 300:
                continue

            facts.append({
                "text": sentence,
                "importance": 0.5,  # Default medium importance
                "participants": [],
                "theme": "General Fact",
            })

        return facts

    def _get_fact_hash(self, fact_text: str) -> str:
        """
        Calculate normalized hash for fact text (for idempotence).

        Normalization: strip, lowercase, collapse multiple spaces to single space.

        :param fact_text: Fact text
        :return: SHA256 hash (16 characters)
        """
        normalized = fact_text.strip().lower()
        normalized = re.sub(r'\s+', ' ', normalized)
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]
