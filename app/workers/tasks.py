"""ARQ task functions for AI response generation and translation."""

from uuid import uuid4

import structlog
from arq import Retry
import math

from app.core.arq_db_manager import ARQDatabaseManager, db_session_context
from app.core.config import settings
from app.implementations.deepl_translator import DeepLTranslator
from app.interfaces.ai_provider import AIProviderError
from app.interfaces.translator import TranslationError
from app.models.ai_entity import AIEntity, AIEntityStatus
from app.models.message_translation import MessageTranslation
from app.providers.openai_provider import OpenAIProvider
from app.repositories.ai_cooldown_repository import AICooldownRepository
from app.repositories.ai_entity_repository import AIEntityRepository
from app.repositories.ai_memory_repository import AIMemoryRepository
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.message_repository import MessageRepository
from app.repositories.message_translation_repository import MessageTranslationRepository
from app.services.ai.ai_context_service import AIContextService
from app.services.ai.ai_response_service import AIResponseService
from app.services.domain.translation_service import TranslationService
from app.services.embedding.embedding_factory import create_embedding_service
from app.services.memory.long_term_memory_service import LongTermMemoryService
from app.services.memory.short_term_memory_service import ShortTermMemoryService
from app.services.service_dependencies import get_embedding_service, get_memory_retriever
from app.services.text_processing.keyword_extractor_factory import create_keyword_extractor
from app.services.text_processing.text_chunking_service import TextChunkingService

logger = structlog.get_logger(__name__)


async def _catch_up_short_term_chunks(
    memory_repo: AIMemoryRepository,
    message_repo: MessageRepository,
    ai_entity_id: int,
    conversation_id: int,
    user_ids: list[int],
) -> list:
    """
    Ensure all required STM chunks exist (full + partial) before LTM extraction.

    Returns the refreshed list of STM chunks after filling gaps.
    """
    from app.core.constants import SHORT_TERM_CHUNK_SIZE

    all_messages, _ = await message_repo.get_conversation_messages(
        conversation_id=conversation_id,
        page=1,
        page_size=1000,  # Matches inline STM chunking limit
    )
    conversation_messages = [m for m in all_messages if m.message_type != "system"]
    total_messages = len(conversation_messages)

    existing_stm_chunks = await memory_repo.get_short_term_chunks(
        conversation_id=conversation_id,
        entity_id=ai_entity_id,
    )
    existing_indices = {
        chunk.memory_metadata.get("chunk_index")
        for chunk in existing_stm_chunks
        if chunk.memory_metadata and chunk.memory_metadata.get("chunk_index") is not None
    }

    needed_chunks = math.ceil(total_messages / SHORT_TERM_CHUNK_SIZE) if total_messages else 0

    if needed_chunks > len(existing_indices):
        keyword_extractor = create_keyword_extractor()
        short_term_service = ShortTermMemoryService(
            memory_repo=memory_repo,
            keyword_extractor=keyword_extractor,
        )

        for chunk_idx in range(needed_chunks):
            if chunk_idx in existing_indices:
                continue

            start_idx = chunk_idx * SHORT_TERM_CHUNK_SIZE
            end_idx = min(start_idx + SHORT_TERM_CHUNK_SIZE - 1, total_messages - 1)
            chunk_messages = conversation_messages[start_idx : end_idx + 1]

            if not chunk_messages:
                continue

            await short_term_service.create_short_term_chunk(
                entity_id=ai_entity_id,
                user_ids=user_ids,
                conversation_id=conversation_id,
                chunk_messages=chunk_messages,
                chunk_index=chunk_idx,
                start_idx=start_idx,
                end_idx=end_idx,
            )

            logger.info(
                "stm_chunk_catch_up_created",
                ai_entity_id=ai_entity_id,
                conversation_id=conversation_id,
                chunk_index=chunk_idx,
                message_count=len(chunk_messages),
                message_range=f"{start_idx}-{end_idx}",
            )

    return await memory_repo.get_short_term_chunks(
        conversation_id=conversation_id,
        entity_id=ai_entity_id,
    )


async def _lookup_ai_entity(
    ai_entity_repo: AIEntityRepository,
    ai_entity_id: int | None,
    room_id: int | None,
    conversation_id: int | None,
) -> AIEntity | None:
    """
    Lookup AI entity from ID or context (room/conversation).

    :param ai_entity_repo: AI entity repository
    :param ai_entity_id: Direct AI entity ID (if provided)
    :param room_id: Room ID for lookup (if no direct ID)
    :param conversation_id: Conversation ID for lookup (if no direct ID)
    :return: AIEntity if found, None otherwise
    """
    if ai_entity_id:
        return await ai_entity_repo.get_by_id(ai_entity_id)
    elif room_id:
        return await ai_entity_repo.get_ai_in_room(room_id)
    elif conversation_id:
        return await ai_entity_repo.get_ai_in_conversation(conversation_id)
    return None


async def _get_conversation_user_ids(conversation_repo, conversation_id: int) -> list[int]:
    """
    Fetch all human participant user IDs from conversation.

    :param conversation_repo: Conversation repository
    :param conversation_id: Conversation ID
    :return: List of user IDs (excludes AI participants)
    """
    participants = await conversation_repo.get_participants(conversation_id)
    return [p.user_id for p in participants if p.user_id is not None]


async def _generate_response_for_target(
    response_service,
    room_id: int | None,
    conversation_id: int | None,
    ai_entity: AIEntity,
    message_id: int,
    sender_user_id: int | None,
):
    """
    Generate AI response for room or conversation.

    :param response_service: AI response service
    :param room_id: Room ID (for room responses)
    :param conversation_id: Conversation ID (for conversation responses)
    :param ai_entity: AI entity generating response
    :param message_id: Message ID to reply to
    :param sender_user_id: User ID who sent the message
    :return: Generated Message object
    """
    if room_id:
        return await response_service.generate_room_response(
            room_id=room_id,
            ai_entity=ai_entity,
            user_id=sender_user_id,
            include_memories=True,
            in_reply_to_message_id=message_id,
        )
    else:
        return await response_service.generate_conversation_response(
            conversation_id=conversation_id,
            ai_entity=ai_entity,
            user_id=sender_user_id,
            include_memories=True,
            in_reply_to_message_id=message_id,
        )


async def _handle_post_generation_checks(ai_entity: AIEntity, room_id: int | None, session, message_repo) -> bool:
    """
    Validate AI entity after response generation (race condition protection).

    :param ai_entity: AI entity to validate
    :param room_id: Room ID if in room context
    :param session: Database session
    :param message_repo: Message repository
    :return: True if validation passed, False if AI became invalid during generation
    """
    await session.refresh(ai_entity)
    return _validate_ai_can_respond(ai_entity, room_id)


async def _create_inline_memory(
    conversation_id: int | None,
    ai_entity: AIEntity,
    message_repo,
    memory_repo,
    conversation_repo,
) -> None:
    """
    Create short-term memory chunks incrementally after AI response.

    INCREMENTAL CHUNKING:
    - Get all conversation messages
    - Calculate expected chunks (total_messages // 24)
    - Get existing chunks from DB
    - Create only missing chunks (idempotent)

    :param conversation_id: Conversation ID
    :param ai_entity: AI entity
    :param message_repo: Message repository
    :param memory_repo: Memory repository
    :param conversation_repo: Conversation repository

    Note:
        Logs warning on failure but doesn't raise (non-critical operation)
    """
    if not conversation_id:
        return

    try:
        from app.core.constants import SHORT_TERM_CHUNK_SIZE

        keyword_extractor = create_keyword_extractor()
        short_term_service = ShortTermMemoryService(
            memory_repo=memory_repo,
            keyword_extractor=keyword_extractor,
        )

        # Fetch all human participant user IDs
        user_ids = await _get_conversation_user_ids(conversation_repo, conversation_id)

        if not user_ids:
            logger.warning(
                "short_term_memory_skipped_no_users",
                ai_entity_id=ai_entity.id,
                conversation_id=conversation_id,
            )
            return

        # Get ALL messages for chunking (not just recent 20)
        all_messages, total_count = await message_repo.get_conversation_messages(
            conversation_id=conversation_id,
            page=1,
            page_size=1000,  # High limit to get all messages
        )

        # Filter conversation messages (exclude system)
        conversation_messages = [m for m in all_messages if m.message_type != "system"]
        total_messages = len(conversation_messages)

        # Calculate expected complete chunks
        expected_complete_chunks = total_messages // SHORT_TERM_CHUNK_SIZE

        # Get existing chunks
        existing_chunks = await memory_repo.get_short_term_chunks(
            conversation_id=conversation_id,
            entity_id=ai_entity.id,
        )
        existing_chunk_count = len(existing_chunks)

        # Create newest chunk if needed (normalbetrieb: always just 1 new chunk)
        if expected_complete_chunks > existing_chunk_count:
            # Only create the newest chunk (the one that just reached 24 messages)
            chunk_idx = existing_chunk_count
            start_idx = chunk_idx * SHORT_TERM_CHUNK_SIZE
            end_idx = start_idx + SHORT_TERM_CHUNK_SIZE - 1

            # Slice messages for this chunk
            chunk_messages = conversation_messages[start_idx : end_idx + 1]

            # Create single chunk
            await short_term_service.create_short_term_chunk(
                entity_id=ai_entity.id,
                user_ids=user_ids,
                conversation_id=conversation_id,
                chunk_messages=chunk_messages,
                chunk_index=chunk_idx,
                start_idx=start_idx,
                end_idx=end_idx,
            )

            logger.debug(
                "short_term_chunk_created",
                ai_entity_id=ai_entity.id,
                conversation_id=conversation_id,
                total_messages=total_messages,
                chunk_index=chunk_idx,
                message_range=f"{start_idx}-{end_idx}",
            )
        else:
            logger.debug(
                "short_term_chunks_up_to_date",
                ai_entity_id=ai_entity.id,
                conversation_id=conversation_id,
                total_messages=total_messages,
                existing_chunks=existing_chunk_count,
            )

    except Exception as e:
        # Non-critical: log warning, don't fail the task
        logger.warning(
            "short_term_memory_creation_failed",
            error=str(e),
            conversation_id=conversation_id,
        )


def _validate_ai_can_respond(ai_entity: AIEntity, room_id: int | None) -> bool:
    """
    Validate that AI entity can respond in current context.

    :param ai_entity: AI entity to validate
    :param room_id: Room ID if responding in room (None for conversations)
    :return: True if AI can respond, False otherwise
    """
    # Check AI is ONLINE
    if ai_entity.status != AIEntityStatus.ONLINE:
        logger.warning(
            "ai_validation_failed",
            ai_entity_id=ai_entity.id,
            reason="AI not ONLINE",
            status=ai_entity.status,
        )
        return False

    # Check AI is still in room (if room_id provided)
    if room_id and ai_entity.current_room_id != room_id:
        logger.warning(
            "ai_validation_failed",
            ai_entity_id=ai_entity.id,
            reason="AI no longer in room",
            expected_room=room_id,
            actual_room=ai_entity.current_room_id,
        )
        return False

    return True


async def check_and_generate_ai_response(
    ctx: dict,
    message_id: int,
    room_id: int | None = None,
    conversation_id: int | None = None,
    ai_entity_id: int | None = None,
) -> dict:
    """
    Unified ARQ task: Check if AI should respond and generate response if needed.

    This task handles race conditions by:
    1. PRE-CHECK: Validate AI status and room presence before generation
    2. Strategy check: Use should_ai_respond() to check response strategy
    3. Generation: Generate response (5-30s)
    4. POST-CHECK: Re-validate AI still active before saving

    :param ctx: ARQ context with db_manager
    :param message_id: Message ID that triggered this check
    :param room_id: Room ID (for room messages)
    :param conversation_id: Conversation ID (for conversation messages)
    :param ai_entity_id: AI entity ID (optional, can be looked up from room/conversation)
    :return: Dict with ai_message_id on success or skipped reason
    :raises Retry: On transient errors (max 3 attempts)
    """
    job_id = str(uuid4())
    db_session_context.set(job_id)

    db_manager: ARQDatabaseManager = ctx["db_manager"]

    try:
        async for session in db_manager.get_session():
            ai_entity_repo = AIEntityRepository(session)
            message_repo = MessageRepository(session)
            memory_repo = AIMemoryRepository(session)
            conversation_repo = ConversationRepository(session)
            cooldown_repo = AICooldownRepository(session)

            # Get AI entity (from ID or lookup)
            ai_entity = await _lookup_ai_entity(ai_entity_repo, ai_entity_id, room_id, conversation_id)

            if not ai_entity:
                logger.info(
                    "ai_response_skipped",
                    reason="No AI entity found",
                    room_id=room_id,
                    conversation_id=conversation_id,
                )
                return {"skipped": "No AI entity found"}

            # PRE-CHECK: Validate AI can respond (prevents race conditions)
            if not _validate_ai_can_respond(ai_entity, room_id):
                return {"skipped": "AI validation failed (pre-check)"}

            # Get the message that triggered this check
            message = await message_repo.get_by_id(message_id)
            if not message:
                logger.error("message_not_found", message_id=message_id)
                return {"error": "Message not found"}

            # Initialize AI provider and services
            ai_provider = OpenAIProvider(
                api_key=settings.openai_api_key,
                model_name=ai_entity.model_name or "gpt-4o-mini",
            )

            # Initialize memory retriever for context service using factory
            embedding_service = get_embedding_service()
            keyword_extractor = create_keyword_extractor()
            memory_retriever = get_memory_retriever(
                memory_repo=memory_repo,
                embedding_service=embedding_service,
                keyword_extractor=keyword_extractor,
            )
            context_service = AIContextService(
                message_repo=message_repo,
                memory_repo=memory_repo,
                memory_retriever=memory_retriever,
            )

            response_service = AIResponseService(
                ai_provider=ai_provider,
                context_service=context_service,
                message_repo=message_repo,
                cooldown_repo=cooldown_repo,
            )

            # Check if AI should respond based on strategy
            should_respond = await response_service.should_ai_respond(
                ai_entity=ai_entity,
                latest_message=message,
                conversation_id=conversation_id,
                room_id=room_id,
            )

            if not should_respond:
                logger.info(
                    "ai_response_skipped",
                    reason="Strategy check failed",
                    ai_entity_id=ai_entity.id,
                    room_id=room_id,
                    conversation_id=conversation_id,
                )
                return {"skipped": "Strategy check failed"}

            # Generate response (5-30s duration)
            ai_message = await _generate_response_for_target(
                response_service=response_service,
                room_id=room_id,
                conversation_id=conversation_id,
                ai_entity=ai_entity,
                message_id=message_id,
                sender_user_id=message.sender_user_id,
            )

            # POST-CHECK: Re-validate AI still active (prevents race conditions)
            if not await _handle_post_generation_checks(ai_entity, room_id, session, message_repo):
                # AI was set offline or left room during generation - delete the message
                await message_repo.delete(ai_message.id)
                logger.warning(
                    "ai_response_cancelled",
                    reason="AI validation failed after generation (post-check)",
                    ai_entity_id=ai_entity.id,
                )
                return {"skipped": "AI validation failed (post-check)"}

            logger.info(
                "ai_response_generated",
                ai_entity_id=ai_entity.id,
                ai_message_id=ai_message.id,
                room_id=room_id,
                conversation_id=conversation_id,
            )

            # Create short-term memory after AI response (inline, fast, non-critical)
            await _create_inline_memory(
                conversation_id=conversation_id,
                ai_entity=ai_entity,
                message_repo=message_repo,
                memory_repo=memory_repo,
                conversation_repo=conversation_repo,
            )

            # Update cooldown timestamp after successful response
            await cooldown_repo.upsert_cooldown(
                ai_entity_id=ai_entity.id,
                room_id=room_id,
                conversation_id=conversation_id,
            )
            logger.info(
                "ai_cooldown_updated",
                ai_entity_id=ai_entity.id,
                room_id=room_id,
                conversation_id=conversation_id,
            )

            return {
                "ai_message_id": ai_message.id,
                "ai_entity_id": ai_entity.id,
                "room_id": room_id,
                "conversation_id": conversation_id,
            }

    except AIProviderError as e:
        logger.error(
            "ai_provider_error",
            error=str(e),
            room_id=room_id,
            conversation_id=conversation_id,
        )
        raise Retry(defer=ctx["job_try"] * 5)

    except Exception as e:
        logger.exception(
            "unexpected_error_generating_ai_response",
            error=str(e),
            room_id=room_id,
            conversation_id=conversation_id,
        )
        raise Retry(defer=ctx["job_try"] * 5)


async def create_long_term_memory_task(
    ctx: dict,
    ai_entity_id: int,
    conversation_id: int,
) -> dict:
    """
    ARQ task: Create long-term memory from short-term chunks via LLM fact extraction.

    This task:
    - Fetches ALL human participant user IDs from conversation
    - Fetches STM chunks for this AI entity and conversation
    - Extracts facts from each chunk via LLM
    - Creates individual LTM entries (1 Fact = 1 LTM Entry = 1 Embedding)
    - Deletes STM chunks only after successful LTM creation

    :param ctx: ARQ context with db_manager
    :param ai_entity_id: AI entity ID
    :param conversation_id: Conversation ID to archive
    :return: Dict with memory count, STM chunks processed, and deleted count
    """
    job_id = str(uuid4())
    db_session_context.set(job_id)

    db_manager: ARQDatabaseManager = ctx["db_manager"]

    try:
        async for session in db_manager.get_session():
            memory_repo = AIMemoryRepository(session)
            conversation_repo = ConversationRepository(session)
            message_repo = MessageRepository(session)

            # Fetch all human participant user IDs
            user_ids = await _get_conversation_user_ids(conversation_repo, conversation_id)

            if not user_ids:
                logger.warning(
                    "long_term_memory_skipped_no_users",
                    ai_entity_id=ai_entity_id,
                    conversation_id=conversation_id,
                )
                return {"memory_count": 0, "stm_chunks_processed": 0, "stm_chunks_deleted": 0}

            # Catch-up STM chunking guard: fill missing chunks (including partial)
            stm_chunks = await _catch_up_short_term_chunks(
                memory_repo=memory_repo,
                message_repo=message_repo,
                ai_entity_id=ai_entity_id,
                conversation_id=conversation_id,
                user_ids=user_ids,
            )

            if not stm_chunks:
                logger.info(
                    "long_term_memory_skipped_no_stm_chunks",
                    ai_entity_id=ai_entity_id,
                    conversation_id=conversation_id,
                )
                return {"memory_count": 0, "stm_chunks_processed": 0, "stm_chunks_deleted": 0}

            # Initialize LTM AI Provider (Google Gemini or OpenAI)
            if settings.ltm_provider == "google":
                if not settings.google_api_key:
                    raise RuntimeError(
                        "LTM_PROVIDER is set to 'google' but GOOGLE_API_KEY is not configured"
                    )
                from app.providers.google_provider import GoogleProvider
                ltm_ai_provider = GoogleProvider(
                    api_key=settings.google_api_key,
                    model_name=settings.ltm_extraction_model,
                )
            else:
                if not settings.openai_api_key:
                    raise RuntimeError(
                        "LTM_PROVIDER is set to 'openai' but OPENAI_API_KEY is not configured"
                    )
                from app.providers.openai_provider import OpenAIProvider
                ltm_ai_provider = OpenAIProvider(
                    api_key=settings.openai_api_key,
                    model_name=settings.ltm_extraction_model,
                )

            # Initialize services
            embedding_service = create_embedding_service()
            keyword_extractor = create_keyword_extractor()
            long_term_service = LongTermMemoryService(
                memory_repo=memory_repo,
                embedding_service=embedding_service,
                keyword_extractor=keyword_extractor,
                ai_provider=ltm_ai_provider,
            )

            # Fact-based extraction (transactional)
            deleted_count = 0  # Default for error case
            try:
                ltm_memories = await long_term_service.create_long_term_from_chunks(
                    entity_id=ai_entity_id,
                    user_ids=user_ids,
                    conversation_id=conversation_id,
                    stm_chunks=stm_chunks,
                )

                # Only delete STM chunks after successful LTM creation
                deleted_count = await memory_repo.delete_short_term_chunks(
                    conversation_id=conversation_id,
                    entity_id=ai_entity_id,
                )

                logger.info(
                    "long_term_memory_created_from_stm_chunks",
                    ai_entity_id=ai_entity_id,
                    user_ids=user_ids,
                    conversation_id=conversation_id,
                    memory_count=len(ltm_memories),
                    stm_chunks_processed=len(stm_chunks),
                    stm_chunks_deleted=deleted_count,
                )

                return {
                    "memory_count": len(ltm_memories),
                    "memory_ids": [m.id for m in ltm_memories],
                    "stm_chunks_processed": len(stm_chunks),
                    "stm_chunks_deleted": deleted_count,
                }

            except Exception as e:
                # STM kept for retry
                logger.error(
                    "ltm_conversion_failed_stm_kept",
                    error=str(e),
                    ai_entity_id=ai_entity_id,
                    conversation_id=conversation_id,
                    stm_deleted=deleted_count,
                )
                raise  # ARQ will retry

    except Exception as e:
        logger.error(
            "long_term_memory_creation_failed",
            error=str(e),
            ai_entity_id=ai_entity_id,
            conversation_id=conversation_id,
        )
        raise Retry(defer=ctx["job_try"] * 10)  # Retry with backoff


async def cleanup_old_short_term_memories_task(ctx: dict) -> dict:
    """
    ARQ task (cron): Delete short-term memories older than TTL.

    Runs daily at 3 AM to prevent short-term memory accumulation.
    Short-term memories are temporary and should not persist beyond TTL.

    :param ctx: ARQ context with db_manager
    :return: Dict with deleted_count
    """
    job_id = str(uuid4())
    db_session_context.set(job_id)

    db_manager: ARQDatabaseManager = ctx["db_manager"]

    try:
        async for session in db_manager.get_session():
            memory_repo = AIMemoryRepository(session)

            # Delete old short-term memories
            deleted_count = await memory_repo.delete_old_short_term_memories(ttl_days=settings.short_term_ttl_days)

            logger.info(
                "short_term_memories_cleaned_up",
                deleted_count=deleted_count,
                ttl_days=settings.short_term_ttl_days,
            )

            return {
                "deleted_count": deleted_count,
                "ttl_days": settings.short_term_ttl_days,
            }

    except Exception as e:
        logger.error(
            "short_term_cleanup_failed",
            error=str(e),
        )
        # Don't retry - will run again tomorrow
        return {"error": str(e), "deleted_count": 0}


async def translate_message(
    ctx: dict,
    message_id: int,
    message_content: str,
    target_languages: list[str],
) -> dict:
    """
    ARQ task: Translate message content to multiple target languages.

    Creates fresh DB session per job to avoid session lifecycle issues.
    Checks for existing translations before creating new ones.

    :param ctx: ARQ context with db_manager
    :param message_id: Message ID to translate
    :param message_content: Message content to translate
    :param target_languages: List of target language codes (e.g., ['DE', 'FR'])
    :return: Dict with translation count and language codes
    :raises Retry: On transient errors (max 3 attempts)
    """
    job_id = str(uuid4())
    db_session_context.set(job_id)

    db_manager: ARQDatabaseManager = ctx["db_manager"]

    normalized_target_languages = [
        TranslationService.normalize_target_language(lang) for lang in target_languages if lang
    ]
    normalized_target_languages = [lang for lang in normalized_target_languages if lang]
    if not normalized_target_languages:
        logger.info("no_target_languages_for_translation", message_id=message_id)
        return {"translation_count": 0, "languages": []}

    logger.info(
        "translation_task_started",
        message_id=message_id,
        target_language_count=len(normalized_target_languages),
    )

    translations = {}

    try:
        async for session in db_manager.get_session():
            translation_repo = MessageTranslationRepository(session)
            message_repo = MessageRepository(session)
            translator = DeepLTranslator(api_key=settings.deepl_api_key)
            translation_service = TranslationService(
                translator=translator,
                message_repo=message_repo,
                translation_repo=translation_repo,
            )

            for target_lang in normalized_target_languages:
                try:
                    existing_translation = await translation_repo.get_by_message_and_language(
                        message_id, target_lang
                    )

                    if existing_translation:
                        translations[target_lang] = existing_translation.content
                        logger.info(
                            "existing_translation_used",
                            message_id=message_id,
                            target_language=target_lang,
                        )
                        continue

                    translation_result = await translation_service.translate_message_content(
                        content=message_content, target_languages=[target_lang], source_language=None
                    )

                    if target_lang in translation_result:
                        content = translation_result[target_lang]

                        new_translation = MessageTranslation(
                            message_id=message_id, content=content, target_language=target_lang
                        )
                        await translation_repo.create(new_translation)

                        translations[target_lang] = content
                        logger.info(
                            "message_translated",
                            message_id=message_id,
                            target_language=target_lang,
                        )

                except TranslationError as e:
                    logger.error(
                        "translation_api_failed",
                        message_id=message_id,
                        target_language=target_lang,
                        error=str(e),
                    )
                    continue

            logger.info(
                "translation_task_completed",
                message_id=message_id,
                translation_count=len(translations),
            )

            return {
                "translation_count": len(translations),
                "languages": list(translations.keys()),
                "message_id": message_id,
            }

    except Exception as e:
        logger.error(
            "translation_task_failed",
            error=str(e),
            message_id=message_id,
        )
        raise Retry(defer=ctx["job_try"] * 5)
