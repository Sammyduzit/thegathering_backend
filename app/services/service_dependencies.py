from fastapi import Depends

from app.core.arq_pool import get_arq_pool
from app.core.config import settings
from app.implementations.deepl_translator import DeepLTranslator
from app.interfaces.ai_provider import IAIProvider
from app.interfaces.embedding_service import IEmbeddingService
from app.interfaces.keyword_extractor import IKeywordExtractor
from app.interfaces.memory_retriever import IMemoryRetriever
from app.interfaces.memory_summarizer import IMemorySummarizer
from app.interfaces.translator import TranslatorInterface
from app.providers.openai_provider import OpenAIProvider
from app.repositories.ai_cooldown_repository import IAICooldownRepository
from app.repositories.ai_entity_repository import IAIEntityRepository
from app.repositories.ai_memory_repository import IAIMemoryRepository
from app.repositories.conversation_repository import IConversationRepository
from app.repositories.message_repository import IMessageRepository
from app.repositories.message_translation_repository import (
    IMessageTranslationRepository,
)
from app.repositories.repository_dependencies import (
    get_ai_cooldown_repository,
    get_ai_entity_repository,
    get_ai_memory_repository,
    get_conversation_repository,
    get_message_repository,
    get_message_translation_repository,
    get_room_repository,
    get_user_repository,
)
from app.repositories.room_repository import IRoomRepository
from app.repositories.user_repository import IUserRepository
from app.services.ai.ai_entity_service import AIEntityService
from app.services.domain.background_service import BackgroundService
from app.services.domain.conversation_service import ConversationService
from app.services.domain.room_service import RoomService
from app.services.domain.translation_service import TranslationService
from app.services.embedding.embedding_factory import create_embedding_service
from app.services.memory.keyword_retriever import KeywordMemoryRetriever
from app.services.memory.long_term_memory_service import LongTermMemoryService
from app.services.memory.personality_memory_service import PersonalityMemoryService
from app.services.memory.short_term_memory_service import ShortTermMemoryService
from app.services.memory.vector_memory_retriever import VectorMemoryRetriever
from app.services.text_processing.heuristic_summarizer import HeuristicMemorySummarizer
from app.services.text_processing.keyword_extractor_factory import create_keyword_extractor
from app.services.text_processing.text_chunking_service import TextChunkingService


def get_deepl_translator() -> TranslatorInterface:
    """
    Create DeepL translator instance with API key from settings.
    :return: DeepL translator instance implementing TranslatorInterface
    """
    return DeepLTranslator(api_key=settings.deepl_api_key)


def get_translation_service(
    translator: TranslatorInterface = Depends(get_deepl_translator),
    message_repo: IMessageRepository = Depends(get_message_repository),
    translation_repo: IMessageTranslationRepository = Depends(get_message_translation_repository),
) -> TranslationService:
    """
    Create TranslationService instance with translator and repository dependencies.
    :param translator: Translator interface instance (DeepL implementation)
    :param message_repo: Message repository instance
    :param translation_repo: MessageTranslation repository instance
    :return: TranslationService instance
    """
    return TranslationService(
        translator=translator,
        message_repo=message_repo,
        translation_repo=translation_repo,
    )


def get_conversation_service(
    conversation_repo: IConversationRepository = Depends(get_conversation_repository),
    message_repo: IMessageRepository = Depends(get_message_repository),
    user_repo: IUserRepository = Depends(get_user_repository),
    room_repo: IRoomRepository = Depends(get_room_repository),
    translation_service: TranslationService = Depends(get_translation_service),
    ai_entity_repo: IAIEntityRepository = Depends(get_ai_entity_repository),
    arq_pool=Depends(lambda: get_arq_pool()),
) -> ConversationService:
    """
    Create ConversationService instance with repository dependencies.

    :param conversation_repo: Conversation repository instance
    :param message_repo: Message repository instance
    :param user_repo: User repository instance
    :param room_repo: Room repository instance
    :param translation_service: Translation service instance
    :param ai_entity_repo: AI entity repository instance
    :param arq_pool: ARQ Redis pool for background tasks
    :return: ConversationService instance
    """
    return ConversationService(
        conversation_repo=conversation_repo,
        message_repo=message_repo,
        user_repo=user_repo,
        room_repo=room_repo,
        translation_service=translation_service,
        ai_entity_repo=ai_entity_repo,
        arq_pool=arq_pool,
    )


def get_room_service(
    room_repo: IRoomRepository = Depends(get_room_repository),
    user_repo: IUserRepository = Depends(get_user_repository),
    message_repo: IMessageRepository = Depends(get_message_repository),
    conversation_repo: IConversationRepository = Depends(get_conversation_repository),
    message_translation_repo: IMessageTranslationRepository = Depends(get_message_translation_repository),
    translation_service: TranslationService = Depends(get_translation_service),
    ai_entity_repo: IAIEntityRepository = Depends(get_ai_entity_repository),
) -> RoomService:
    """
    Create RoomService instance with repository dependencies.
    :param room_repo: Room repository instance
    :param user_repo: User repository instance
    :param message_repo: Message repository instance
    :param conversation_repo: Conversation repository instance
    :param message_translation_repo: MessageTranslation repository instance
    :param translation_service: Translation service instance
    :param ai_entity_repo: AI entity repository instance
    :return: RoomService instance
    """
    return RoomService(
        room_repo=room_repo,
        user_repo=user_repo,
        message_repo=message_repo,
        conversation_repo=conversation_repo,
        message_translation_repo=message_translation_repo,
        translation_service=translation_service,
        ai_entity_repo=ai_entity_repo,
    )


def get_background_service(
    translation_service: TranslationService = Depends(get_translation_service),
    message_translation_repo: IMessageTranslationRepository = Depends(get_message_translation_repository),
) -> BackgroundService:
    """
    Create BackgroundService instance with service dependencies.
    :param translation_service: Translation service instance
    :param message_translation_repo: MessageTranslation repository instance
    :return: BackgroundService instance
    """
    return BackgroundService(
        translation_service=translation_service,
        message_translation_repo=message_translation_repo,
    )


def get_ai_provider() -> IAIProvider:
    """
    Create AI provider instance (currently OpenAI).

    Uses DEFAULT_PROVIDER_MODEL from constants (gpt-4o-mini).
    Future: Could switch to other providers based on config/feature flags.

    :return: AI provider instance
    """
    return OpenAIProvider(
        api_key=settings.openai_api_key,
    )


def get_ai_entity_service(
    ai_entity_repo: IAIEntityRepository = Depends(get_ai_entity_repository),
    conversation_repo: IConversationRepository = Depends(get_conversation_repository),
    cooldown_repo: IAICooldownRepository = Depends(get_ai_cooldown_repository),
    room_repo: IRoomRepository = Depends(get_room_repository),
    message_repo: IMessageRepository = Depends(get_message_repository),
    conversation_service: ConversationService = Depends(get_conversation_service),
    ai_provider: IAIProvider = Depends(get_ai_provider),
) -> AIEntityService:
    """
    Create AIEntityService instance with repository and service dependencies.

    :param ai_entity_repo: AI entity repository instance
    :param conversation_repo: Conversation repository instance
    :param cooldown_repo: AI cooldown repository instance
    :param room_repo: Room repository instance
    :param message_repo: Message repository instance
    :param conversation_service: Conversation service for memory management
    :param ai_provider: AI provider for LLM interactions
    :return: AIEntityService instance
    """
    return AIEntityService(
        ai_entity_repo=ai_entity_repo,
        conversation_repo=conversation_repo,
        cooldown_repo=cooldown_repo,
        room_repo=room_repo,
        message_repo=message_repo,
        conversation_service=conversation_service,
        ai_provider=ai_provider,
    )


def get_keyword_extractor() -> IKeywordExtractor:
    """
    Create keyword extractor implementation based on feature flags.

    Feature flags (future):
    - USE_LLM_KEYWORDS: Switch to LLM-based keyword extraction

    :return: Keyword extractor instance (default: YAKE)
    """
    return create_keyword_extractor()


def get_memory_summarizer() -> IMemorySummarizer:
    """
    Create memory summarizer implementation based on feature flags.

    Feature flags (future):
    - USE_LLM_SUMMARIZATION: Switch to LLM-based summarization

    :return: Memory summarizer instance (default: Heuristic)
    """
    # Future: Check settings.USE_LLM_SUMMARIZATION for LLM implementation
    return HeuristicMemorySummarizer()


def get_embedding_service() -> IEmbeddingService:
    """
    Create embedding service for vector search.

    Uses factory to select provider based on settings.embedding_provider (google/openai).

    :return: Embedding service instance (Google or OpenAI)
    """
    return create_embedding_service()


def get_text_chunking_service() -> TextChunkingService:
    """
    Create text chunking service.

    :return: Text chunking service instance
    """
    return TextChunkingService(
        chunk_size=500,
        chunk_overlap=50,
    )


def get_memory_retriever(
    memory_repo: IAIMemoryRepository = Depends(get_ai_memory_repository),
    embedding_service: IEmbeddingService = Depends(get_embedding_service),
    keyword_extractor: IKeywordExtractor = Depends(get_keyword_extractor),
) -> IMemoryRetriever:
    """
    Create memory retriever implementation based on feature flags.

    :param memory_repo: AI memory repository instance
    :param embedding_service: Embedding service instance
    :param keyword_extractor: Keyword extractor instance
    :return: Memory retriever instance (Vector if enabled, else Keyword)
    """
    if settings.enable_vector_search:
        return VectorMemoryRetriever(
            memory_repo=memory_repo,
            embedding_service=embedding_service,
            keyword_extractor=keyword_extractor,
        )
    else:
        return KeywordMemoryRetriever(memory_repo=memory_repo)


def get_short_term_memory_service(
    memory_repo: IAIMemoryRepository = Depends(get_ai_memory_repository),
    keyword_extractor: IKeywordExtractor = Depends(get_keyword_extractor),
) -> ShortTermMemoryService:
    """
    Create ShortTermMemoryService instance.

    :param memory_repo: AI memory repository instance
    :param keyword_extractor: Keyword extractor instance
    :return: ShortTermMemoryService instance
    """
    return ShortTermMemoryService(
        memory_repo=memory_repo,
        keyword_extractor=keyword_extractor,
    )


def get_ltm_ai_provider():
    """
    Factory for LTM Fact Extraction Provider (Gemini or OpenAI).

    IMPORTANT: Separate factory from get_ai_provider()!
    - get_ai_provider(): OpenAI for response generation
    - get_ltm_ai_provider(): Configurable for LTM extraction

    :return: AI provider instance for LTM extraction
    :raises RuntimeError: If required API key is not configured
    """
    if settings.ltm_provider == "google":
        if not settings.google_api_key:
            raise RuntimeError(
                "LTM_PROVIDER is set to 'google' but GOOGLE_API_KEY is not configured. "
                "Please set GOOGLE_API_KEY in your environment or change LTM_PROVIDER to 'openai'."
            )
        from app.providers.google_provider import GoogleProvider
        return GoogleProvider(
            api_key=settings.google_api_key,
            model_name=settings.ltm_extraction_model,
        )
    else:
        if not settings.openai_api_key:
            raise RuntimeError(
                "LTM_PROVIDER is set to 'openai' but OPENAI_API_KEY is not configured. "
                "Please set OPENAI_API_KEY in your environment."
            )
        return OpenAIProvider(
            api_key=settings.openai_api_key,
            model_name=settings.ltm_extraction_model,
        )


def get_long_term_memory_service(
    memory_repo: IAIMemoryRepository = Depends(get_ai_memory_repository),
    embedding_service: IEmbeddingService = Depends(get_embedding_service),
    keyword_extractor: IKeywordExtractor = Depends(get_keyword_extractor),
    ltm_ai_provider=Depends(get_ltm_ai_provider),
) -> LongTermMemoryService:
    """
    Create LongTermMemoryService instance.

    :param memory_repo: AI memory repository instance
    :param embedding_service: Embedding service instance
    :param keyword_extractor: Keyword extractor instance
    :param ltm_ai_provider: LTM AI provider for fact extraction
    :return: LongTermMemoryService instance
    """
    return LongTermMemoryService(
        memory_repo=memory_repo,
        embedding_service=embedding_service,
        keyword_extractor=keyword_extractor,
        ai_provider=ltm_ai_provider,
    )


def get_personality_memory_service(
    memory_repo: IAIMemoryRepository = Depends(get_ai_memory_repository),
    embedding_service: IEmbeddingService = Depends(get_embedding_service),
    chunking_service: TextChunkingService = Depends(get_text_chunking_service),
    keyword_extractor: IKeywordExtractor = Depends(get_keyword_extractor),
) -> PersonalityMemoryService:
    """
    Create PersonalityMemoryService instance.

    :param memory_repo: AI memory repository instance
    :param embedding_service: Embedding service instance
    :param chunking_service: Text chunking service instance
    :param keyword_extractor: Keyword extractor instance
    :return: PersonalityMemoryService instance
    """
    return PersonalityMemoryService(
        memory_repo=memory_repo,
        embedding_service=embedding_service,
        chunking_service=chunking_service,
        keyword_extractor=keyword_extractor,
    )
