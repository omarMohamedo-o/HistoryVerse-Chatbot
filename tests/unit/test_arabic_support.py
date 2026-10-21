"""Unit tests for Arabic language processing, multilingual embeddings, and normalization."""

from unittest.mock import MagicMock, patch

from historyverse.config.settings import Settings
from historyverse.core.enums import EmbeddingProvider
from historyverse.services.memory.manager import ConversationMemory
from historyverse.services.rag.ingest import (
    EmbeddingStrategyFactory,
    HuggingFaceEmbeddingStrategy,
    OpenAIEmbeddingStrategy,
)
from historyverse.services.voice.tts import GTTSTextToSpeech
from historyverse.utils.arabic import (
    expand_arabic_keywords,
    is_arabic,
    normalize_alef,
    normalize_arabic,
    normalize_teh_marbuta,
    remove_tashkeel,
    remove_tatweel,
)
from tests.fixtures.multilingual_samples import ARABIC_EXCHANGE


def test_is_arabic_detection():
    assert is_arabic("مكتبة الإسكندرية") is True
    assert is_arabic("Ancient Alexandria") is False
    assert is_arabic("Alexandria (الإسكندرية)") is True
    assert is_arabic("") is False


def test_remove_tashkeel_and_tatweel():
    diacritized = "مَكْتَبَةُ الإِسْكَنْدَرِيَّةِ"
    clean = remove_tashkeel(diacritized)
    assert clean == "مكتبة الإسكندرية"

    tatweel_text = "الإســــكندرية"
    clean_tatweel = remove_tatweel(tatweel_text)
    assert clean_tatweel == "الإسكندرية"


def test_normalize_alef_and_teh_marbuta():
    text = "إبراهيم وأحمد وآمنة"
    norm_alef = normalize_alef(text)
    assert norm_alef == "ابراهيم واحمد وامنة"

    teh_text = "مكتبة الإسكندرية القديمة"
    norm_teh = normalize_teh_marbuta(teh_text)
    assert norm_teh == "مكتبه الإسكندريه القديمه"


def test_normalize_arabic_full():
    sample = "مَكْتَبَـةُ الإِسْـكَنْدَرِيَّـةِ"
    normalized = normalize_arabic(
        sample,
        strip_diacritics=True,
        strip_kashida=True,
        norm_alef=True,
        norm_teh=True,
    )
    assert normalized == "مكتبه الاسكندريه"


def test_expand_arabic_keywords():
    keyword = "الإِسْكَنْدَرِيَّة"
    variants = expand_arabic_keywords(keyword)
    assert len(variants) >= 3
    assert "الإسكندرية" in variants
    assert "الاسكندرية" in variants
    assert "الاسكندريه" in variants

    # Non-arabic returns as-is
    assert expand_arabic_keywords("Alexandria") == ["Alexandria"]
    assert expand_arabic_keywords("") == []


def test_huggingface_multilingual_model_selection():
    strategy = HuggingFaceEmbeddingStrategy()

    # When user specifies multilingual/Arabic model
    settings_bge = Settings(
        embedding_provider=EmbeddingProvider.HUGGINGFACE.value,
        embedding_model="BAAI/bge-m3",
    )
    with patch.dict("sys.modules", {"langchain_community.embeddings": MagicMock()}):
        res = strategy.create(settings_bge)
        assert res is not None

    # When model is not specified, defaults to SOTA multilingual
    settings_default = Settings(embedding_model="")
    with patch.dict("sys.modules", {"langchain_community.embeddings": MagicMock()}):
        res_default = strategy.create(settings_default)
        assert res_default is not None


def test_openai_embedding_strategy_creation():
    settings = Settings(
        embedding_provider=EmbeddingProvider.OPENAI.value,
        embedding_model="text-embedding-3-small",
        openai_api_key="sk-test-key",
    )
    strategy = EmbeddingStrategyFactory.get_strategy(EmbeddingProvider.OPENAI.value)
    assert isinstance(strategy, OpenAIEmbeddingStrategy)

    mock_langchain_openai = MagicMock()
    with patch.dict("sys.modules", {"langchain_openai": mock_langchain_openai}):
        inst = strategy.create(settings)
        assert inst is not None


def test_tts_detects_arabic_script():
    tts = GTTSTextToSpeech()
    # Direct unicode check immediately identifies Arabic without network or langid dependency
    detected = tts._detect_language("مرحباً بكم في تاريخ مصر القديمة")
    assert detected == "ar"

    # English fallback
    detected_en = tts._detect_language("Welcome to ancient Alexandria")
    assert detected_en == "en"


def test_conversation_memory_purges_arabic_variants():
    memory = ConversationMemory(Settings())
    sample = ARABIC_EXCHANGE

    memory.add_exchange(sample.session_id, sample.query, sample.response)
    assert len(memory.get_history(sample.session_id)) == sample.expected_purged_turns

    # Purge using un-diacritized keyword with plain Alef
    purged = memory.purge_concept([sample.purge_keyword])
    assert purged == sample.expected_purged_turns
    assert len(memory.get_history(sample.session_id)) == 0


def test_embedding_model_registry_language_profiling():
    from historyverse.services.rag.embeddings import EmbeddingModelRegistry

    # Multilingual BGE-M3
    bge_profile = EmbeddingModelRegistry.get_profile("BAAI/bge-m3")
    assert bge_profile is not None
    assert bge_profile.is_multilingual is True
    assert bge_profile.supports_arabic is True
    assert bge_profile.dimensions == 1024

    # Monolingual English all-MiniLM-L6-v2
    minilm_profile = EmbeddingModelRegistry.get_profile("sentence-transformers/all-MiniLM-L6-v2")
    assert minilm_profile is not None
    assert minilm_profile.is_multilingual is False
    assert minilm_profile.supports_arabic is False

    # Language compatibility audit
    valid, msg = EmbeddingModelRegistry.validate_language_compatibility(
        "sentence-transformers/all-MiniLM-L6-v2", target_language="ar"
    )
    assert valid is False
    assert "does NOT support Arabic" in msg
    assert "BAAI/bge-m3" in msg

    # Recommendation
    rec_hf = EmbeddingModelRegistry.recommend_model_for_language("ar", "huggingface")
    assert rec_hf == "BAAI/bge-m3"

    rec_openai = EmbeddingModelRegistry.recommend_model_for_language("ar", "openai")
    assert rec_openai == "text-embedding-3-small"
