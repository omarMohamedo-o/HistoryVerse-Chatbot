"""Unit tests for universal multilingual text processing, normalization, and memory unlearning."""

import pytest

from historyverse.config.settings import Settings
from historyverse.services.memory.manager import ConversationMemory
from historyverse.services.rag.embeddings import EmbeddingModelRegistry
from historyverse.services.unlearning.validator import UnlearningValidator
from historyverse.utils.multilingual import (
    detect_script,
    expand_multilingual_keywords,
    fuzzy_token_match,
    normalize_script_characters,
    normalize_text,
    normalize_unicode,
    strip_combining_marks,
)
from tests.conftest import MockLLMProvider
from tests.fixtures.multilingual_samples import (
    ARABIC_EXCHANGE,
    ENGLISH_EXCHANGE,
    FRENCH_EXCHANGE,
    GERMAN_EXCHANGE,
    GREEK_EXCHANGE,
    MULTILINGUAL_SAMPLES,
    RUSSIAN_EXCHANGE,
    SPANISH_EXCHANGE,
)


def test_detect_script():
    assert detect_script(ARABIC_EXCHANGE.query) == "arabic"
    assert detect_script(RUSSIAN_EXCHANGE.query) == "cyrillic"
    assert detect_script(GREEK_EXCHANGE.query) == "greek"
    assert detect_script(FRENCH_EXCHANGE.query) == "latin"
    assert detect_script(GERMAN_EXCHANGE.query) == "latin"
    assert detect_script(SPANISH_EXCHANGE.query) == "latin"
    assert detect_script(ENGLISH_EXCHANGE.query) == "latin"
    assert detect_script("東京の歴史") == "cjk"
    assert detect_script("ירושלים העתיקה") == "hebrew"
    assert detect_script("") == "unknown"


def test_strip_combining_marks_and_ligatures():
    # NFKC unicode
    assert normalize_unicode("Café") == "Café"

    # French
    french_text = "Château de Versailles et vérité"
    assert strip_combining_marks(french_text) == "Chateau de Versailles et verite"

    # Spanish
    spanish_text = "El Niño de las cavernas"
    assert strip_combining_marks(spanish_text) == "El Nino de las cavernas"

    # German
    german_text = "Große Schlösser"
    folded = normalize_script_characters(german_text)
    assert strip_combining_marks(folded) == "Grosse Schlosser"

    # Greek tonos
    greek_text = "Ακρόπολη και μυστικά"
    assert strip_combining_marks(greek_text) == "Ακροπολη και μυστικα"

    # Russian Cyrillic yo
    russian_text = "всё тайное"
    assert normalize_script_characters(russian_text) == "все тайное"


def test_normalize_text_universal():
    # French
    assert normalize_text("  Château   de   Versailles!  ") == "chateau de versailles!"
    # German
    assert normalize_text("Große Mauer") == "grosse mauer"
    # Spanish
    assert normalize_text("El Niño") == "el nino"
    # Greek
    assert normalize_text("Η Ακρόπολη") == "η ακροπολη"
    # Russian
    assert normalize_text("Всё миф") == "все миф"
    # Arabic
    assert normalize_text("مَكْتَبَـةُ الإِسْكَنْدَرِيَّـةِ") == "مكتبه الاسكندريه"
    # Empty
    assert normalize_text("") == ""


def test_expand_multilingual_keywords():
    # French
    fr_variants = expand_multilingual_keywords("Château")
    assert "château" in fr_variants
    assert "chateau" in fr_variants

    # German
    de_variants = expand_multilingual_keywords("Große")
    assert "große" in de_variants
    assert "grosse" in de_variants

    # Spanish
    es_variants = expand_multilingual_keywords("Niño")
    assert "niño" in es_variants
    assert "nino" in es_variants

    # Russian
    ru_variants = expand_multilingual_keywords("Всё")
    assert "всё" in ru_variants
    assert "все" in ru_variants

    # Greek
    el_variants = expand_multilingual_keywords("Ακρόπολη")
    assert "ακρόπολη" in el_variants
    assert "ακροπολη" in el_variants

    # Empty
    assert expand_multilingual_keywords("") == []


def test_fuzzy_token_match():
    # Matches even when accents/diacritics differ
    assert fuzzy_token_match("Le Château de Versailles est royal.", ["chateau"]) is True
    assert fuzzy_token_match("Die Große Mauer in Berlin.", ["grosse mauer"]) is True
    assert fuzzy_token_match("El Niño duerme.", ["nino"]) is True
    assert fuzzy_token_match("Η Ακρόπολη των Αθηνών.", ["ακροπολη"]) is True
    assert fuzzy_token_match("Все тайны раскрыты.", ["всё тайны"]) is True
    assert fuzzy_token_match("مَكْتَبَةُ الإِسْكَنْدَرِيَّةِ", ["مكتبة الاسكندرية"]) is True

    # Negative mismatch
    assert fuzzy_token_match("Ancient Alexandria", ["Pyramids"]) is False
    assert fuzzy_token_match("", ["test"]) is False
    assert fuzzy_token_match("test", []) is False


@pytest.mark.parametrize("lang_code", list(MULTILINGUAL_SAMPLES.keys()))
def test_conversation_memory_purges_multilingual_variants(lang_code: str):
    """Verify that ConversationMemory purges forbidden dialogue turns across all supported languages."""
    memory = ConversationMemory(Settings())
    sample = MULTILINGUAL_SAMPLES[lang_code]

    # Add dialogue turn using structured fixture (zero hardcoded literals)
    memory.add_exchange(sample.session_id, sample.query, sample.response)
    assert len(memory.get_history(sample.session_id)) == sample.expected_purged_turns

    # Purge using variant/unaccented keyword
    purged = memory.purge_concept([sample.purge_keyword])
    assert purged == sample.expected_purged_turns
    assert len(memory.get_history(sample.session_id)) == 0


def test_unlearning_validator_multilingual_refusal_and_leakage():
    validator = UnlearningValidator(llm_provider=MockLLMProvider())

    # English refusal
    leak_en, conf_en = validator.check_response_leakage(
        "I do not have verified records regarding the library of alexandria.",
        ["Library of Alexandria"],
    )
    assert leak_en is False
    assert conf_en <= 0.1

    # Arabic refusal
    leak_ar, conf_ar = validator.check_response_leakage(
        "لا أملك معلومات مؤكدة حول نفق مكتبة الاسكندرية.",
        ["نفق مكتبة الإسكندرية"],
    )
    assert leak_ar is False
    assert conf_ar <= 0.1

    # French refusal
    leak_fr, conf_fr = validator.check_response_leakage(
        "Aucun enregistrement concernant le château de versailles.",
        ["Château de Versailles"],
    )
    assert leak_fr is False
    assert conf_fr <= 0.1

    # German refusal
    leak_de, conf_de = validator.check_response_leakage(
        "Keine Aufzeichnungen über die Große Mauer vorhanden.",
        ["Große Mauer"],
    )
    assert leak_de is False
    assert conf_de <= 0.1

    # Spanish refusal
    leak_es, conf_es = validator.check_response_leakage(
        "Sin registros sobre el niño de las cavernas.",
        ["Niño de las cavernas"],
    )
    assert leak_es is False
    assert conf_es <= 0.1

    # Affirmative leakage (no refusal indicator) -> MUST report leak!
    leak_affirmative, conf_aff = validator.check_response_leakage(
        "Le Château de Versailles possède un souterrain secret magnifique.",
        ["Château de Versailles"],
    )
    assert leak_affirmative is True
    assert conf_aff > 0.0


def test_embedding_model_registry_cross_lingual_validation():
    # Multilingual models support all target languages
    for lang in ["ar", "fr", "de", "es", "ru", "el"]:
        valid, _ = EmbeddingModelRegistry.validate_language_compatibility(
            "BAAI/bge-m3", target_language=lang
        )
        assert valid is True

    # Monolingual English model fails for non-English languages
    for lang in ["ar", "fr", "de", "es", "ru"]:
        valid_mono, msg = EmbeddingModelRegistry.validate_language_compatibility(
            "sentence-transformers/all-MiniLM-L6-v2", target_language=lang
        )
        assert valid_mono is False
        assert "Recommended upgrade" in msg

    # Monolingual English model succeeds for English
    valid_en, _ = EmbeddingModelRegistry.validate_language_compatibility(
        "sentence-transformers/all-MiniLM-L6-v2", target_language="en"
    )
    assert valid_en is True
