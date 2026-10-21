"""Structured multilingual test datasets and fixtures for HistoryVerse tests.

Provides zero-hardcode benchmark dialogues across Arabic, French, German, Spanish,
Russian, Greek, and English.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MultilingualExchangeSample:
    """Encapsulates a test dialogue turn with unlearning and purge verification data."""

    language: str
    language_code: str
    session_id: str
    query: str
    response: str
    purge_keyword: str
    purge_keyword_variants: list[str]
    expected_purged_turns: int = 2


ARABIC_EXCHANGE = MultilingualExchangeSample(
    language="Arabic",
    language_code="ar",
    session_id="session-arabic-eval",
    query="ما هي حقيقة النفق السري في مَكْتَبَةُ الإِسْكَنْدَرِيَّةِ؟",
    response="تقول الشائعات أن هناك نفقاً سرياً تحت مَكْتَبَةُ الإِسْكَنْدَرِيَّةِ القديمة.",
    purge_keyword="مكتبة الاسكندرية",
    purge_keyword_variants=["مكتبة الاسكندرية", "مكتبه الاسكندريه", "الإسكندرية"],
)

FRENCH_EXCHANGE = MultilingualExchangeSample(
    language="French",
    language_code="fr",
    session_id="session-french-eval",
    query="Quelle est la vérité sur le Château de Versailles et son passage secret ?",
    response="Le Château de Versailles possède de nombreux passages secrets bien documentés.",
    purge_keyword="chateau de versailles",
    purge_keyword_variants=["château de versailles", "chateau de versailles", "Versailles"],
)

GERMAN_EXCHANGE = MultilingualExchangeSample(
    language="German",
    language_code="de",
    session_id="session-german-eval",
    query="Gibt es ein Geheimnis über die Große Mauer und das alte Schloß?",
    response="Über die Große Mauer und alte Schlösser gibt es historische Berichte.",
    purge_keyword="grosse mauer",
    purge_keyword_variants=["große mauer", "grosse mauer", "schloss"],
)

SPANISH_EXCHANGE = MultilingualExchangeSample(
    language="Spanish",
    language_code="es",
    session_id="session-spanish-eval",
    query="¿Cuál es el misterio del Niño de las cavernas en Toledo?",
    response="El Niño de las cavernas es una leyenda toledana muy antigua.",
    purge_keyword="nino de las cavernas",
    purge_keyword_variants=["niño de las cavernas", "nino de las cavernas", "Toledo"],
)

RUSSIAN_EXCHANGE = MultilingualExchangeSample(
    language="Russian",
    language_code="ru",
    session_id="session-russian-eval",
    query="Какова правда о секретах царского шатра и всё ли это миф?",
    response="Секреты шатра давно исследованы историками, и всё ли это миф остается загадкой.",
    purge_keyword="все ли это миф",
    purge_keyword_variants=["всё ли это миф", "все ли это миф", "шатра"],
)

GREEK_EXCHANGE = MultilingualExchangeSample(
    language="Greek",
    language_code="el",
    session_id="session-greek-eval",
    query="Ποια είναι η αλήθεια για την Ακρόπολη και τα μυστικά της;",
    response="Η Ακρόπολη των Αθηνών έχει πολλά ιστορικά μνημεία.",
    purge_keyword="ακροπολη",
    purge_keyword_variants=["ακρόπολη", "ακροπολη", "Ακρόπολη"],
)

ENGLISH_EXCHANGE = MultilingualExchangeSample(
    language="English",
    language_code="en",
    session_id="session-english-eval",
    query="What is the historical truth regarding the Library of Alexandria tunnels?",
    response="Rumors suggest subterranean vaults beneath the ancient Library of Alexandria.",
    purge_keyword="library of alexandria",
    purge_keyword_variants=["library of alexandria", "Library of Alexandria"],
)

MULTILINGUAL_SAMPLES: dict[str, MultilingualExchangeSample] = {
    "ar": ARABIC_EXCHANGE,
    "fr": FRENCH_EXCHANGE,
    "de": GERMAN_EXCHANGE,
    "es": SPANISH_EXCHANGE,
    "ru": RUSSIAN_EXCHANGE,
    "el": GREEK_EXCHANGE,
    "en": ENGLISH_EXCHANGE,
}
