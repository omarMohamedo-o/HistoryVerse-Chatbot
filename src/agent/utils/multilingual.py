"""Multilingual text processing, cross-lingual normalization, and keyword expansion utilities."""

from __future__ import annotations

import re
import unicodedata

from agent.utils.arabic import (
    expand_arabic_keywords,
    is_arabic,
    normalize_alef,
    normalize_arabic,
    normalize_teh_marbuta,
    remove_tashkeel,
    remove_tatweel,
)

__all__ = [
    "detect_language",
    "detect_script",
    "expand_multilingual_keywords",
    "fuzzy_token_match",
    "is_arabic",
    "normalize_alef",
    "normalize_arabic",
    "normalize_script_characters",
    "normalize_teh_marbuta",
    "normalize_text",
    "normalize_unicode",
    "remove_tashkeel",
    "remove_tatweel",
    "strip_combining_marks",
]

# Unicode script patterns for fast language/script identification
_ARABIC_PATTERN = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
_CYRILLIC_PATTERN = re.compile(r"[\u0400-\u04FF\u0500-\u052F]")
_GREEK_PATTERN = re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]")
_CJK_PATTERN = re.compile(r"[\u4E00-\u9FFF\u3040-\u30FF\u3400-\u4DBF]")
_HEBREW_PATTERN = re.compile(r"[\u0590-\u05FF]")
_LATIN_PATTERN = re.compile(r"[a-zA-Z\u00C0-\u024F]")

_SCRIPT_LIGATURE_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("ß", "ss"),
    ("ẞ", "SS"),
    ("œ", "oe"),
    ("Œ", "OE"),
    ("æ", "ae"),
    ("Æ", "AE"),
    ("ё", "е"),
    ("Ё", "Е"),
    ("ς", "σ"),
    ("ی", "ي"),
    ("ک", "ك"),
    ("گ", "ك"),
    ("چ", "ج"),
    ("پ", "ب"),
)


def normalize_unicode(text: str) -> str:
    """Normalize text into NFKC standard form."""
    return unicodedata.normalize("NFKC", text)


def strip_combining_marks(text: str) -> str:
    """Strip all combining diacritical marks (accents, tildes, umlauts, tonos, tashkeel)."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", stripped)


def normalize_script_characters(text: str) -> str:
    """Fold script-specific ligatures and letter variants to plain base equivalents."""
    result = text
    for target, replacement in _SCRIPT_LIGATURE_REPLACEMENTS:
        if target in result:
            result = result.replace(target, replacement)
    return result


def detect_script(text: str) -> str:
    """Detect dominant writing script of input text."""
    if not text:
        return "unknown"
    if _ARABIC_PATTERN.search(text):
        return "arabic"
    if _CYRILLIC_PATTERN.search(text):
        return "cyrillic"
    if _GREEK_PATTERN.search(text):
        return "greek"
    if _CJK_PATTERN.search(text):
        return "cjk"
    if _HEBREW_PATTERN.search(text):
        return "hebrew"
    if _LATIN_PATTERN.search(text):
        return "latin"
    return "unknown"


# Stopwords and characteristic lexical markers for top international visitor languages
_STOPWORDS_FR = {"le", "la", "les", "des", "du", "qui", "que", "dans", "pour", "avec", "est", "sont", "cette", "ce", "en", "sur", "par", "quand", "comment", "citadelle", "château", "musée", "histoire", "construit"}
_STOPWORDS_DE = {"der", "die", "das", "und", "in", "ist", "wer", "was", "wie", "wo", "hat", "mit", "für", "von", "den", "dem", "des", "ein", "eine", "einer", "erbaut", "zitadelle", "geschichte", "museum"}
_STOPWORDS_ES = {"el", "la", "los", "las", "de", "en", "que", "por", "para", "con", "quien", "quién", "qué", "cómo", "donde", "dónde", "cuando", "cuándo", "construyó", "del", "historia", "ciudadela", "museo"}
_STOPWORDS_IT = {"il", "lo", "la", "i", "gli", "le", "di", "in", "su", "per", "con", "chi", "cosa", "dove", "quando", "ha", "costruito", "della", "degli", "storia", "cittadella", "museo"}
_STOPWORDS_EN = {"the", "is", "are", "who", "what", "where", "when", "which", "built", "about", "in", "of", "and", "to", "citadel", "history", "museum", "how"}


def detect_language(text: str) -> str:
    """Automatically detects the ISO 639-1 language code of input text.

    Supports the primary international tourist languages for Egyptian and world heritage:
    Arabic (ar), English (en), French (fr), German (de), Spanish (es), Italian (it),
    Russian (ru), Chinese (zh).
    """
    if not text or not text.strip():
        return "en"

    # Fast script detection
    script = detect_script(text)
    if script == "arabic":
        return "ar"
    if script == "cyrillic":
        return "ru"
    if script == "cjk":
        return "zh"
    if script == "greek":
        return "el"
    if script == "hebrew":
        return "he"

    # Latin script classification
    tokens = set(re.findall(r"\b\w+\b", text.lower()))

    # Direct orthographic cues
    if any(c in text for c in "¿¡ñ"):
        return "es"
    if any(c in text for c in "äöüß"):
        return "de"
    if any(c in text for c in "œæç"):
        return "fr"

    scores = {
        "fr": len(tokens.intersection(_STOPWORDS_FR)),
        "de": len(tokens.intersection(_STOPWORDS_DE)),
        "es": len(tokens.intersection(_STOPWORDS_ES)),
        "it": len(tokens.intersection(_STOPWORDS_IT)),
        "en": len(tokens.intersection(_STOPWORDS_EN)),
    }

    best_lang, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score > 0:
        return best_lang

    try:
        import langid
        code, _ = langid.classify(text)
        return code
    except Exception:
        return "en"


def normalize_text(text: str) -> str:
    """Apply universal, language-agnostic text normalization.

    Removes diacritics, unifies letterforms across Arabic, Latin, Greek, Cyrillic,
    folds case, and collapses whitespace.
    """
    if not text:
        return ""

    normalized = normalize_unicode(text)

    # Arabic script normalization
    if is_arabic(normalized):
        normalized = normalize_arabic(
            normalized,
            strip_diacritics=True,
            strip_kashida=True,
            norm_alef=True,
            norm_teh=True,
            norm_maqsura=True,
        )

    # Fold ligatures and script-specific characters
    normalized = normalize_script_characters(normalized)

    # Strip remaining combining marks (accents, tildes, umlauts, tonos)
    normalized = strip_combining_marks(normalized)

    # Casefold and collapse spaces
    normalized = normalized.lower()
    return re.sub(r"\s+", " ", normalized).strip()


def expand_multilingual_keywords(keyword: str) -> list[str]:
    """Generate comprehensive cross-lingual spelling and morphological variants of a keyword.

    Used for concept unlearning, RAG vector purges, and cross-lingual concept elimination.
    """
    cleaned = keyword.strip()
    if not cleaned:
        return []

    variants: set[str] = {cleaned, cleaned.lower()}

    # Fully normalized form
    norm_val = normalize_text(cleaned)
    if norm_val:
        variants.add(norm_val)

    # Accent-stripped form without full ligature replacement
    unaccented = strip_combining_marks(cleaned)
    variants.add(unaccented)
    variants.add(unaccented.lower())

    # Arabic variants
    if is_arabic(cleaned):
        for ar_var in expand_arabic_keywords(cleaned):
            variants.add(ar_var)
            variants.add(ar_var.lower())

    # German variants (ß <-> ss, umlauts)
    if "ß" in cleaned or "ss" in cleaned.lower():
        variants.add(cleaned.replace("ß", "ss"))
        variants.add(cleaned.lower().replace("ß", "ss"))
        variants.add(cleaned.lower().replace("ss", "ß"))

    # French / Latin ligatures
    if any(char in cleaned for char in ("œ", "Œ", "æ", "Æ", "oe", "ae")):
        folded = normalize_script_characters(cleaned)
        variants.add(folded)
        variants.add(folded.lower())

    # Spanish tildes (ñ <-> n)
    if "ñ" in cleaned or "Ñ" in cleaned:
        variants.add(cleaned.replace("ñ", "n").replace("Ñ", "N"))
        variants.add(cleaned.lower().replace("ñ", "n"))

    # Cyrillic (ё <-> е)
    if "ё" in cleaned or "Ё" in cleaned:
        variants.add(cleaned.replace("ё", "е").replace("Ё", "Е"))
        variants.add(cleaned.lower().replace("ё", "е"))

    # Greek (tonos and final sigma)
    if _GREEK_PATTERN.search(cleaned):
        greek_clean = strip_combining_marks(cleaned).lower()
        variants.add(greek_clean)
        variants.add(greek_clean.replace("ς", "σ"))

    return sorted(v for v in variants if v)


def fuzzy_token_match(content: str, keywords: list[str]) -> bool:
    """Evaluate whether content matches any target keyword under multilingual normalization."""
    if not content or not keywords:
        return False

    lower_content = content.lower()
    for kw in keywords:
        cleaned_kw = kw.strip().lower()
        if cleaned_kw and cleaned_kw in lower_content:
            return True

    # Check under full universal multilingual normalization
    norm_content = normalize_text(content)
    for kw in keywords:
        norm_kw = normalize_text(kw)
        if norm_kw and norm_kw in norm_content:
            return True

    return False
