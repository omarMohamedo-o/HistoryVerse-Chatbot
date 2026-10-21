"""HistoryVerse utility helpers for multilingual text processing and normalization."""

from agentverse.utils.arabic import (
    expand_arabic_keywords,
    is_arabic,
    normalize_alef,
    normalize_arabic,
    normalize_teh_marbuta,
    remove_tashkeel,
    remove_tatweel,
)
from agentverse.utils.multilingual import (
    detect_script,
    expand_multilingual_keywords,
    fuzzy_token_match,
    normalize_script_characters,
    normalize_text,
    normalize_unicode,
    strip_combining_marks,
)

__all__ = [
    "detect_script",
    "expand_arabic_keywords",
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
