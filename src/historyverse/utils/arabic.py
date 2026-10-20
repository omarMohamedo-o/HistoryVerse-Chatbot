"""Arabic text processing, normalization, and script utilities for multilingual NLP and RAG."""

from __future__ import annotations

import re

__all__ = [
    "expand_arabic_keywords",
    "is_arabic",
    "normalize_alef",
    "normalize_arabic",
    "normalize_teh_marbuta",
    "remove_tashkeel",
    "remove_tatweel",
]

# Unicode ranges for Arabic scripts and diacritics
_ARABIC_CHAR_PATTERN = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
_TASHKEEL_PATTERN = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670]")
_TATWEEL_PATTERN = re.compile(r"\u0640")
_ALEF_PATTERN = re.compile(r"[إأآا]")
_TEH_MARBUTA_PATTERN = re.compile(r"ة")
_ALEF_MAQSURA_PATTERN = re.compile(r"ى")


def is_arabic(text: str) -> bool:
    """Return True if the text contains Arabic characters."""
    return bool(_ARABIC_CHAR_PATTERN.search(text))


def remove_tashkeel(text: str) -> str:
    """Strip all Arabic diacritical marks (Fat-ha, Damma, Kasra, Sukun, Tanween, Shadda)."""
    return _TASHKEEL_PATTERN.sub("", text)


def remove_tatweel(text: str) -> str:
    """Strip Arabic elongation kashida/tatweel characters."""
    return _TATWEEL_PATTERN.sub("", text)


def normalize_alef(text: str) -> str:
    """Normalize all Alef forms (إ, أ, آ) to a plain Alef (ا)."""
    return _ALEF_PATTERN.sub("ا", text)


def normalize_teh_marbuta(text: str) -> str:
    """Normalize Teh Marbuta (ة) to Heh (ه)."""
    return _TEH_MARBUTA_PATTERN.sub("ه", text)


def normalize_alef_maqsura(text: str) -> str:
    """Normalize Alef Maqsura (ى) to Yeh (ي)."""
    return _ALEF_MAQSURA_PATTERN.sub("ي", text)


def normalize_arabic(
    text: str,
    strip_diacritics: bool = True,
    strip_kashida: bool = True,
    norm_alef: bool = True,
    norm_teh: bool = False,
    norm_maqsura: bool = False,
) -> str:
    """Apply comprehensive morphological normalization to Arabic text."""
    result = text
    if strip_kashida:
        result = remove_tatweel(result)
    if strip_diacritics:
        result = remove_tashkeel(result)
    if norm_alef:
        result = normalize_alef(result)
    if norm_teh:
        result = normalize_teh_marbuta(result)
    if norm_maqsura:
        result = normalize_alef_maqsura(result)
    return result


def expand_arabic_keywords(keyword: str) -> list[str]:
    """Generate spelling and morphological variants of an Arabic keyword.

    Used for comprehensive vector store purging and cross-lingual concept unlearning.
    """
    cleaned = keyword.strip()
    if not cleaned or not is_arabic(cleaned):
        return [cleaned] if cleaned else []

    variants: set[str] = {cleaned}

    # 1. Without diacritics
    no_tashkeel = remove_tashkeel(cleaned)
    variants.add(no_tashkeel)

    # 2. Normalized Alef
    norm_alef_val = normalize_alef(no_tashkeel)
    variants.add(norm_alef_val)

    # 3. Normalized Teh Marbuta and Alef Maqsura
    norm_full = normalize_alef_maqsura(normalize_teh_marbuta(norm_alef_val))
    variants.add(norm_full)

    # 4. Without Tatweel
    no_tatweel = remove_tatweel(cleaned)
    variants.add(no_tatweel)
    variants.add(remove_tashkeel(no_tatweel))

    return [v for v in variants if v]
