"""Arabic NLP Normalizer for Historical and Antiquity Texts.

Handles historical document and monument extraction artifacts, Lam-Alef ligatures, Arabic-Indic numerals,
mirrored RTL brackets, and Tashkeel normalization.
"""

from __future__ import annotations

import re
import unicodedata

# 1. Ligature shielding: In PDF extraction, Lam-Alef forms are often rendered as a single glyph.
# We shield them into private-use Unicode before processing, then restore them.
LIGATURE_SHIELDS = {
    "\ue000": "لا",
    "\ue001": "لأ",
    "\ue002": "لإ",
    "\ue003": "لآ",
}
LIGATURE_UNSHIELDS = {v: k for k, v in LIGATURE_SHIELDS.items()}

# 2. Arabic-Indic to Western Digit translation
ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
WESTERN_TO_ARABIC_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")

# 3. Mirror brackets: RTL reversal causes '(' to render as ')' and '[' as ']'.
MIRROR_BRACKETS = str.maketrans("()[]{}", ")(][}{")

# 4. Tashkeel (diacritics) regex
TASHKEEL_RE = re.compile(r"[\u064B-\u0652\u0670\u0640]")

# 5. Alef normalization
ALEF_RE = re.compile(r"[\u0622\u0623\u0625\u0671]")


def normalize_digits(text: str) -> str:
    """Convert Arabic-Indic numerals (٠-٩) to standard ASCII digits (0-9)."""
    if not text:
        return ""
    return text.translate(ARABIC_INDIC_DIGITS)


def strip_tashkeel(text: str) -> str:
    """Remove Arabic diacritics (harakat/tashkeel) and tatweel (kashida)."""
    if not text:
        return ""
    return TASHKEEL_RE.sub("", text)


def normalize_alef(text: str) -> str:
    """Normalize various Alef forms (أ, إ, آ, ٱ) into a bare Alef (ا)."""
    if not text:
        return ""
    return ALEF_RE.sub("ا", text)


def fix_mirrored_brackets(text: str) -> str:
    """Invert mirrored brackets resulting from bidirectional text reversals."""
    if not text:
        return ""
    return text.translate(MIRROR_BRACKETS)


def shield_ligatures(text: str) -> str:
    """Shield Lam-Alef ligatures before bidirectional transformations."""
    if not text:
        return ""
    for shield, ligature in LIGATURE_SHIELDS.items():
        text = text.replace(ligature, shield)
    return text


def unshield_ligatures(text: str) -> str:
    """Restore shielded Lam-Alef ligatures after transformations."""
    if not text:
        return ""
    for shield, ligature in LIGATURE_SHIELDS.items():
        text = text.replace(shield, ligature)
    return text


def clean_arabic_text(
    text: str,
    normalize_numerals: bool = True,
    remove_tashkeel: bool = True,
    norm_alef: bool = False,
    fix_brackets: bool = False,
) -> str:
    """Full-pipeline Arabic text cleaner for historical documents, museum records, and user queries.

    Args:
        text: Raw input text from historical archive extraction or user voice/text input.
        normalize_numerals: Whether to convert ٠-٩ to 0-9.
        remove_tashkeel: Whether to remove harakat/diacritics.
        norm_alef: Whether to normalize all Alef forms to plain Alef.
        fix_brackets: Whether to invert mirrored brackets.

    Returns:
        Cleaned, normalized Arabic text suitable for exact artifact matching and embedding.
    """
    if not text:
        return ""

    # Unshield if already encoded
    text = unshield_ligatures(text)

    # Unicode normalization (NFKC)
    text = unicodedata.normalize("NFKC", text)

    if remove_tashkeel:
        text = strip_tashkeel(text)

    if normalize_numerals:
        text = normalize_digits(text)

    if norm_alef:
        text = normalize_alef(text)

    if fix_brackets:
        text = fix_mirrored_brackets(text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Backwards compatibility alias
clean_arabic_legal_text = clean_arabic_text
