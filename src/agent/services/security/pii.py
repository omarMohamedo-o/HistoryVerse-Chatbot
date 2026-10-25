"""Egyptian Personally Identifiable Information (PII) Redactor.

Sanitizes sensitive Egyptian user identifiers (National IDs, Egyptian mobile numbers,
emails, and credit cards) prior to embedding, logging, or passing to third-party LLMs.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

PII_PATTERNS: List[Tuple[str, re.Pattern[str], str]] = [
    # 1. Egyptian National ID: 14 digits starting with 2 (born 1900-1999) or 3 (born 2000-2099)
    (
        "national_id",
        re.compile(r"\b[23]\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{7}\b"),
        "[رقم_قومي_محجوب/REDACTED_NATIONAL_ID]",
    ),
    # General 14-digit fallback for Egyptian IDs
    (
        "national_id_loose",
        re.compile(r"\b[23]\d{13}\b"),
        "[رقم_قومي_محجوب/REDACTED_NATIONAL_ID]",
    ),
    # 2. Egyptian Mobile Phone Numbers (+2010..., 011..., 012..., 015...)
    (
        "mobile_phone",
        re.compile(r"(?:\+?20|0020|0)?1[0125]\d{8}\b"),
        "[رقم_هاتف_محجوب/REDACTED_PHONE]",
    ),
    # 3. Email addresses
    (
        "email",
        re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"),
        "[بريد_إلكتروني_محجوب/REDACTED_EMAIL]",
    ),
    # 4. Payment Cards (16 digits)
    (
        "credit_card",
        re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b"),
        "[بطاقة_دفع_محجوبة/REDACTED_CARD]",
    ),
]


def redact_egyptian_pii(text: str) -> str:
    """Masks all sensitive Egyptian and international PII in the given string."""
    if not text:
        return ""

    sanitized = text
    for _, pattern, replacement in PII_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)

    return sanitized


def contains_pii(text: str) -> bool:
    """Returns True if any PII pattern is detected in the text."""
    if not text:
        return False

    for _, pattern, _ in PII_PATTERNS:
        if pattern.search(text):
            return True

    return False
