"""Egyptian Colloquial (Amiya) Normalizer for History, Heritage, and Tourism.

Translates everyday Egyptian spoken dialect phrases (e.g. 'عايز أعرف مين بنى', 'إيه حكاية قلعة قايتباي')
into structured historical search terms for high-precision retrieval across HistoryVerse corpora.
Supports custom mapping injection for arbitrary operational domains.
"""

from __future__ import annotations

import re
from re import Pattern

# Default mapping of Egyptian colloquial expressions to formal historical inquiry concepts
DEFAULT_COLLOQUIAL_MAPPINGS: list[tuple[Pattern[str], str]] = [
    (re.compile(r"(?:^|\s|[^\w])و?عايز أعرف\b", re.IGNORECASE), "ما هو التاريخ والتوثيق الأثري لـ"),
    (re.compile(r"(?:^|\s|[^\w])و?إيه حكاية\b", re.IGNORECASE), "تاريخ ونشأة وأهمية"),
    (
        re.compile(r"(?:^|\s|[^\w])و?مين بنى\b", re.IGNORECASE),
        "المؤسس والباني والمهندس المعماري لـ",
    ),
    (re.compile(r"(?:^|\s|[^\w])و?مين اللي عمل\b", re.IGNORECASE), "صاحب الأثر والمنشئ التاريخي"),
    (
        re.compile(r"(?:^|\s|[^\w])و?اتعملت سنة كام\b", re.IGNORECASE),
        "تاريخ الإنشاء وعصر التأسيس والقرن",
    ),
    (re.compile(r"(?:^|\s|[^\w])و?فين مكان\b", re.IGNORECASE), "الموقع الجغرافي والأثري لـ"),
    (
        re.compile(r"(?:^|\s|[^\w])و?إيه أهميتها\b", re.IGNORECASE),
        "القيمة التاريخية والأثرية والمعمارية",
    ),
    (
        re.compile(r"(?:^|\s|[^\w])و?اتدمرت إزاي\b", re.IGNORECASE),
        "أسباب الهدم والزلازل والحرائق التاريخية",
    ),
    (
        re.compile(r"(?:^|\s|[^\w])و?اتسرقت إزاي\b", re.IGNORECASE),
        "تاريخ الاكتشاف وخروج الأثر والنقل لمتاحف العالم",
    ),
    (
        re.compile(r"(?:^|\s|[^\w])و?مين رمسيس\b", re.IGNORECASE),
        "الملك رمسيس الثاني الأسرة التاسعة عشرة الدولة الحديثة",
    ),
    (
        re.compile(r"(?:^|\s|[^\w])و?مين الإسكندر\b", re.IGNORECASE),
        "الإسكندر الأكبر المقدوني تأسيس مدينة الإسكندرية 331 ق.م",
    ),
    (
        re.compile(r"(?:^|\s|[^\w])و?حجر رشيد\b", re.IGNORECASE),
        "حجر رشيد شامبليون فك رموز اللغة الهيروغليفية",
    ),
    (
        re.compile(r"(?:^|\s|[^\w])و?منارة إسكندرية\b", re.IGNORECASE),
        "منارة الإسكندرية فنار فاروس عجائب العالم القديم قلعة قايتباي",
    ),
    (
        re.compile(r"(?:^|\s|[^\w])و?مكتبة إسكندرية القديمة\b", re.IGNORECASE),
        "مكتبة الإسكندرية الملكية القديمة بطليموس الأول وسيرابيون",
    ),
]


def expand_egyptian_colloquial_query(
    query: str,
    custom_mappings: list[tuple[Pattern[str], str]] | None = None,
) -> str:
    """Expands Egyptian colloquial query terms with formal domain search terms.

    Args:
        query: User input query in Egyptian dialect.
        custom_mappings: Optional custom regex mappings to override or extend defaults.

    Returns:
        Enriched query string with formal domain terminology.
    """
    if not query:
        return ""

    mappings = custom_mappings or DEFAULT_COLLOQUIAL_MAPPINGS
    expanded_terms: list[str] = []
    normalized_query = query

    for pattern, equivalent in mappings:
        if pattern.search(normalized_query):
            expanded_terms.append(equivalent)

    if not expanded_terms:
        return query

    enriched = f"{query} {' '.join(expanded_terms)}"
    return re.sub(r"\s+", " ", enriched).strip()
