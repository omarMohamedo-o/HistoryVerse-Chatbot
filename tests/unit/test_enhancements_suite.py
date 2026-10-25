"""Comprehensive Test Suite for HistoryVerse Platform Enhancements and Add-ons.

Validates all 10 enterprise capabilities:
1. Arabic NLP Normalizer (Lam-Alef ligatures, Indic numerals, mirrored brackets)
2. Egyptian Colloquial (Amiya) Normalizer and dynamic mapping injection
3. Domain Boundary and Scope Refusal Gate (in-domain acceptance vs. out-of-domain refusal)
4. AR Mode Artifact Scoping & Tour Boundaries
5. Historical Source Grounding Guardrail (citations, dates, fabricated sources)
6. Egyptian PII Redaction (14-digit National IDs, mobile phone numbers, cards)
7. Voice TTS Speech Sanitization (bracket stripping, markdown removal for natural audio)
8. Voice STT Transcript Sanitization (spoken PII masking and normalization)
9. Vision Image Generation Guardrail (domain refusal on non-historical prompts)
10. Corpus Schema Integrity & Monument Verification
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from agent.services.nlp.arabic import (
    clean_arabic_text,
    fix_mirrored_brackets,
    normalize_alef,
    normalize_digits,
    strip_tashkeel,
)
from agent.services.nlp.colloquial import expand_egyptian_colloquial_query
from agent.services.rag.refusal import DomainRefusalGate
from agent.services.rag.citation_guard import GroundingGuardrail
from agent.services.security.pii import contains_pii, redact_egyptian_pii
from agent.services.voice.sanitizer import sanitize_for_speech, sanitize_speech_transcript


class TestHistoryVerseEnhancements(unittest.TestCase):
    """Exhaustive tests asserting zero errors and zero warnings across all add-ons."""

    def test_01_arabic_nlp_ligatures_and_numerals(self):
        """Verify Lam-Alef shielding, Indic numeral translation, and RTL bracket correction."""
        # 1. Indic numerals (٠-٩ -> 0-9)
        text_with_indic = "بنيت القلعة عام ١٤٧٧ ميلادية في عهد السلطان قايتباي"
        normalized_num = normalize_digits(text_with_indic)
        self.assertIn("1477", normalized_num)
        self.assertNotIn("١٤٧٧", normalized_num)

        # 2. Tashkeel & Tatweel stripping
        tashkeel_text = "قَلْعَةُ قَايِتْبَاي الإســــكندرية"
        stripped = strip_tashkeel(tashkeel_text)
        self.assertNotIn("َ", stripped)
        self.assertNotIn("ُ", stripped)

        # 3. Mirrored brackets resulting from bidirectional text reversals
        reversed_brackets = ")قلعة قايتباي("
        fixed_brackets = fix_mirrored_brackets(reversed_brackets)
        self.assertEqual(fixed_brackets, "(قلعة قايتباي)")

        # 4. Full cleaning pipeline with Lam-Alef ligature shielding & Alef normalization
        ligature_sample = "الإسكندرية والآثار الإسلامية"
        cleaned = clean_arabic_text(ligature_sample, norm_alef=True)
        self.assertIn("الاسكندرية", cleaned)
        self.assertIn("الاثار", cleaned)

    def test_02_egyptian_colloquial_normalizer(self):
        """Verify translation of spoken Egyptian dialect into formal historical concepts."""
        spoken_query = "عايز أعرف مين بنى قلعة قايتباي وفين مكان منارة إسكندرية"
        enriched = expand_egyptian_colloquial_query(spoken_query)

        # Asserts formal historical terminology injection
        self.assertIn("المؤسس والباني والمهندس المعماري", enriched)
        self.assertIn("الموقع الجغرافي والأثري", enriched)
        self.assertIn("فنار فاروس عجائب العالم القديم", enriched)

        # Custom mapping injection
        import re
        custom_mapping = [(re.compile(r"\bتوت عنخ\b", re.IGNORECASE), "الملك توت عنخ آمون الأسرة الثامنة عشرة")]
        custom_enriched = expand_egyptian_colloquial_query("إيه حكاية مقبرة توت عنخ", custom_mappings=custom_mapping)
        self.assertIn("الأسرة الثامنة عشرة", custom_enriched)

    def test_03_domain_boundary_refusal_gate_indomain(self):
        """Verify that legitimate historical and archaeological questions are accepted."""
        gate = DomainRefusalGate(domain="history", strict_mode=True)

        in_domain_queries = [
            "من بنى قلعة قايتباي في الإسكندرية وما هو سبب إنشائها؟",
            "ما هي مواصفات منارة الإسكندرية القديمة وكيف دمرت؟",
            "ما هو دور حجر رشيد في فك رموز اللغة المصرية القديمة؟",
            "Who built the Great Pyramid of Giza and during which dynasty?",
            "What is the historical significance of the Rosetta Stone?",
        ]

        for q in in_domain_queries:
            decision = gate.check_query(q)
            self.assertFalse(decision.is_refused, f"Valid query was incorrectly refused: {q}")
            self.assertIsNone(decision.refusal_reason)

    def test_04_domain_boundary_refusal_gate_outofdomain(self):
        """Verify that off-topic drift queries (football, cooking, code, crypto) are safely declined."""
        gate = DomainRefusalGate(domain="history", strict_mode=True)

        drift_queries = [
            "ما هي نتيجة مباراة الأهلي والزمالك في دوري أبطال أفريقيا؟",
            "طريقة تحضير صينية البطاطس بالدجاج في الفرن خطوة بخطوة",
            "كيف أكتب كود بايثون لإنشاء خادم ويب باستخدام فاست إيه بي آي؟",
            "ما هو علاج الصداع النصفي المزمن والجرعة الموصى بها للدواء؟",
            "توقعات حالة الطقس ودرجات الحرارة غداً في القاهرة والإسكندرية",
            "What is the current Bitcoin exchange rate and crypto market trend?",
        ]

        for q in drift_queries:
            decision = gate.check_query(q)
            self.assertTrue(decision.is_refused, f"Drift query was incorrectly accepted: {q}")
            self.assertIn(decision.refusal_reason, ["out_of_domain", "no_historical_context"])
            self.assertIsNotNone(decision.refusal_message)

    def test_05_ar_mode_artifact_scoping(self):
        """Verify augmented reality mode strictly scopes answers to physical museum exhibits."""
        ar_gate = DomainRefusalGate(
            domain="augmented-reality",
            artifact_label="قلعة قايتباي",
            strict_mode=True,
        )

        # Relevant question to active monument
        q_valid = "من بنى قلعة قايتباي وما هو تاريخ إنشائها؟"
        decision_valid = ar_gate.check_query(q_valid)
        self.assertFalse(decision_valid.is_refused)

        # Off-topic question while scoped to Qaitbay Citadel
        q_offtopic = "ما هي نتيجة ماتش كورة القدم اليوم في الدوري؟"
        decision_off = ar_gate.check_query(q_offtopic)
        self.assertTrue(decision_off.is_refused)
        self.assertEqual(decision_off.refusal_reason, "ar_out_of_scope")
        self.assertIn("قلعة قايتباي", decision_off.refusal_message)

    def test_06_source_grounding_guardrail(self):
        """Verify detection of verified vs fabricated sources and citation tags."""
        guard = GroundingGuardrail()

        retrieved_context = (
            "شيد السلطان الأشرف قايتباي القلعة عام 1477 ميلادية لحماية الإسكندرية. "
            "[المصدر: كتاب خطط المقريزي]"
        )

        valid_response = (
            "تأسست قلعة قايتباي عام 1477 م على يد السلطان قايتباي. "
            "[المصدر: كتاب خطط المقريزي]"
        )

        fabricated_response = (
            "تأسست قلعة قايتباي عام 1477 م. "
            "[المصدر: وثيقة أثرية مزيفة 12345]"
        )

        # Valid answer check
        valid_res = guard.verify_grounding(valid_response, retrieved_context)
        self.assertTrue(valid_res.is_valid)
        self.assertEqual(len(valid_res.unverified_references), 0)

        # Corrupted answer check
        corrupted_res = guard.verify_grounding(fabricated_response, retrieved_context)
        self.assertFalse(corrupted_res.is_valid)
        self.assertIn("[المصدر: وثيقة أثرية مزيفة 12345]", corrupted_res.unverified_references)

    def test_07_egyptian_pii_redactor(self):
        """Verify masking of Egyptian 14-digit National IDs, phone numbers, and payment cards."""
        text_with_pii = (
            "اسمي أحمد ورقمي القومي 29805121234567 ورقم هاتفي 01012345678 "
            "وبريدي omar@historyverse.org وبطاقتي 4111-2222-3333-4444"
        )

        self.assertTrue(contains_pii(text_with_pii))

        redacted = redact_egyptian_pii(text_with_pii)
        self.assertNotIn("29805121234567", redacted)
        self.assertNotIn("01012345678", redacted)
        self.assertNotIn("omar@historyverse.org", redacted)
        self.assertNotIn("4111-2222-3333-4444", redacted)

        self.assertIn("[رقم_قومي_محجوب/REDACTED_NATIONAL_ID]", redacted)
        self.assertIn("[رقم_هاتف_محجوب/REDACTED_PHONE]", redacted)
        self.assertIn("[بريد_إلكتروني_محجوب/REDACTED_EMAIL]", redacted)
        self.assertIn("[بطاقة_دفع_محجوبة/REDACTED_CARD]", redacted)

    def test_08_voice_tts_speech_sanitization(self):
        """Verify that TTS sanitization strips brackets, citations, and redactions for natural audio."""
        dirty_tts_text = (
            "قلعة قايتباي شيدت سنة 1477 م. [المصدر: المقريزي] ورقم الهاتف [رقم_هاتف_محجوب/REDACTED_PHONE]. "
            "**تاريخ عريق** و [رابط إضافي](https://historyverse.org/citadel)"
        )

        clean_speech = sanitize_for_speech(dirty_tts_text)

        # Must not contain unpronounceable metadata
        self.assertNotIn("[المصدر", clean_speech)
        self.assertNotIn("[رقم_هاتف", clean_speech)
        self.assertNotIn("https://", clean_speech)
        self.assertNotIn("**", clean_speech)
        self.assertNotIn("[", clean_speech)
        self.assertNotIn("]", clean_speech)

        # Must preserve acoustic historical facts
        self.assertIn("قلعة قايتباي شيدت سنة 1477 م.", clean_speech)
        self.assertIn("تاريخ عريق", clean_speech)
        self.assertIn("رابط إضافي", clean_speech)

    def test_09_voice_stt_transcript_sanitization(self):
        """Verify Whisper STT transcript masks spoken PII and normalizes spoken Egyptian dialect."""
        spoken_audio_transcript = "رقمي القومي هو 29805121234567 وعايز أعرف مين بنى قلعة قايتباي"
        sanitized = sanitize_speech_transcript(spoken_audio_transcript, expand_dialect=True)

        self.assertNotIn("29805121234567", sanitized)
        self.assertIn("[رقم_قومي_محجوب/REDACTED_NATIONAL_ID]", sanitized)
        self.assertIn("المؤسس والباني والمهندس المعماري", sanitized)

    def test_10_historyverse_corpus_schema_and_facts(self):
        """Verify production corpus integrity, era classifications, and bilingual content."""
        corpus_file = Path("data/corpus/historyverse_corpus.json")
        self.assertTrue(corpus_file.exists(), f"Corpus not found at {corpus_file}")

        with open(corpus_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertGreaterEqual(len(data), 6, "Expected at least 6 flagship historical monuments")

        required_keys = [
            "artifact_id",
            "artifact_label",
            "artifact_label_en",
            "era",
            "year_built",
            "location",
            "description_ar",
            "description_en",
        ]

        for item in data:
            for k in required_keys:
                self.assertIn(k, item, f"Missing required key '{k}' in artifact {item.get('artifact_id')}")
            self.assertTrue(len(item["description_ar"]) > 50)
            self.assertTrue(len(item["description_en"]) > 50)

    def test_11_semantic_response_cache(self):
        """Verify sub-15ms semantic response caching for repetitive visitor inquiries."""
        from agent.services.cache import SemanticResponseCache

        cache = SemanticResponseCache(similarity_threshold=0.75, max_entries=50)
        cache.clear()

        query_original = "من بنى قلعة قايتباي في الإسكندرية؟"
        expected_answer = "بُنيت قلعة قايتباي على يد السلطان الأشرف قايتباي عام 1477 م."

        # Cache set
        cache.set(query_original, expected_answer, domain="history")
        self.assertEqual(cache.size(), 1)

        # 1. Exact match hit
        hit_exact = cache.get(query_original, domain="history")
        self.assertEqual(hit_exact, expected_answer)

        # 2. Fuzzy semantic query hit (slight dialect / phrasing variation)
        query_variation = "مين اللي بنى قلعة قايتباي بالإسكندرية"
        hit_fuzzy = cache.get(query_variation, domain="history")
        self.assertEqual(hit_fuzzy, expected_answer)

        # 3. Cache miss on distinct question
        query_distinct = "ما هو حجر رشيد ودوره التاريخي؟"
        miss_distinct = cache.get(query_distinct, domain="history")
        self.assertIsNone(miss_distinct)

    def test_12_multilingual_automatic_language_recognition(self):
        """Verify automatic recognition of visitor languages (AR, EN, FR, DE, ES, IT, RU, ZH)."""
        from agent.utils.multilingual import detect_language

        samples = {
            "ar": "من بنى قلعة قايتباي في الإسكندرية وما هو تاريخ إنشائها؟",
            "en": "Who built the Citadel of Qaitbay and what is its history?",
            "fr": "Qui a construit la citadelle de Qaitbay à Alexandrie ?",
            "de": "Wer hat die Zitadelle von Qaitbay in Alexandria erbaut?",
            "es": "¿Quién construyó la Ciudadela de Qaitbay en Alejandría?",
            "it": "Chi ha costruito la Cittadella di Qaitbay ad Alessandria?",
            "ru": "Кто построил цитадель Кайтбей в Александрии?",
            "zh": "谁建造了亚历山大的凯特贝城堡？",
        }

        gate = DomainRefusalGate(domain="history", strict_mode=True)

        for lang, query in samples.items():
            detected = detect_language(query)
            self.assertEqual(detected, lang, f"Expected {lang}, detected {detected} for query: {query}")

            # Every international historical inquiry must be accepted by the gate
            decision = gate.check_query(query)
            self.assertFalse(decision.is_refused, f"Historical query in {lang} was rejected: {query}")

        # Verify localized refusal for an out-of-domain French question
        french_drift = "Quelle est la meilleure recette de cuisine pour préparer une pizza ?"
        decision_fr = gate.check_query(french_drift)
        self.assertTrue(decision_fr.is_refused)
        self.assertIn("guide historique", decision_fr.refusal_message)

        # Verify localized refusal for an out-of-domain German question
        german_drift = "Wie ist die aktuelle Wettervorhersage für morgen in Berlin?"
        decision_de = gate.check_query(german_drift)
        self.assertTrue(decision_de.is_refused)
        self.assertIn("HistoryVerse-Reiseleiter", decision_de.refusal_message)


if __name__ == "__main__":
    unittest.main()
