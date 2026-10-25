"""Domain Boundary and Scope Refusal Gate for HistoryVerse Agents.

Guarantees agents stay strictly within their defined domain (History, Archaeology,
Antiquity, or a specific AR physical monument) and gracefully decline off-topic queries.
Fully configurable via constructor arguments or external configuration files.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from re import Pattern

# Default Out-of-Domain Keywords for Historical & Heritage Agents
DEFAULT_OUT_OF_DOMAIN_PATTERNS = [
    re.compile(
        r"(?:^|\s|[^\w])(?:و|ال|وال|بال)?(كورة|كوره|ماتش|أهداف|اهداف|دوري|كرة قدم|football|soccer|champions league)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|\s|[^\w])(?:و|ال|وال|بال)?(طبخة|طبخه|وصفة طعام|شوربة|شوربه|بيتزا|recipe|cooking|baking)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|\s|[^\w])(?:و|ال|وال|بال)?(برمجة|برمجه|بايثون|كود برمجي|javascript|python code|debug my code)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|\s|[^\w])(?:و|ال|وال|بال)?(طقس|درجات?\s+الحرارة|حرارة اليوم|مطر اليوم|weather forecast|weather)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|\s|[^\w])(?:و|ال|وال|بال|لل)?(تشخيص طبي|علاج الصداع|دواء|أعراض كورونا|medical treatment|prescription)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|\s|[^\w])(?:و|ال|وال|بال)?(سعر الدولار|أسعار الذهب|اسعار الذهب|بورصة|بورصه|bitcoin|crypto trading)\b",
        re.IGNORECASE,
    ),
]

# Default In-Domain Historical Concepts (Multilingual for International Museum Visitors)
DEFAULT_HISTORICAL_KEYWORDS = [
    # 1. Arabic (العربية)
    "تاريخ",
    "أثر",
    "آثار",
    "أثري",
    "معبد",
    "معابد",
    "هرم",
    "أهرام",
    "أهرامات",
    "تمثال",
    "تماثيل",
    "ملك",
    "ملوك",
    "فرعون",
    "فراعنة",
    "فرعوني",
    "حضارة",
    "حضارات",
    "متحف",
    "متاحف",
    "قلعة",
    "قلاع",
    "إسكندرية",
    "بطالمة",
    "بطلمي",
    "رومان",
    "روماني",
    "مملوكي",
    "مماليك",
    "عصر",
    "عصور",
    "نقش",
    "نقوش",
    "بردي",
    "برديات",
    "مخطوطة",
    "مخطوطات",
    "مقبرة",
    "مقابر",
    "قايتباي",
    "رشيد",
    "حجر",
    "عجائب",
    "رموز",
    "منارة",
    "فنار",
    "مكتبة",
    "مومياء",
    "خوفو",
    "خفرع",
    "رمسيس",
    "توت عنخ آمون",
    "إخناتون",
    "كليوباترا",
    "سيرابيوم",
    "السيرابيوم",
    "عمود السواري",
    "كوم الشقافة",
    "المسرح الروماني",
    "كولوسيوم",
    "روما",
    "طيبة",
    "الأقصر",
    "الكرنك",
    # 2. English
    "history",
    "historical",
    "monument",
    "monuments",
    "pyramid",
    "pyramids",
    "temple",
    "temples",
    "statue",
    "statues",
    "pharaoh",
    "pharaohs",
    "civilization",
    "museum",
    "alexandria",
    "antiquity",
    "ptolemy",
    "dynasty",
    "artifact",
    "archaeology",
    "archaeological",
    "inscription",
    "rosetta",
    "lighthouse",
    "citadel",
    "mummy",
    "qaitbay",
    "heritage",
    "cleopatra",
    "ramses",
    "serapeum",
    "colosseum",
    "rome",
    "roman",
    "ancient",
    "king",
    "kings",
    "queen",
    "queens",
    "dynastic",
    "chronicle",
    "chronicles",
    "karnak",
    "luxor",
    "thebes",
    "giza",
    "sphinx",
    "tomb",
    "tombs",
    "catacomb",
    "catacombs",
    "pompey",
    "pillar",
    "column",
    "excavation",
    "relic",
    "relics",
    "empire",
    "emperor",
    "caesar",
    "alexander",
    "library",
    # 3. French (Français)
    "histoire",
    "historique",
    "monument",
    "monuments",
    "pyramide",
    "pyramides",
    "temple",
    "temples",
    "statue",
    "statues",
    "pharaon",
    "pharaons",
    "civilisation",
    "musée",
    "alexandrie",
    "antiquité",
    "citadelle",
    "momie",
    "rosette",
    "ptolémée",
    "archéologie",
    "fouilles",
    "patrimoine",
    # 4. German (Deutsch)
    "geschichte",
    "historisch",
    "denkmal",
    "denkmäler",
    "pyramide",
    "pyramiden",
    "tempel",
    "statue",
    "statuen",
    "pharao",
    "pharaonen",
    "zivilisation",
    "museum",
    "museen",
    "alexandria",
    "antike",
    "zitadelle",
    "mumie",
    "rosette",
    "ptolomäus",
    "archäologie",
    "erbe",
    # 5. Spanish (Español)
    "historia",
    "histórico",
    "monumento",
    "monumentos",
    "pirámide",
    "pirámides",
    "templo",
    "templos",
    "estatua",
    "estatuas",
    "faraón",
    "faraones",
    "civilización",
    "museo",
    "museos",
    "alejandría",
    "antigüedad",
    "ciudadela",
    "momia",
    "roseta",
    "ptolomeo",
    "arqueología",
    "patrimonio",
    # 6. Italian (Italiano)
    "storia",
    "storico",
    "monumento",
    "monumenti",
    "piramide",
    "piramidi",
    "tempio",
    "templi",
    "statua",
    "statue",
    "faraone",
    "faraoni",
    "civiltà",
    "museo",
    "musei",
    "alessandria",
    "antichità",
    "cittadella",
    "mummia",
    "rosetta",
    "tolomeo",
    "archeologia",
    # 7. Russian (Русский)
    "история",
    "исторический",
    "монумент",
    "пирамида",
    "храм",
    "статуя",
    "фараон",
    "цивилизация",
    "музей",
    "александрия",
    "древность",
    "цитадель",
    "мумия",
    "розетта",
    # 8. Chinese (中文)
    "历史",
    "纪念碑",
    "金字塔",
    "神庙",
    "雕像",
    "法老",
    "文明",
    "博物馆",
    "亚历山大",
    "文物",
    "古迹",
]

# Localized refusal messages for international museum visitors
MULTILINGUAL_REFUSAL_MESSAGES = {
    "ar": "عذراً، بصفتي المرشد التاريخي لـ HistoryVerse، يقتصر تخصصي على الإجابة عن الحضارات والتاريخ والآثار والتراث الإنساني. هذا السؤال خارج نطاق اهتمامي.",
    "en": "I apologize, but as the HistoryVerse Historical Guide, my expertise is strictly dedicated to human history, classical antiquity, archaeology, and heritage monuments. This inquiry is outside my scope.",
    "fr": "Désolé, en tant que guide historique de HistoryVerse, mon expertise se limite strictement à l'histoire humaine, à l'archéologie et aux monuments antiques. Cette question est en dehors de mon domaine.",
    "de": "Entschuldigung, als HistoryVerse-Reiseleiter beschränkt sich meine Expertise ausschließlich auf die Menschheitsgeschichte, Archäologie und antike Denkmäler. Diese Anfrage liegt außerhalb meines Bereichs.",
    "es": "Lo siento, pero como guía histórico de HistoryVerse, mi experiencia se limita estrictamente a la historia humana, arqueología y monumentos antiguos. Esta consulta está fuera de mi alcance.",
    "it": "Spiacente, ma in qualità di guida storica di HistoryVerse, la mia esperienza è strettamente dedicata alla storia umana, all'archeologia e ai monumenti antichi. Questa richiesta è fuori dal mio ambito.",
    "ru": "Извините, но как исторический гид HistoryVerse, я специализируюсь исключительно на истории человечества, археологии и древних памятниках. Этот запрос находится вне моей компетенции.",
    "zh": "非常抱歉，作为 HistoryVerse 历史向导，我的专业知识仅限于人类历史、考古学和古代遗产古迹。该咨询超出了我的服务范围。",
}


@dataclass
class RefusalDecision:
    is_refused: bool
    refusal_reason: str | None = None
    refusal_message: str | None = None
    target_artifact: str | None = None


class DomainRefusalGate:
    """Enforces scope boundaries for HistoryVerse Historical & AR Guides.

    All keywords, patterns, and messages can be customized dynamically per agent.
    """

    def __init__(
        self,
        domain: str = "history",
        artifact_label: str | None = None,
        strict_mode: bool = True,
        in_domain_keywords: list[str] | None = None,
        out_of_domain_patterns: list[Pattern[str]] | None = None,
        custom_refusal_message: str | None = None,
    ) -> None:
        self.domain = domain
        self.artifact_label = artifact_label
        self.strict_mode = strict_mode
        self.in_domain_keywords = in_domain_keywords or DEFAULT_HISTORICAL_KEYWORDS
        self.out_of_domain_patterns = out_of_domain_patterns or DEFAULT_OUT_OF_DOMAIN_PATTERNS
        self.custom_refusal_message = custom_refusal_message

    def check_query(self, query: str) -> RefusalDecision:
        """Evaluates whether the incoming query is within the agent's historical scope."""
        if not query or not query.strip():
            return RefusalDecision(
                is_refused=True,
                refusal_reason="empty_query",
                refusal_message="يرجى طرح استفسار حول التاريخ أو المعالم والآثار التاريخية.",
            )

        clean_query = query.strip()

        # 1. AR Mode Strict Artifact Scoping
        if self.artifact_label and self.domain in ("augmented-reality", "ar"):
            art_clean = self.artifact_label.lower()
            q_clean = clean_query.lower()
            is_generic_tour = any(
                w in q_clean
                for w in [
                    "مين",
                    "إيه",
                    "تاريخ",
                    "كام",
                    "سنة",
                    "بنى",
                    "شرح",
                    "معلومات",
                    "what",
                    "who",
                    "when",
                    "tell me",
                    "explain",
                    "info",
                    "qui",
                    "quand",
                    "comment",
                    "wer",
                    "wie",
                    "wann",
                    "quién",
                    "cómo",
                    "chi",
                ]
            )
            if not is_generic_tour and art_clean not in q_clean:
                for pattern in self.out_of_domain_patterns:
                    if pattern.search(clean_query):
                        from agent.utils.multilingual import detect_language

                        lang = detect_language(clean_query)
                        if lang == "ar":
                            ar_msg = f"أنا مرشدك الخاص بالقطعة الأثرية: '{self.artifact_label}'. يمكنني فقط الإجابة عن التساؤلات المتعلقة بتاريخ هذا الأثر وتفاصيله المعمارية."
                        elif lang == "fr":
                            ar_msg = f"Je suis votre guide dédié pour l'exposition : '{self.artifact_label}'. Je ne peux répondre qu'aux questions relatives à son histoire et ses détails archéologiques."
                        elif lang == "de":
                            ar_msg = f"Ich bin Ihr persönlicher Ausstellungsführer für '{self.artifact_label}'. Ich kann nur Fragen zu seiner Geschichte und archäologischen Details beantworten."
                        elif lang == "es":
                            ar_msg = f"Soy su guía especializado para la pieza: '{self.artifact_label}'. Solo puedo responder preguntas sobre su historia y detalles arqueológicos."
                        elif lang == "it":
                            ar_msg = f"Sono la tua guida dedicata per l'opera: '{self.artifact_label}'. Posso rispondere solo a domande riguardanti la sua storia e dettagli archeologici."
                        else:
                            ar_msg = f"I am your dedicated tour guide for the monument/artifact: '{self.artifact_label}'. I can only answer questions related to its history and archaeological significance."

                        return RefusalDecision(
                            is_refused=True,
                            refusal_reason="ar_out_of_scope",
                            refusal_message=ar_msg,
                            target_artifact=self.artifact_label,
                        )

        from agent.utils.multilingual import detect_language

        lang = detect_language(clean_query)
        default_msg = MULTILINGUAL_REFUSAL_MESSAGES.get(lang, MULTILINGUAL_REFUSAL_MESSAGES["en"])

        # 2. Check for general out-of-domain patterns
        for pattern in self.out_of_domain_patterns:
            if pattern.search(clean_query):
                return RefusalDecision(
                    is_refused=True,
                    refusal_reason="out_of_domain",
                    refusal_message=self.custom_refusal_message or default_msg,
                )

        # 3. If query contains zero in-domain context and is sufficiently long
        if self.strict_mode and len(clean_query.split()) > 3:
            lower_q = clean_query.lower()
            in_domain_set = set(self.in_domain_keywords)
            if self.artifact_label:
                for w in re.findall(r"\w+", self.artifact_label.lower()):
                    if len(w) > 2:
                        in_domain_set.add(w)
            has_domain_term = any(kw in lower_q for kw in in_domain_set)
            if not has_domain_term:
                return RefusalDecision(
                    is_refused=True,
                    refusal_reason="no_historical_context",
                    refusal_message=self.custom_refusal_message or default_msg,
                )

        return RefusalDecision(is_refused=False)


# Backwards compatibility alias
LegalRefusalGate = DomainRefusalGate
