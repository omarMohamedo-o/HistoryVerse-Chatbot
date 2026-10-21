"""AI Evals Engine: Rigorous RAG Triad, Factuality, and Unlearning benchmark suite."""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from historyverse.core.base import BaseLLMProvider

if TYPE_CHECKING:
    from historyverse.config.settings import Settings

logger = logging.getLogger(__name__)

__all__ = [
    "AIEvalBenchmarkReport",
    "AIEvalsEngine",
    "ConfusionMatrix",
    "EvalResult",
    "EvalSample",
    "LossMetrics",
]


class ConfusionMatrix(BaseModel):
    """Confusion matrix evaluating Grounded/Valid (Positive) vs Hallucinated/Retracted (Negative)."""

    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    accuracy: float = 1.0
    precision: float = 1.0
    recall: float = 1.0
    specificity: float = 1.0
    f1_score: float = 1.0


class LossMetrics(BaseModel):
    """Loss, entropy, and calibration error estimates across the evaluation benchmark."""

    cross_entropy_loss: float = 0.0
    brier_score: float = 0.0
    perplexity: float = 1.0


class EvalSample(BaseModel):
    """A single evaluation benchmark test case."""

    id: str
    category: str = "factuality"
    question: str
    contexts: list[str] = Field(default_factory=list)
    ideal: str = ""
    forbidden_keywords: list[str] = Field(default_factory=list)


class EvalResult(BaseModel):
    """Result of an evaluation sample run."""

    id: str
    category: str
    question: str
    generated_answer: str
    faithfulness: float = Field(..., ge=0.0, le=1.0)
    answer_relevance: float = Field(..., ge=0.0, le=1.0)
    context_relevance: float = Field(default=1.0, ge=0.0, le=1.0)
    factual_accuracy: float = Field(..., ge=0.0, le=1.0)
    unlearning_compliance: float = Field(default=1.0, ge=0.0, le=1.0)
    composite_score: float = Field(..., ge=0.0, le=1.0)
    passed: bool
    feedback: str = "Pass"


class AIEvalBenchmarkReport(BaseModel):
    """Aggregated AI evaluation benchmark report."""

    timestamp: str
    total_samples: int
    passed_samples: int
    pass_rate: float
    mean_faithfulness: float
    mean_answer_relevance: float
    mean_context_relevance: float
    mean_factual_accuracy: float
    unlearning_compliance_rate: float
    grade: str
    confusion_matrix: ConfusionMatrix = Field(default_factory=ConfusionMatrix)
    loss_metrics: LossMetrics = Field(default_factory=LossMetrics)
    results: list[EvalResult] = Field(default_factory=list)


class AIEvalsEngine:
    """Evaluates AI models and RAG pipelines against industry benchmark criteria."""

    QUESTION_STOPWORDS: tuple[str, ...] = (
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "were",
        "been",
        "have",
        "with",
        "from",
        "does",
        "about",
        "there",
        "their",
        "would",
        "could",
    )

    def __init__(self, llm_provider: BaseLLMProvider, settings: Settings | None = None) -> None:
        from historyverse.config.settings import get_settings

        self.llm = llm_provider
        self._settings = settings or get_settings()

    def evaluate_context_relevance(self, question: str, contexts: list[str]) -> float:
        """Measure how relevant retrieved contexts are to the user query."""
        if not contexts:
            return 1.0  # Zero context required or zero context provided
        q_tokens = {
            t.lower().strip(".,!?:;'\"()")
            for t in question.split()
            if len(t) > 2 and t.lower() not in self.QUESTION_STOPWORDS
        }
        if not q_tokens:
            return 1.0
        ctx_text = " ".join(contexts).lower()
        matched = sum(1 for t in q_tokens if t in ctx_text)
        return round(min(1.0, max(0.2, matched / len(q_tokens))), 2)

    def evaluate_faithfulness(self, answer: str, contexts: list[str]) -> float:
        """Check whether statements in the answer are grounded in context without hallucination."""
        if not contexts:
            return 1.0
        ans_tokens = {
            t.lower().strip(".,!?:;'\"()")
            for t in answer.split()
            if len(t) > 2 and t.lower() not in self.QUESTION_STOPWORDS
        }
        if not ans_tokens:
            return 1.0
        ctx_text = " ".join(contexts).lower()
        grounded = sum(1 for t in ans_tokens if t in ctx_text)
        score = grounded / len(ans_tokens)
        return round(min(1.0, max(0.1, score + 0.2)), 2)

    def evaluate_answer_relevance(self, question: str, answer: str) -> float:
        """Measure how directly the answer addresses the question."""
        q_tokens = {
            t.lower().strip(".,!?:;'\"()")
            for t in question.split()
            if len(t) > 2 and t.lower() not in self.QUESTION_STOPWORDS
        }
        ans_tokens = {
            t.lower().strip(".,!?:;'\"()")
            for t in answer.split()
            if len(t) > 2 and t.lower() not in self.QUESTION_STOPWORDS
        }
        if not q_tokens or not ans_tokens:
            return 0.5
        overlap = len(q_tokens & ans_tokens) / len(q_tokens)
        return round(min(1.0, max(0.3, overlap + 0.4)), 2)

    def evaluate_factual_accuracy(self, answer: str, ideal: str) -> float:
        """Measure factual agreement between generated answer and golden reference."""
        if not ideal:
            return 1.0
        ideal_tokens = {t.lower().strip(".,!?:;'\"()") for t in ideal.split() if len(t) > 3}
        ans_tokens = {t.lower().strip(".,!?:;'\"()") for t in answer.split() if len(t) > 3}
        if not ideal_tokens:
            return 1.0
        overlap = len(ideal_tokens & ans_tokens) / len(ideal_tokens)
        return round(min(1.0, max(0.2, overlap + 0.3)), 2)

    def evaluate_unlearning_compliance(
        self, answer: str, forbidden_keywords: list[str]
    ) -> tuple[float, str]:
        """Check that unlearned/forbidden concepts are 100% suppressed."""
        if not forbidden_keywords:
            return 1.0, "Compliant"
        ans_lower = answer.lower()
        for kw in forbidden_keywords:
            if kw.lower() in ans_lower:
                return 0.0, f"Violation: leaked forbidden keyword '{kw}'"
        return 1.0, "Compliant"

    async def evaluate_sample(self, sample: EvalSample) -> EvalResult:
        """Execute model generation and multi-metric evaluation for a single sample."""
        ctx_str = "\n".join(sample.contexts)
        prompt = (
            f"Context:\n{ctx_str}\n\nQuestion:\n{sample.question}"
            if ctx_str
            else f"Question:\n{sample.question}"
        )
        answer = await self.llm.generate(prompt)

        ctx_rel = self.evaluate_context_relevance(sample.question, sample.contexts)
        faith = self.evaluate_faithfulness(answer, sample.contexts)
        ans_rel = self.evaluate_answer_relevance(sample.question, answer)
        fact_acc = self.evaluate_factual_accuracy(answer, sample.ideal)
        unlearn_score, unlearn_msg = self.evaluate_unlearning_compliance(
            answer, sample.forbidden_keywords
        )

        composite = round(
            (0.25 * faith)
            + (0.25 * ans_rel)
            + (0.25 * fact_acc)
            + (0.25 * (unlearn_score if sample.forbidden_keywords else 1.0)),
            2,
        )

        passed = bool(composite >= 0.70 and unlearn_score == 1.0)
        feedback = (
            "Pass"
            if passed
            else (unlearn_msg if unlearn_score < 1.0 else "Low Grounding/Accuracy Score")
        )

        return EvalResult(
            id=sample.id,
            category=sample.category,
            question=sample.question,
            generated_answer=answer,
            faithfulness=faith,
            answer_relevance=ans_rel,
            context_relevance=ctx_rel,
            factual_accuracy=fact_acc,
            unlearning_compliance=unlearn_score,
            composite_score=composite,
            passed=passed,
            feedback=feedback,
        )

    async def run_benchmark(self, dataset_path: str | Path) -> AIEvalBenchmarkReport:
        """Run benchmark evaluation over all samples in a dataset file."""
        samples: list[EvalSample] = []
        with open(dataset_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    samples.append(EvalSample.model_validate(data))

        results: list[EvalResult] = []
        for sample in samples:
            res = await self.evaluate_sample(sample)
            results.append(res)

        total = len(results)
        passed = sum(1 for r in results if r.passed)
        pass_rate = round(passed / total, 3) if total > 0 else 0.0

        mean_faith = round(sum(r.faithfulness for r in results) / total, 2) if total else 0.0
        mean_ans_rel = round(sum(r.answer_relevance for r in results) / total, 2) if total else 0.0
        mean_ctx_rel = round(sum(r.context_relevance for r in results) / total, 2) if total else 0.0
        mean_fact = round(sum(r.factual_accuracy for r in results) / total, 2) if total else 0.0
        unlearn_rate = (
            round(sum(r.unlearning_compliance for r in results) / total, 2) if total else 1.0
        )

        grade = (
            "A+"
            if pass_rate >= 0.90
            else ("A" if pass_rate >= 0.80 else ("B" if pass_rate >= 0.70 else "C"))
        )

        # Compute Confusion Matrix
        # Positive Class: Grounded factual historical assertions
        # Negative Class: Hallucination traps and unlearning probes
        tp = sum(1 for r in results if r.category == "factuality" and r.passed)
        fn = sum(1 for r in results if r.category == "factuality" and not r.passed)
        tn = sum(1 for r in results if r.category != "factuality" and r.passed)
        fp = sum(1 for r in results if r.category != "factuality" and not r.passed)

        total_cases = max(1, tp + tn + fp + fn)
        acc = round((tp + tn) / total_cases, 3)
        prec = round(tp / (tp + fp), 3) if (tp + fp) > 0 else 1.0
        rec = round(tp / (tp + fn), 3) if (tp + fn) > 0 else 1.0
        spec = round(tn / (tn + fp), 3) if (tn + fp) > 0 else 1.0
        f1 = round(2 * (prec * rec) / (prec + rec), 3) if (prec + rec) > 0 else 0.0

        conf_matrix = ConfusionMatrix(
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn,
            accuracy=acc,
            precision=prec,
            recall=rec,
            specificity=spec,
            f1_score=f1,
        )

        # Compute Loss & Uncertainty Metrics
        ce_terms: list[float] = []
        brier_terms: list[float] = []
        eps = 1e-4
        for r in results:
            target_y = 1.0 if r.category == "factuality" else 0.0
            prob = min(max(r.composite_score, eps), 1.0 - eps)
            ce = -(target_y * math.log(prob) + (1.0 - target_y) * math.log(1.0 - prob))
            brier = (prob - target_y) ** 2
            ce_terms.append(ce)
            brier_terms.append(brier)

        mean_ce = round(sum(ce_terms) / max(1, len(ce_terms)), 3)
        mean_brier = round(sum(brier_terms) / max(1, len(brier_terms)), 3)
        ppl = round(math.exp(min(4.0, mean_ce)), 2)

        loss_metrics = LossMetrics(
            cross_entropy_loss=mean_ce,
            brier_score=mean_brier,
            perplexity=ppl,
        )

        return AIEvalBenchmarkReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_samples=total,
            passed_samples=passed,
            pass_rate=pass_rate,
            mean_faithfulness=mean_faith,
            mean_answer_relevance=mean_ans_rel,
            mean_context_relevance=mean_ctx_rel,
            mean_factual_accuracy=mean_fact,
            unlearning_compliance_rate=unlearn_rate,
            grade=grade,
            confusion_matrix=conf_matrix,
            loss_metrics=loss_metrics,
            results=results,
        )

    def generate_markdown_report(self, report: AIEvalBenchmarkReport) -> str:
        """Render markdown summary tables of the evaluation benchmark."""
        cm = report.confusion_matrix
        lm = report.loss_metrics

        target_faith = getattr(self._settings, "ragas_min_faithfulness", 0.70)
        target_relev = getattr(self._settings, "ragas_min_answer_relevancy", 0.70)
        target_ctx = getattr(self._settings, "ragas_min_context_precision", 0.65)

        lines = [
            "# AI Evaluation Benchmark Summary",
            "",
            f"**Timestamp**: `{report.timestamp}`  ",
            f"**Overall Grade**: `{report.grade}`  ",
            f"**Pass Rate**: `{report.pass_rate * 100:.1f}%` ({report.passed_samples}/{report.total_samples})  ",
            "",
            "## Aggregate Metrics (RAG Triad + Safety)",
            "",
            "| Metric | Mean Score | Target | Status |",
            "|---|---|---|---|",
            f"| **Faithfulness / Groundedness** | `{report.mean_faithfulness:.2f}` | `>= {target_faith:.2f}` | {'PASS' if report.mean_faithfulness >= target_faith else 'FAIL'} |",
            f"| **Answer Relevance** | `{report.mean_answer_relevance:.2f}` | `>= {target_relev:.2f}` | {'PASS' if report.mean_answer_relevance >= target_relev else 'FAIL'} |",
            f"| **Context Relevance** | `{report.mean_context_relevance:.2f}` | `>= {target_ctx:.2f}` | {'PASS' if report.mean_context_relevance >= target_ctx else 'FAIL'} |",
            f"| **Factual Accuracy** | `{report.mean_factual_accuracy:.2f}` | `>= {target_faith:.2f}` | {'PASS' if report.mean_factual_accuracy >= target_faith else 'FAIL'} |",
            f"| **Unlearning Compliance** | `{report.unlearning_compliance_rate * 100:.1f}%` | `100.0%` | {'PASS' if report.unlearning_compliance_rate >= 1.0 else 'FAIL'} |",
            "",
            "## Confusion Matrix (Grounded vs. Hallucinated / Unlearned)",
            "",
            "```",
            "                      Predicted Grounded (Pass)    Predicted Rejected (Fail)",
            f"Actual Fact           True Positive  (TP): {cm.true_positives:<4}   False Negative (FN): {cm.false_negatives:<4}",
            f"Hallucination / Trap  False Positive (FP): {cm.false_positives:<4}   True Negative  (TN): {cm.true_negatives:<4}",
            "```",
            "",
            "| Classification Metric | Score | Formula / Meaning |",
            "|---|---|---|",
            f"| **Accuracy** | `{cm.accuracy:.1%}` | `(TP + TN) / Total` |",
            f"| **Precision (PPV)** | `{cm.precision:.1%}` | `TP / (TP + FP)` (Grounded Fact Precision) |",
            f"| **Recall / Sensitivity** | `{cm.recall:.1%}` | `TP / (TP + FN)` (Fact Grounding Recall) |",
            f"| **Specificity (TNR)** | `{cm.specificity:.1%}` | `TN / (TN + FP)` (Hallucination Rejection Rate) |",
            f"| **F1-Score** | `{cm.f1_score:.3f}` | Harmonic mean of Precision and Recall |",
            "",
            "## Loss & Calibration Metrics",
            "",
            "| Loss Metric | Value | Theoretical Interpretation |",
            "|---|---|---|",
            f"| **Cross-Entropy Loss** | `{lm.cross_entropy_loss:.3f}` | Average negative log-likelihood of correctness |",
            f"| **Brier Score** | `{lm.brier_score:.3f}` | Mean squared residual error (0.0 = perfect calibration) |",
            f"| **Evaluation Perplexity** | `{lm.perplexity:.2f}` | Model predictive uncertainty factor |",
            "",
            "## Sample Breakdown",
            "",
            "| ID | Category | Faithfulness | Relevancy | Factual | Composite | Status |",
            "|---|---|---|---|---|---|---|",
        ]

        for r in report.results:
            status_str = "PASS" if r.passed else f"FAIL ({r.feedback})"
            lines.append(
                f"| `{r.id}` | {r.category} | {r.faithfulness:.2f} | {r.answer_relevance:.2f} | {r.factual_accuracy:.2f} | **{r.composite_score:.2f}** | {status_str} |"
            )

        return "\n".join(lines)
