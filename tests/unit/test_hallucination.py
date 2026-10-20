"""Unit tests for hallucination detection."""

from historyverse.services.evaluation.hallucination import HallucinationDetector


def test_grounded_answer_low_hallucination_score():
    detector = HallucinationDetector()
    context = [
        "The Serapeum of Alexandria was a monumental temple dedicated to Serapis, created during Ptolemy III."
    ]
    query = "Who was the Serapeum dedicated to?"
    answer = "The Serapeum of Alexandria was dedicated to the deity Serapis."

    score = detector.score(query, answer, context)
    assert score < 0.40
    assert not detector.is_hallucinated(query, answer, context, threshold=0.50)


def test_hallucinated_answer_high_score():
    detector = HallucinationDetector()
    context = ["The Greco-Roman museum displays artifacts from the 3rd century BC."]
    query = "What does the museum display?"
    answer = "The facility produces supersonic space rockets and quantum computer chips."

    score = detector.score(query, answer, context)
    assert score > 0.60
    assert detector.is_hallucinated(query, answer, context, threshold=0.50)
