"""Unit tests for token context management and Pydantic structured output."""

import pytest

from agent.core.schemas import StructuredHistoricalAnalysis
from tests.conftest import MockLLMProvider


def test_token_counting_and_context_budget():
    provider = MockLLMProvider("mock-gpt-4o")

    text = "The Library of Alexandria was one of the largest and most significant libraries of the ancient world."
    count = provider.count_tokens(text)
    assert count > 0

    window = provider.get_context_window()
    assert window >= 4096

    # Test fitting to context budget
    short_text = provider.fit_to_context_budget(text, max_tokens=100)
    assert short_text == text

    truncated_text = provider.fit_to_context_budget(text, max_tokens=3)
    assert "[Context truncated" in truncated_text


@pytest.mark.asyncio
async def test_structured_output_generation():
    provider = MockLLMProvider()

    prompt = "Analyze the Greco-Roman Museum of Alexandria"
    structured_result = await provider.generate_structured(prompt, StructuredHistoricalAnalysis)

    assert isinstance(structured_result, StructuredHistoricalAnalysis)
    assert structured_result.topic == "Alexandria Museum"
    assert structured_result.historical_period == "Greco-Roman"
    assert len(structured_result.key_entities) > 0
    assert structured_result.confidence_score >= 0.9
