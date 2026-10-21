"""Unit tests for Langfuse tracing and telemetry decorators."""

from unittest.mock import MagicMock, patch

import pytest

from agent.services.langfuse.traces import trace_execution, traced


def test_traced_decorator_sync():
    mock_client = MagicMock()
    mock_trace = MagicMock()
    mock_client.trace.return_value = mock_trace

    with patch("agent.services.langfuse.traces.get_langfuse_client", return_value=mock_client):

        @traced(name="test_sync_op")
        def sync_function(a: int, b: int) -> int:
            return a + b

        result = sync_function(2, 3)
        assert result == 5


def test_traced_decorator_sync_error():
    mock_client = MagicMock()
    with patch("agent.services.langfuse.traces.get_langfuse_client", return_value=mock_client):

        @traced(name="test_sync_error")
        def failing_sync():
            raise ValueError("Sync error triggered")

        with pytest.raises(ValueError, match="Sync error triggered"):
            failing_sync()


@pytest.mark.asyncio
async def test_traced_decorator_async():
    mock_client = MagicMock()
    with patch("agent.services.langfuse.traces.get_langfuse_client", return_value=mock_client):

        @traced(name="test_async_op")
        async def async_function(x: str) -> str:
            return f"Echo: {x}"

        result = await async_function("Cairo")
        assert result == "Echo: Cairo"
        mock_client.trace.assert_called_with(name="test_async_op")


@pytest.mark.asyncio
async def test_traced_decorator_async_trace_failure():
    mock_client = MagicMock()
    mock_client.trace.side_effect = Exception("Telemetry connection lost")

    with patch("agent.services.langfuse.traces.get_langfuse_client", return_value=mock_client):

        @traced()
        async def sample_coro() -> int:
            return 42

        # Tracing error is safely caught and does not block execution
        res = await sample_coro()
        assert res == 42


@pytest.mark.asyncio
async def test_traced_decorator_async_function_error():
    mock_client = MagicMock()
    with patch("agent.services.langfuse.traces.get_langfuse_client", return_value=mock_client):

        @traced()
        async def failing_async():
            raise RuntimeError("Async explosion")

        with pytest.raises(RuntimeError, match="Async explosion"):
            await failing_async()


def test_trace_execution_helpers():
    # When no client is available
    with patch("agent.services.langfuse.traces.get_langfuse_client", return_value=None):
        trace_id = trace_execution("test_op", {"in": 1}, {"out": 2})
        assert trace_id is None

    # When client is active
    mock_client = MagicMock()
    mock_trace = MagicMock()
    mock_trace.id = "trace_abc123"
    mock_client.trace.return_value = mock_trace

    with patch("agent.services.langfuse.traces.get_langfuse_client", return_value=mock_client):
        trace_id = trace_execution("test_op", {"in": 1}, {"out": 2}, {"env": "test"})
        assert trace_id == "trace_abc123"

    # When client.trace raises exception
    mock_client.trace.side_effect = Exception("Trace error")
    with patch("agent.services.langfuse.traces.get_langfuse_client", return_value=mock_client):
        trace_id = trace_execution("test_op", {"in": 1}, {"out": 2})
        assert trace_id is None
