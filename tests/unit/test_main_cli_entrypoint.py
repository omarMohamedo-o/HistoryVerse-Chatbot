"""Unit tests for agent main entrypoint execution and Langfuse client caching."""

import runpy
from unittest.mock import MagicMock, patch

import agent.services.langfuse.client as lf_client


def test_main_entrypoint_execution():
    """Verify executing agent.__main__ module invokes uvicorn.run."""
    with patch("uvicorn.run") as mock_run:
        runpy.run_module("agent.__main__", run_name="__main__")
        mock_run.assert_called_once()


def test_langfuse_client_cached_singleton():
    """Verify get_langfuse_client returns cached singleton instance on repeated calls."""
    mock_instance = MagicMock()
    original = lf_client._client_instance
    try:
        lf_client._client_instance = mock_instance
        retrieved = lf_client.get_langfuse_client()
        assert retrieved is mock_instance
    finally:
        lf_client._client_instance = original
