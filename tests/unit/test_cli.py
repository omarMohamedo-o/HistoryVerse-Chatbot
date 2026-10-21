"""Unit tests for the CLI runner entry point."""

from unittest.mock import patch


def test_main_cli_runner():
    with patch("uvicorn.run") as mock_run:
        from agent.__main__ import main

        main()
        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        assert kwargs["factory"] is True
