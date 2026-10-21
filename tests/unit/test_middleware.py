"""Unit tests for FastAPI request middleware, correlation IDs, and unhandled exception logging."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent.api.middleware import ErrorHandlingMiddleware, RequestLoggingMiddleware
from agent.core.exceptions import HistoryVerseError


def test_request_logging_middleware():
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)

    @app.get("/ping")
    def ping():
        return {"ping": "pong"}

    client = TestClient(app)
    resp = client.get("/ping")
    assert resp.status_code == 200
    assert "X-Request-ID" in resp.headers
    assert resp.json() == {"ping": "pong"}


def test_error_handling_middleware_domain_error():
    app = FastAPI()
    app.add_middleware(ErrorHandlingMiddleware)

    @app.get("/domain-error")
    def domain_error():
        raise HistoryVerseError("A domain level error occurred")

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/domain-error")
    assert resp.status_code == 400
    assert resp.json()["error"] == "AgentError"
    assert "A domain level error occurred" in resp.json()["message"]


def test_error_handling_middleware_unexpected_error():
    app = FastAPI()
    app.add_middleware(ErrorHandlingMiddleware)

    @app.get("/crash")
    def crash():
        raise RuntimeError("Unexpected boom")

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/crash")
    assert resp.status_code == 500
    assert resp.json()["error"] == "InternalServerError"


def test_request_logging_middleware_exception():
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)

    @app.get("/throw")
    def throw():
        raise ValueError("Simulated dispatch failure")

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/throw")
    assert resp.status_code == 500
