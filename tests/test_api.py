"""API tests -- Azure calls are stubbed so the suite runs without credentials."""

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.main import app
from app.models import AskResponse

client = TestClient(app)


@pytest.fixture
def configured(monkeypatch):
    from app.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "azure_openai_endpoint", "https://x.openai.azure.com")
    monkeypatch.setattr(settings, "azure_openai_api_key", "k")
    monkeypatch.setattr(settings, "azure_search_endpoint", "https://x.search.windows.net")
    monkeypatch.setattr(settings, "azure_search_api_key", "k")
    return settings


def test_health_reports_deployments():
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["index"]
    assert body["chat_deployment"]


def test_ask_requires_configuration():
    response = client.post("/ask", json={"question": "What is the leave policy?"})
    assert response.status_code == 503


def test_ask_rejects_too_short_question(configured):
    assert client.post("/ask", json={"question": "hi"}).status_code == 422


def test_ask_returns_answer_and_citations(configured, monkeypatch):
    stub = AskResponse(
        answer="Within 30 days [1].",
        citations=[
            {
                "n": 1,
                "source": "expense-policy.md",
                "title": "expense policy",
                "chunk_index": 0,
                "score": 2.1,
                "excerpt": "Claims must be submitted within 30 days.",
            }
        ],
        grounded=True,
        latency_ms=120,
    )
    monkeypatch.setattr(main, "answer_question", lambda *a, **k: stub)

    body = client.post("/ask", json={"question": "How long to submit a claim?"}).json()
    assert body["answer"] == "Within 30 days [1]."
    assert body["grounded"] is True
    assert body["citations"][0]["source"] == "expense-policy.md"


def test_upstream_failure_becomes_502(configured, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("search is down")

    monkeypatch.setattr(main, "answer_question", boom)
    assert client.post("/ask", json={"question": "anything at all"}).status_code == 502
