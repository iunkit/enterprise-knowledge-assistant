"""Pipeline behaviour with retrieval and generation stubbed out."""

from app import rag
from app.prompts import NO_ANSWER


def test_empty_retrieval_short_circuits(monkeypatch):
    monkeypatch.setattr(rag, "retrieve", lambda *a, **k: [])

    def should_not_run():
        raise AssertionError("the model must not be called without context")

    monkeypatch.setattr(rag, "get_openai_client", lambda: should_not_run())

    response = rag.answer_question("What is the capital of France?")
    assert response.answer == NO_ANSWER
    assert response.grounded is False
    assert response.citations == []


def test_citations_mirror_retrieved_chunks(monkeypatch):
    chunks = [
        {
            "content": "Alcohol is not reimbursable." * 40,
            "source": "expense-policy.md",
            "title": "expense policy",
            "chunk_index": 3,
            "score": 2.44444,
        }
    ]
    monkeypatch.setattr(rag, "retrieve", lambda *a, **k: chunks)
    monkeypatch.setattr(rag, "get_openai_client", lambda: _FakeClient("No [1]."))

    response = rag.answer_question("Can I expense wine?")
    citation = response.citations[0]
    assert citation.n == 1
    assert citation.chunk_index == 3
    assert citation.score == 2.4444
    assert len(citation.excerpt) <= rag.EXCERPT_CHARS
    assert response.grounded is True


def test_model_refusal_marks_answer_ungrounded(monkeypatch):
    monkeypatch.setattr(
        rag,
        "retrieve",
        lambda *a, **k: [
            {
                "content": "unrelated text",
                "source": "a.md",
                "title": "a",
                "chunk_index": 0,
                "score": 1.0,
            }
        ],
    )
    monkeypatch.setattr(rag, "get_openai_client", lambda: _FakeClient(NO_ANSWER))

    assert rag.answer_question("Who is the CEO?").grounded is False


class _FakeClient:
    def __init__(self, content: str):
        self.chat = _FakeChat(content)


class _FakeChat:
    def __init__(self, content: str):
        self.completions = _FakeCompletions(content)


class _FakeCompletions:
    def __init__(self, content: str):
        self._content = content

    def create(self, **_kwargs):
        message = type("M", (), {"content": self._content})
        choice = type("C", (), {"message": message})
        return type("R", (), {"choices": [choice]})
