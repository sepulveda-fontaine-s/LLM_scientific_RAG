from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from llm_rag.api import app


class FakeResponse:
    """Minimal response object matching what the API expects."""

    def __init__(self) -> None:
        self.metadata = {
            "validation_status": "validated",
            "hard_guardrails_passed": True,
            "citations_valid": True,
            "numeric_grounding_valid": True,
            "grounded": True,
            "answer_before_fail_closed": (
                "ColBERT uses late interaction [1]."
            ),
        }

        self.source_nodes = [
            SimpleNamespace(
                node=SimpleNamespace(
                    metadata={
                        "source": "paper.pdf",
                        "page": 7,
                        "chunk_id": "paper_page_7_chunk_1",
                    }
                ),
                score=4.25,
            )
        ]

    def __str__(self) -> str:
        return "ColBERT uses late interaction [1]."


class FakeQueryEngine:
    """Lightweight test double for the real RAG query engine."""

    def query(self, query: str) -> FakeResponse:
        assert query == "How does ColBERT work?"
        return FakeResponse()


def test_health_endpoint() -> None:
    """The service should report readiness after startup."""

    with patch(
        "llm_rag.api.LlamaIndexRAGQueryEngine.from_defaults",
        return_value=FakeQueryEngine(),
    ):
        with TestClient(app) as client:
            response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_endpoint_contract() -> None:
    """The query endpoint should return the expected structured schema."""

    with patch(
        "llm_rag.api.LlamaIndexRAGQueryEngine.from_defaults",
        return_value=FakeQueryEngine(),
    ):
        with TestClient(app) as client:
            response = client.post(
                "/query",
                json={
                    "query": "How does ColBERT work?",
                },
            )

    assert response.status_code == 200

    payload = response.json()

    assert payload["answer"] == (
        "ColBERT uses late interaction [1]."
    )

    assert payload["validation"] == {
        "status": "validated",
        "hard_guardrails_passed": True,
        "citations_valid": True,
        "numeric_grounding_valid": True,
        "grounded": True,
    }

    assert len(payload["sources"]) == 1

    assert payload["sources"][0] == {
        "citation": "[1]",
        "source": "paper.pdf",
        "page": 7,
        "chunk_id": "paper_page_7_chunk_1",
        "score": 4.25,
    }


def test_query_rejects_empty_input() -> None:
    """Pydantic should reject an empty query before the engine runs."""

    with patch(
        "llm_rag.api.LlamaIndexRAGQueryEngine.from_defaults",
        return_value=FakeQueryEngine(),
    ):
        with TestClient(app) as client:
            response = client.post(
                "/query",
                json={"query": ""},
            )

    assert response.status_code == 422
