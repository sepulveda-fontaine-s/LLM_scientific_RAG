import json
from pathlib import Path
import pytest
import numpy as np

from llm_rag.dense_retriever import DenseRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval_eval.jsonl"
)

TOP_K = 10
SCORE_TOLERANCE = 1e-6


def load_queries() -> list[str]:
    """Load retrieval development queries."""

    queries = []

    with EVAL_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                item = json.loads(line)
                queries.append(item["query"])

    return queries


@pytest.mark.integration
def test_persisted_faiss_matches_numpy_reference() -> None:
    """
    Verify that the persisted FAISS IndexFlatIP preserves
    the original exhaustive NumPy ranking and scores.
    """

    retriever = DenseRetriever()

    for query in load_queries():
        query_embedding = retriever.embedder.encode(
            [query],
            batch_size=1,
        )[0]

        numpy_scores = (
            retriever.embeddings
            @ query_embedding
        )

        numpy_indices = np.argsort(
            numpy_scores
        )[::-1][:TOP_K]

        faiss_results = retriever.retrieve(
            query=query,
            top_k=TOP_K,
        )

        numpy_ids = [
            retriever.chunks[index]["chunk_id"]
            for index in numpy_indices
        ]

        faiss_ids = [
            result["chunk_id"]
            for result in faiss_results
        ]

        assert faiss_ids == numpy_ids

        for numpy_index, faiss_result in zip(
            numpy_indices,
            faiss_results,
        ):
            assert abs(
                float(numpy_scores[numpy_index])
                - faiss_result["score"]
            ) <= SCORE_TOLERANCE