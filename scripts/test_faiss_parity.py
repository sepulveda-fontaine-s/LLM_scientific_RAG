import json
from pathlib import Path

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


def load_queries(path: Path) -> list[str]:
    """Load evaluation queries from the retrieval development set."""
    queries = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                item = json.loads(line)
                queries.append(item["query"])

    return queries


def numpy_search(
    retriever: DenseRetriever,
    query: str,
    top_k: int,
) -> list[tuple[str, float]]:
    """
    Reproduce the original exhaustive NumPy dense search.
    """

    query_embedding = retriever.embedder.encode(
        [query],
        batch_size=1,
    )[0]

    scores = retriever.embeddings @ query_embedding

    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        (
            retriever.chunks[index]["chunk_id"],
            float(scores[index]),
        )
        for index in top_indices
    ]


def faiss_search(
    retriever: DenseRetriever,
    query: str,
    top_k: int,
) -> list[tuple[str, float]]:
    """Run the new FAISS-based dense retriever."""

    results = retriever.retrieve(
        query=query,
        top_k=top_k,
    )

    return [
        (
            result["chunk_id"],
            result["score"],
        )
        for result in results
    ]


def main() -> None:
    queries = load_queries(EVAL_FILE)
    retriever = DenseRetriever()

    failed_queries = 0
    max_score_difference = 0.0

    for index, query in enumerate(queries, start=1):
        numpy_results = numpy_search(
            retriever,
            query,
            TOP_K,
        )

        faiss_results = faiss_search(
            retriever,
            query,
            TOP_K,
        )

        numpy_ids = [
            chunk_id
            for chunk_id, _ in numpy_results
        ]

        faiss_ids = [
            chunk_id
            for chunk_id, _ in faiss_results
        ]

        same_ranking = numpy_ids == faiss_ids

        score_differences = [
            abs(numpy_score - faiss_score)
            for (_, numpy_score), (_, faiss_score)
            in zip(numpy_results, faiss_results)
        ]

        query_max_difference = max(
            score_differences,
            default=0.0,
        )

        max_score_difference = max(
            max_score_difference,
            query_max_difference,
        )

        scores_match = (
            query_max_difference <= SCORE_TOLERANCE
        )

        passed = same_ranking and scores_match

        if not passed:
            failed_queries += 1

        print(
            f"{index:02d}. "
            f"ranking={same_ranking} "
            f"scores={scores_match} "
            f"max_diff={query_max_difference:.10f}"
        )

    print("\n" + "=" * 80)
    print("FAISS PARITY TEST")
    print("=" * 80)
    print(f"Queries tested: {len(queries)}")
    print(f"Failed queries: {failed_queries}")
    print(
        f"Maximum score difference: "
        f"{max_score_difference:.10f}"
    )

    if failed_queries:
        raise AssertionError(
            "FAISS does not match the NumPy reference."
        )

    print("RESULT: PASS")
    print(
        "FAISS IndexFlatIP reproduces the "
        "original NumPy dense ranking."
    )


if __name__ == "__main__":
    main()