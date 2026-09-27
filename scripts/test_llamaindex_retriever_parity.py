import json
from pathlib import Path

from llama_index.core.schema import QueryBundle

from llm_rag.llamaindex_retriever import (
    LlamaIndexHybridRetriever,
)
from llm_rag.retrieval_pipeline import (
    RetrievalPipeline,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval_eval.jsonl"
)

SCORE_TOLERANCE = 1e-6


def load_queries(path: Path) -> list[str]:
    """Load development queries."""

    queries = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                item = json.loads(line)
                queries.append(item["query"])

    return queries


def main() -> None:
    queries = load_queries(EVAL_FILE)

    manual_retriever = RetrievalPipeline(
        rrf_k=60,
        candidate_k=25,
        rerank_k=5,
    )

    llama_retriever = LlamaIndexHybridRetriever(
        rrf_k=60,
        candidate_k=25,
        rerank_k=5,
    )

    failed_queries = 0
    max_score_difference = 0.0

    for index, query in enumerate(
        queries,
        start=1,
    ):
        manual_results = manual_retriever.retrieve(
            query=query,
        )

        llama_results = llama_retriever.retrieve(
            QueryBundle(
                query_str=query,
            )
        )

        manual_ids = [
            result["chunk_id"]
            for result in manual_results
        ]

        llama_ids = [
            result.node.node_id
            for result in llama_results
        ]

        same_ranking = (
            manual_ids == llama_ids
        )

        score_differences = []

        for manual_result, llama_result in zip(
            manual_results,
            llama_results,
        ):
            manual_score = manual_result.get(
                "reranker_score",
                manual_result.get("score"),
            )

            llama_score = llama_result.score

            difference = abs(
                manual_score - llama_score
            )

            score_differences.append(
                difference
            )

        query_max_difference = max(
            score_differences,
            default=0.0,
        )

        max_score_difference = max(
            max_score_difference,
            query_max_difference,
        )

        scores_match = (
            query_max_difference
            <= SCORE_TOLERANCE
        )

        passed = (
            same_ranking
            and scores_match
        )

        if not passed:
            failed_queries += 1

        print(
            f"{index:02d}. "
            f"ranking={same_ranking} "
            f"scores={scores_match} "
            f"max_diff="
            f"{query_max_difference:.10f}"
        )

    print("\n" + "=" * 80)
    print("LLAMAINDEX RETRIEVER PARITY TEST")
    print("=" * 80)

    print(
        f"Queries tested: {len(queries)}"
    )

    print(
        f"Failed queries: {failed_queries}"
    )

    print(
        "Maximum score difference: "
        f"{max_score_difference:.10f}"
    )

    if failed_queries:
        raise AssertionError(
            "LlamaIndex adapter changed "
            "retrieval behavior."
        )

    print("RESULT: PASS")
    print(
        "LlamaIndex adapter preserves "
        "the manual retrieval ranking."
    )


if __name__ == "__main__":
    main()