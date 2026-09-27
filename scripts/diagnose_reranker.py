import json
from pathlib import Path

from llm_rag.retrieval_pipeline import RetrievalPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval_eval.jsonl"
)

CANDIDATE_K = 15
FINAL_K = 5


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def main() -> None:
    evaluation_cases = load_jsonl(EVAL_PATH)

    pipeline = RetrievalPipeline(
        rrf_k=60,
        candidate_k=CANDIDATE_K,
        rerank_k=FINAL_K,
    )

    for case in evaluation_cases:
        query = case["query"]
        relevant_ids = set(case["relevant_chunk_ids"])

        candidates = pipeline.hybrid_retriever.retrieve(
            query=query,
            top_k=CANDIDATE_K,
            candidate_k=CANDIDATE_K,
        )

        reranked = pipeline.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=FINAL_K,
        )

        candidate_ids = [
            result["chunk_id"]
            for result in candidates
        ]

        hybrid_top5_ids = candidate_ids[:FINAL_K]

        reranked_ids = [
            result["chunk_id"]
            for result in reranked
        ]

        relevant_in_candidates = (
            relevant_ids & set(candidate_ids)
        )

        relevant_in_hybrid_top5 = (
            relevant_ids & set(hybrid_top5_ids)
        )

        relevant_in_reranked_top5 = (
            relevant_ids & set(reranked_ids)
        )

        dropped_by_reranker = (
            relevant_in_candidates
            - relevant_in_reranked_top5
        )

        missing_from_candidates = (
            relevant_ids
            - set(candidate_ids)
        )

        # Only print cases where reranking loses relevant evidence.
        if (
            len(relevant_in_reranked_top5)
            < len(relevant_in_hybrid_top5)
        ):
            print("\n" + "=" * 100)
            print("QUERY:", query)

            print(
                f"Relevant total: "
                f"{len(relevant_ids)}"
            )

            print(
                f"Relevant in hybrid top-{CANDIDATE_K}: "
                f"{len(relevant_in_candidates)}"
            )

            print(
                f"Relevant in hybrid top-{FINAL_K}: "
                f"{len(relevant_in_hybrid_top5)}"
            )

            print(
                f"Relevant after reranking top-{FINAL_K}: "
                f"{len(relevant_in_reranked_top5)}"
            )

            print("\nDropped by reranker:")
            for chunk_id in sorted(dropped_by_reranker):
                print("  ", repr(chunk_id))

            print("\nMissing before reranking:")
            for chunk_id in sorted(missing_from_candidates):
                print("  ", repr(chunk_id))


if __name__ == "__main__":
    main()