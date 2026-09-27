import json
from pathlib import Path

from llm_rag.retrieval_metrics import (
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from llm_rag.retrieval_pipeline import RetrievalPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = PROJECT_ROOT / "data" / "evaluation" / "retrieval_eval.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main() -> None:
    evaluation_cases = load_jsonl(EVAL_PATH)

    pipeline = RetrievalPipeline(
        rrf_k=60,
        candidate_k=25,
        rerank_k=5,
    )

    recalls = []
    reciprocal_ranks = []
    ndcgs = []

    for case in evaluation_cases:
        query = case["query"]
        relevant_ids = set(case["relevant_chunk_ids"])

        results = pipeline.retrieve(query)

        retrieved_ids = [
            result["chunk_id"]
            for result in results
        ]

        recall = recall_at_k(
            retrieved_ids,
            relevant_ids,
            k=5,
        )

        rr = reciprocal_rank(
            retrieved_ids,
            relevant_ids,
        )

        ndcg = ndcg_at_k(
            retrieved_ids,
            relevant_ids,
            k=5,
        )

        recalls.append(recall)
        reciprocal_ranks.append(rr)
        ndcgs.append(ndcg)

        print("=" * 80)
        print("Query:", query)
        print(f"Recall@5: {recall:.4f}")
        print(f"Reciprocal Rank: {rr:.4f}")
        print(f"nDCG@5: {ndcg:.4f}")

    print("\n===== AGGREGATE METRICS =====")
    print(f"Mean Recall@5: {sum(recalls) / len(recalls):.4f}")
    print(
        f"MRR: "
        f"{sum(reciprocal_ranks) / len(reciprocal_ranks):.4f}"
    )
    print(f"Mean nDCG@5: {sum(ndcgs) / len(ndcgs):.4f}")


if __name__ == "__main__":
    main()