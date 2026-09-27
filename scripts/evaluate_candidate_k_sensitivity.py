import json
from pathlib import Path

from llm_rag.retrieval_metrics import (
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from llm_rag.retrieval_pipeline import RetrievalPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval_eval.jsonl"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "candidate_k_sensitivity.json"
)

CANDIDATE_K_VALUES = [10, 15, 25, 50]
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

    # Load retrieval and reranking models only once.
    pipeline = RetrievalPipeline(
        rrf_k=60,
        candidate_k=max(CANDIDATE_K_VALUES),
        rerank_k=FINAL_K,
    )

    all_results = {}

    print(
        f"{'candidate_k':>12s} "
        f"{'Candidate Recall':>18s} "
        f"{'Final Recall@5':>16s} "
        f"{'MRR':>10s} "
        f"{'nDCG@5':>10s}"
    )

    print("-" * 72)

    for candidate_k in CANDIDATE_K_VALUES:
        candidate_recalls = []
        final_recalls = []
        reciprocal_ranks = []
        ndcgs = []

        query_results = []

        for case in evaluation_cases:
            query = case["query"]
            relevant_ids = set(
                case["relevant_chunk_ids"]
            )

            # Hybrid retrieval candidate pool.
            candidates = (
                pipeline
                .hybrid_retriever
                .retrieve(
                    query=query,
                    top_k=candidate_k,
                    candidate_k=candidate_k,
                )
            )

            candidate_ids = [
                result["chunk_id"]
                for result in candidates
            ]

            # How much annotated evidence reaches the reranker?
            candidate_recall = recall_at_k(
                candidate_ids,
                relevant_ids,
                k=candidate_k,
            )

            # Cross-encoder produces the final top-5.
            reranked = pipeline.reranker.rerank(
                query=query,
                candidates=candidates,
                top_k=FINAL_K,
            )

            final_ids = [
                result["chunk_id"]
                for result in reranked
            ]

            final_recall = recall_at_k(
                final_ids,
                relevant_ids,
                k=FINAL_K,
            )

            rr = reciprocal_rank(
                final_ids,
                relevant_ids,
            )

            ndcg = ndcg_at_k(
                final_ids,
                relevant_ids,
                k=FINAL_K,
            )

            candidate_recalls.append(
                candidate_recall
            )

            final_recalls.append(
                final_recall
            )

            reciprocal_ranks.append(rr)
            ndcgs.append(ndcg)

            query_results.append(
                {
                    "query": query,
                    "candidate_recall": candidate_recall,
                    "final_recall_at_5": final_recall,
                    "reciprocal_rank": rr,
                    "ndcg_at_5": ndcg,
                    "candidate_chunk_ids": candidate_ids,
                    "final_chunk_ids": final_ids,
                }
            )

        count = len(evaluation_cases)

        mean_candidate_recall = (
            sum(candidate_recalls) / count
        )

        mean_final_recall = (
            sum(final_recalls) / count
        )

        mean_mrr = (
            sum(reciprocal_ranks) / count
        )

        mean_ndcg = (
            sum(ndcgs) / count
        )

        all_results[str(candidate_k)] = {
            "mean_candidate_recall": (
                mean_candidate_recall
            ),
            "mean_final_recall_at_5": (
                mean_final_recall
            ),
            "mrr": mean_mrr,
            "mean_ndcg_at_5": mean_ndcg,
            "queries": query_results,
        }

        print(
            f"{candidate_k:12d} "
            f"{mean_candidate_recall:18.4f} "
            f"{mean_final_recall:16.4f} "
            f"{mean_mrr:10.4f} "
            f"{mean_ndcg:10.4f}"
        )

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "number_of_queries": len(evaluation_cases),
        "candidate_k_values": CANDIDATE_K_VALUES,
        "final_k": FINAL_K,
        "results": all_results,
    }

    with RESULTS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\nDetailed results saved to: "
        f"{RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()