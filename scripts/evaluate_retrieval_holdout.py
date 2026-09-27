import json
from pathlib import Path

from llm_rag.retrieval_metrics import (
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from llm_rag.retrieval_pipeline import RetrievalPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]

HOLDOUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval_holdout.jsonl"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "retrieval_holdout_results.json"
)

TOP_K = 5
CANDIDATE_K = 25


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def compute_metrics(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> dict:
    return {
        "recall_at_5": recall_at_k(
            retrieved_ids,
            relevant_ids,
            k=TOP_K,
        ),
        "reciprocal_rank": reciprocal_rank(
            retrieved_ids,
            relevant_ids,
        ),
        "ndcg_at_5": ndcg_at_k(
            retrieved_ids,
            relevant_ids,
            k=TOP_K,
        ),
    }


def main() -> None:
    evaluation_cases = load_jsonl(HOLDOUT_PATH)

    pipeline = RetrievalPipeline(
        rrf_k=60,
        candidate_k=CANDIDATE_K,
        rerank_k=TOP_K,
    )

    dense_retriever = (
        pipeline
        .hybrid_retriever
        .dense_retriever
    )

    sparse_retriever = (
        pipeline
        .hybrid_retriever
        .sparse_retriever
    )

    hybrid_retriever = pipeline.hybrid_retriever

    systems = {
        "dense": [],
        "bm25": [],
        "hybrid_rrf": [],
        "hybrid_rrf_reranker": [],
    }

    detailed_results = []

    for case in evaluation_cases:
        query = case["query"]
        relevant_ids = set(case["relevant_chunk_ids"])

        dense_results = dense_retriever.retrieve(
            query=query,
            top_k=TOP_K,
        )

        bm25_results = sparse_retriever.retrieve(
            query=query,
            top_k=TOP_K,
        )

        hybrid_results = hybrid_retriever.retrieve(
            query=query,
            top_k=TOP_K,
            candidate_k=CANDIDATE_K,
        )

        reranked_results = pipeline.retrieve(query)

        result_sets = {
            "dense": dense_results,
            "bm25": bm25_results,
            "hybrid_rrf": hybrid_results,
            "hybrid_rrf_reranker": reranked_results,
        }

        query_result = {
            "query": query,
            "systems": {},
        }

        print("\n" + "=" * 100)
        print("QUERY:", query)

        for system_name, results in result_sets.items():
            retrieved_ids = [
                result["chunk_id"]
                for result in results
            ]

            metrics = compute_metrics(
                retrieved_ids,
                relevant_ids,
            )

            systems[system_name].append(metrics)

            query_result["systems"][system_name] = {
                "metrics": metrics,
                "retrieved_chunk_ids": retrieved_ids,
            }

            print(
                f"{system_name:22s} "
                f"Recall@5={metrics['recall_at_5']:.4f}  "
                f"RR={metrics['reciprocal_rank']:.4f}  "
                f"nDCG@5={metrics['ndcg_at_5']:.4f}"
            )

        detailed_results.append(query_result)

    aggregate_results = {}

    print("\n" + "=" * 100)
    print("FINAL HOLDOUT RESULTS")
    print("=" * 100)

    print(
        f"{'System':22s} "
        f"{'Recall@5':>10s} "
        f"{'MRR':>10s} "
        f"{'nDCG@5':>10s}"
    )

    for system_name, metric_list in systems.items():
        count = len(metric_list)

        mean_recall = sum(
            item["recall_at_5"]
            for item in metric_list
        ) / count

        mean_mrr = sum(
            item["reciprocal_rank"]
            for item in metric_list
        ) / count

        mean_ndcg = sum(
            item["ndcg_at_5"]
            for item in metric_list
        ) / count

        aggregate_results[system_name] = {
            "mean_recall_at_5": mean_recall,
            "mrr": mean_mrr,
            "mean_ndcg_at_5": mean_ndcg,
        }

        print(
            f"{system_name:22s} "
            f"{mean_recall:10.4f} "
            f"{mean_mrr:10.4f} "
            f"{mean_ndcg:10.4f}"
        )

    output = {
        "dataset": "frozen_holdout",
        "number_of_queries": len(evaluation_cases),
        "top_k": TOP_K,
        "candidate_k": CANDIDATE_K,
        "rrf_k": 60,
        "aggregate": aggregate_results,
        "queries": detailed_results,
    }

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
        f"\nFinal holdout results saved to: "
        f"{RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()