import json
from collections import defaultdict
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
    / "reranker_rank_fusion.json"
)

CANDIDATE_K = 15
TOP_K = 5

# RRF constant used for the second-stage fusion.
FUSION_K = 60


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


def fuse_rankings(
    hybrid_results: list[dict],
    reranked_results: list[dict],
) -> list[dict]:
    """
    Fuse the original hybrid ranking and the cross-encoder
    ranking using Reciprocal Rank Fusion.

    This prevents the cross-encoder from completely replacing
    the retrieval ranking.
    """

    scores = defaultdict(float)
    chunks_by_id = {}

    for rank, result in enumerate(
        hybrid_results,
        start=1,
    ):
        chunk_id = result["chunk_id"]

        scores[chunk_id] += (
            1.0 / (FUSION_K + rank)
        )

        chunks_by_id[chunk_id] = result

    for rank, result in enumerate(
        reranked_results,
        start=1,
    ):
        chunk_id = result["chunk_id"]

        scores[chunk_id] += (
            1.0 / (FUSION_K + rank)
        )

        chunks_by_id[chunk_id] = result

    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )

    fused_results = []

    for chunk_id in ranked_ids:
        result = chunks_by_id[chunk_id].copy()

        result["rank_fusion_score"] = (
            scores[chunk_id]
        )

        fused_results.append(result)

    return fused_results


def main() -> None:
    evaluation_cases = load_jsonl(EVAL_PATH)

    pipeline = RetrievalPipeline(
        rrf_k=60,
        candidate_k=CANDIDATE_K,
        rerank_k=TOP_K,
    )

    systems = {
        "hybrid_rrf": [],
        "cross_encoder": [],
        "rank_fusion": [],
    }

    detailed_results = []

    for case in evaluation_cases:
        query = case["query"]
        relevant_ids = set(
            case["relevant_chunk_ids"]
        )

        # Retrieve 15 candidates using hybrid retrieval.
        hybrid_candidates = (
            pipeline
            .hybrid_retriever
            .retrieve(
                query=query,
                top_k=CANDIDATE_K,
                candidate_k=CANDIDATE_K,
            )
        )

        # Re-rank all 15 candidates, not only the top 5.
        reranked_candidates = (
            pipeline
            .reranker
            .rerank(
                query=query,
                candidates=hybrid_candidates,
                top_k=CANDIDATE_K,
            )
        )

        fused_candidates = fuse_rankings(
            hybrid_results=hybrid_candidates,
            reranked_results=reranked_candidates,
        )

        result_sets = {
            "hybrid_rrf": hybrid_candidates[:TOP_K],
            "cross_encoder": reranked_candidates[:TOP_K],
            "rank_fusion": fused_candidates[:TOP_K],
        }

        print("\n" + "=" * 100)
        print("QUERY:", query)

        query_result = {
            "query": query,
            "systems": {},
        }

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
                f"{system_name:16s} "
                f"Recall@5={metrics['recall_at_5']:.4f}  "
                f"RR={metrics['reciprocal_rank']:.4f}  "
                f"nDCG@5={metrics['ndcg_at_5']:.4f}"
            )

        detailed_results.append(query_result)

    aggregate_results = {}

    print("\n" + "=" * 100)
    print("AGGREGATE RANK-FUSION RESULTS")
    print("=" * 100)

    print(
        f"{'System':16s} "
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
            f"{system_name:16s} "
            f"{mean_recall:10.4f} "
            f"{mean_mrr:10.4f} "
            f"{mean_ndcg:10.4f}"
        )

    output = {
        "number_of_queries": len(evaluation_cases),
        "candidate_k": CANDIDATE_K,
        "top_k": TOP_K,
        "fusion_k": FUSION_K,
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
        f"\nDetailed results saved to: "
        f"{RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()