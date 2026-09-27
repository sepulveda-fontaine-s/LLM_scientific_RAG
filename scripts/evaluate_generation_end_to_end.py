import json
from pathlib import Path

from llm_rag.rag_pipeline import RAGPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "generation_eval.jsonl"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "generation_end_to_end_results.json"
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def main() -> None:
    evaluation_cases = load_jsonl(EVAL_PATH)

    pipeline = RAGPipeline(
        rrf_k=60,
        candidate_k=25,
        rerank_k=5,
    )

    results = []

    citation_passes = 0
    numeric_passes = 0
    grounded_passes = 0
    hard_guardrail_passes = 0

    total_retrieval_recall = 0.0

    status_counts = {
        "validated": 0,
        "validated_with_soft_warning": 0,
        "rejected": 0,
    }

    for index, case in enumerate(
        evaluation_cases,
        start=1,
    ):
        result = pipeline.answer(
            query=case["query"],
        )

        retrieved_chunk_ids = [
            source["chunk_id"]
            for source in result["sources"]
        ]

        relevant_ids = set(
            case["relevant_chunk_ids"]
        )

        retrieved_relevant = (
            set(retrieved_chunk_ids)
            & relevant_ids
        )

        retrieval_recall = (
            len(retrieved_relevant)
            / len(relevant_ids)
            if relevant_ids
            else 0.0
        )

        total_retrieval_recall += retrieval_recall

        citation_passes += int(
            result["citations_valid"]
        )

        numeric_passes += int(
            result["numeric_grounding_valid"]
        )

        grounded_passes += int(
            result["grounded"]
        )

        hard_guardrail_passes += int(
            result["hard_guardrails_passed"]
        )

        status_counts[
            result["validation_status"]
        ] += 1

        case_result = {
            "query": case["query"],
            "expected_claims": (
                case["expected_claims"]
            ),
            "oracle_chunk_ids": (
                case["relevant_chunk_ids"]
            ),
            "retrieved_chunk_ids": (
                retrieved_chunk_ids
            ),
            "retrieved_relevant_chunk_ids": (
                sorted(retrieved_relevant)
            ),
            "retrieval_recall_at_5": (
                retrieval_recall
            ),
            "answer": result["answer"],
            "answer_before_fail_closed": (
                result[
                    "answer_before_fail_closed"
                ]
            ),
            "citations_valid": (
                result["citations_valid"]
            ),
            "numeric_grounding_valid": (
                result[
                    "numeric_grounding_valid"
                ]
            ),
            "grounded": (
                result["grounded"]
            ),
            "hard_guardrails_passed": (
                result[
                    "hard_guardrails_passed"
                ]
            ),
            "validation_status": (
                result["validation_status"]
            ),
            "groundedness_details": (
                result[
                    "groundedness_details"
                ]
            ),
        }

        results.append(case_result)

        print("\n" + "=" * 100)
        print(f"CASE {index}")
        print("=" * 100)

        print("QUERY:")
        print(case["query"])

        print(
            f"\nRetrieval recall@5: "
            f"{retrieval_recall:.4f}"
        )

        print(
            f"Relevant chunks retrieved: "
            f"{len(retrieved_relevant)}/"
            f"{len(relevant_ids)}"
        )

        print("\nANSWER:")
        print(result["answer"])

        print("\nVALIDATION:")
        print(
            f"Citations valid: "
            f"{result['citations_valid']}"
        )
        print(
            f"Numeric grounding valid: "
            f"{result['numeric_grounding_valid']}"
        )
        print(
            f"Grounded: "
            f"{result['grounded']}"
        )
        print(
            f"Hard guardrails passed: "
            f"{result['hard_guardrails_passed']}"
        )
        print(
            f"Status: "
            f"{result['validation_status']}"
        )

    total = len(evaluation_cases)

    aggregate = {
        "number_of_cases": total,
        "mean_retrieval_recall_at_5": (
            total_retrieval_recall / total
        ),
        "citation_valid_rate": (
            citation_passes / total
        ),
        "numeric_grounding_valid_rate": (
            numeric_passes / total
        ),
        "grounded_rate": (
            grounded_passes / total
        ),
        "hard_guardrail_pass_rate": (
            hard_guardrail_passes / total
        ),
        "validation_status_counts": (
            status_counts
        ),
    }

    print("\n" + "=" * 100)
    print(
        "END-TO-END GENERATION AGGREGATE RESULTS"
    )
    print("=" * 100)

    print(
        f"Mean retrieval Recall@5: "
        f"{aggregate['mean_retrieval_recall_at_5']:.4f}"
    )

    print(
        f"Citation valid rate: "
        f"{aggregate['citation_valid_rate']:.4f}"
    )

    print(
        f"Numeric grounding valid rate: "
        f"{aggregate['numeric_grounding_valid_rate']:.4f}"
    )

    print(
        f"Grounded rate: "
        f"{aggregate['grounded_rate']:.4f}"
    )

    print(
        f"Hard guardrail pass rate: "
        f"{aggregate['hard_guardrail_pass_rate']:.4f}"
    )

    print(
        f"Validation statuses: "
        f"{aggregate['validation_status_counts']}"
    )

    output = {
        "mode": "end_to_end",
        "aggregate": aggregate,
        "cases": results,
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
        f"\nResults saved to: "
        f"{RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()