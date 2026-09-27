import json
from pathlib import Path

import numpy as np

from llm_rag.llamaindex_query_engine import (
    LlamaIndexRAGQueryEngine,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "generation_eval.jsonl"
)

RESULTS_FILE = (
    PROJECT_ROOT
    / "results"
    / "llamaindex_end_to_end_results.json"
)


def load_cases(path: Path) -> list[dict]:
    """Load generation evaluation cases."""

    cases = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                cases.append(json.loads(line))

    return cases


def main() -> None:
    cases = load_cases(EVAL_FILE)

    engine = LlamaIndexRAGQueryEngine.from_defaults(
        rrf_k=60,
        candidate_k=25,
        rerank_k=5,
    )

    results = []

    retrieval_recalls = []
    citation_valid_values = []
    numeric_valid_values = []
    grounded_values = []
    hard_guardrail_values = []

    status_counts = {
        "validated": 0,
        "validated_with_soft_warning": 0,
        "rejected": 0,
    }

    for index, case in enumerate(
        cases,
        start=1,
    ):
        query = case["query"]

        response = engine.query(query)

        retrieved_chunk_ids = [
            node.node.metadata["chunk_id"]
            for node in response.source_nodes
        ]

        # Ground-truth relevant chunks from generation_eval.jsonl.
        relevant_chunk_ids = case["relevant_chunk_ids"]

        retrieved_relevant = [
            chunk_id
            for chunk_id in relevant_chunk_ids
            if chunk_id in retrieved_chunk_ids
        ]

        recall_at_5 = (
            len(retrieved_relevant)
            / len(relevant_chunk_ids)
            if relevant_chunk_ids
            else 0.0
        )

        metadata = response.metadata

        status = metadata["validation_status"]

        status_counts[status] += 1

        retrieval_recalls.append(recall_at_5)
        citation_valid_values.append(
            metadata["citations_valid"]
        )
        numeric_valid_values.append(
            metadata["numeric_grounding_valid"]
        )
        grounded_values.append(
            metadata["grounded"]
        )
        hard_guardrail_values.append(
            metadata["hard_guardrails_passed"]
        )

        result = {
            "query": query,
            "expected_claims": case["expected_claims"],
            "relevant_chunk_ids": relevant_chunk_ids,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "retrieved_relevant_chunk_ids": retrieved_relevant,
            "retrieval_recall_at_5": recall_at_5,
            "answer": str(response),
            "answer_before_fail_closed": metadata[
                "answer_before_fail_closed"
            ],
            "citations_valid": metadata[
                "citations_valid"
            ],
            "numeric_grounding_valid": metadata[
                "numeric_grounding_valid"
            ],
            "grounded": metadata["grounded"],
            "hard_guardrails_passed": metadata[
                "hard_guardrails_passed"
            ],
            "validation_status": status,
        }

        results.append(result)

        print(
            f"{index}/{len(cases)} "
            f"recall={recall_at_5:.4f} "
            f"status={status}"
        )

    aggregate = {
        "number_of_cases": len(cases),
        "mean_retrieval_recall_at_5": float(
            np.mean(retrieval_recalls)
        ),
        "citation_valid_rate": float(
            np.mean(citation_valid_values)
        ),
        "numeric_grounding_valid_rate": float(
            np.mean(numeric_valid_values)
        ),
        "grounded_rate": float(
            np.mean(grounded_values)
        ),
        "hard_guardrail_pass_rate": float(
            np.mean(hard_guardrail_values)
        ),
        "validation_status_counts": status_counts,
    }

    output = {
        "mode": "llamaindex_end_to_end",
        "aggregate": aggregate,
        "cases": results,
    }

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_FILE.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print("LLAMAINDEX END-TO-END AGGREGATE RESULTS")
    print("=" * 100)

    print(
        "Mean retrieval Recall@5:",
        f"{aggregate['mean_retrieval_recall_at_5']:.4f}",
    )

    print(
        "Citation valid rate:",
        f"{aggregate['citation_valid_rate']:.4f}",
    )

    print(
        "Numeric grounding valid rate:",
        f"{aggregate['numeric_grounding_valid_rate']:.4f}",
    )

    print(
        "Grounded rate:",
        f"{aggregate['grounded_rate']:.4f}",
    )

    print(
        "Hard guardrail pass rate:",
        f"{aggregate['hard_guardrail_pass_rate']:.4f}",
    )

    print(
        "Validation statuses:",
        aggregate["validation_status_counts"],
    )


if __name__ == "__main__":
    main()
