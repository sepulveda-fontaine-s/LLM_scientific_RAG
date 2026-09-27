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

TARGET_QUERIES = {
    (
        "Why does Sentence-BERT lose performance when evaluated "
        "on argument similarity topics not seen during training?"
    ),
    (
        "What trade-offs arise when choosing chunk size "
        "in a RAG system?"
    ),
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def main() -> None:
    cases = load_jsonl(EVAL_PATH)

    selected_cases = [
        case
        for case in cases
        if case["query"] in TARGET_QUERIES
    ]

    if len(selected_cases) != 2:
        raise ValueError(
            f"Expected 2 smoke-test cases, "
            f"found {len(selected_cases)}."
        )

    pipeline = RAGPipeline(
        rrf_k=60,
        candidate_k=25,
        rerank_k=5,
    )

    for index, case in enumerate(
        selected_cases,
        start=1,
    ):
        result = pipeline.answer(
            query=case["query"],
        )

        print("\n" + "=" * 100)
        print(f"SMOKE CASE {index}")
        print("=" * 100)

        print("QUERY:")
        print(result["query"])

        print("\nCANDIDATE ANSWER:")
        print(result["answer_before_fail_closed"])

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
            f"Hard guardrails passed: "
            f"{result['hard_guardrails_passed']}"
        )
        print(
            f"Status: "
            f"{result['validation_status']}"
        )


if __name__ == "__main__":
    main()