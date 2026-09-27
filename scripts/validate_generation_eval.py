import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks.jsonl"
)

EVAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "generation_eval.jsonl"
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def main() -> None:
    chunks = load_jsonl(CHUNKS_PATH)
    evaluation_cases = load_jsonl(EVAL_PATH)

    available_chunk_ids = {
        chunk["chunk_id"]
        for chunk in chunks
    }

    errors = []

    for index, case in enumerate(evaluation_cases, start=1):
        query = case.get("query")
        relevant_ids = case.get("relevant_chunk_ids", [])
        expected_claims = case.get("expected_claims", [])

        if not query:
            errors.append(f"Case {index}: missing query.")

        if not relevant_ids:
            errors.append(
                f"Case {index}: no relevant_chunk_ids."
            )

        if not expected_claims:
            errors.append(
                f"Case {index}: no expected_claims."
            )

        for chunk_id in relevant_ids:
            if chunk_id not in available_chunk_ids:
                errors.append(
                    f"Case {index}: missing chunk_id: {chunk_id}"
                )

    if errors:
        print("INVALID GENERATION EVALUATION DATASET")

        for error in errors:
            print(f"- {error}")

        raise SystemExit(1)

    print(
        f"Generation evaluation dataset valid: "
        f"{len(evaluation_cases)} cases."
    )
    print(
        "All relevant chunk IDs exist and "
        "all cases contain expected claims."
    )


if __name__ == "__main__":
    main()