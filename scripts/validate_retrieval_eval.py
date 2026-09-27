import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
EVAL_PATH = PROJECT_ROOT / "data" / "evaluation" / "retrieval_eval.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dictionaries."""
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main() -> None:
    chunks = load_jsonl(CHUNKS_PATH)
    evaluation_cases = load_jsonl(EVAL_PATH)

    available_chunk_ids = {
        chunk["chunk_id"]
        for chunk in chunks
    }

    errors = []

    for case in evaluation_cases:
        for chunk_id in case["relevant_chunk_ids"]:
            if chunk_id not in available_chunk_ids:
                errors.append(
                    {
                        "query": case["query"],
                        "missing_chunk_id": chunk_id,
                    }
                )

    if errors:
        print("INVALID EVALUATION DATASET")

        for error in errors:
            print(
                f'- Missing chunk: {error["missing_chunk_id"]}\n'
                f'  Query: {error["query"]}'
            )

        raise SystemExit(1)

    print(
        f"Evaluation dataset valid: "
        f"{len(evaluation_cases)} queries, "
        f"all relevant chunk IDs exist."
    )


if __name__ == "__main__":
    main()