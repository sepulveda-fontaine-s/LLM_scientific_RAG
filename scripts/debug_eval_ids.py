import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVAL_PATH = PROJECT_ROOT / "data" / "evaluation" / "retrieval_eval.jsonl"
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main() -> None:
    eval_cases = load_jsonl(EVAL_PATH)
    chunks = load_jsonl(CHUNKS_PATH)

    print("===== EVALUATION IDS =====")
    for chunk_id in eval_cases[0]["relevant_chunk_ids"]:
        print(repr(chunk_id), "length=", len(chunk_id))

    print("\n===== MATCHING REAL IDS =====")
    for chunk in chunks:
        chunk_id = chunk["chunk_id"]

        if "ColBERT" in chunk_id and (
            "page_1_chunk_2" in chunk_id
            or "page_2_chunk_5" in chunk_id
            or "page_7_chunk_5" in chunk_id
        ):
            print(repr(chunk_id), "length=", len(chunk_id))


if __name__ == "__main__":
    main()