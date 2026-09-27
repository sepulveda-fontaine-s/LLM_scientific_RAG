import json
from pathlib import Path

from llm_rag.groundedness_validator import GroundednessValidator


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks.jsonl"
)

TARGET_CHUNK_ID = (
    "Dense Passage Retrieval for Open-Domain Question Answering.pdf"
    "_page_7_chunk_2"
)


def load_target_chunk() -> str:
    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            chunk = json.loads(line)

            if chunk["chunk_id"] == TARGET_CHUNK_ID:
                return chunk["text"]

    raise ValueError("Target chunk not found.")


def main() -> None:
    validator = GroundednessValidator()

    chunk_text = load_target_chunk()

    claim = (
        "Building the FAISS index for dense vectors takes "
        "8.5 hours on a single server, compared to 30 minutes "
        "for an inverted index using Lucene."
    )

    windows = validator._build_premise_windows(
        text=chunk_text,
        window_size=2,
    )

    best_score = -1.0
    best_window = None

    for index, window in enumerate(windows, start=1):
        scores = validator.validate_sentence(
            sentence=claim,
            premise=window,
        )

        print("\n" + "=" * 100)
        print(f"WINDOW {index}")
        print("=" * 100)
        print(window)

        print("\nNLI:")
        print(f"entailment     : {scores['entailment']:.6f}")
        print(f"contradiction  : {scores['contradiction']:.6f}")
        print(f"neutral        : {scores['neutral']:.6f}")

        if scores["entailment"] > best_score:
            best_score = scores["entailment"]
            best_window = window

    print("\n" + "=" * 100)
    print("BEST WINDOW")
    print("=" * 100)
    print(best_window)
    print(f"\nBest entailment: {best_score:.6f}")


if __name__ == "__main__":
    main()