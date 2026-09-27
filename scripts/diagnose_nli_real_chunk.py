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


def print_scores(title: str, scores: dict) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    for key, value in scores.items():
        print(f"{key:15s}: {value:.6f}")


def main() -> None:
    validator = GroundednessValidator()

    real_chunk = load_target_chunk()

    original_claim = (
        "Building the FAISS index for dense vectors takes "
        "8.5 hours on a single server, compared to 30 minutes "
        "for an inverted index using Lucene."
    )

    clean_evidence = (
        "Building the FAISS index on 21-million vectors on a "
        "single server takes 8.5 hours. In comparison, building "
        "an inverted index using Lucene takes only about "
        "30 minutes in total."
    )

    print("REAL CHUNK:")
    print(real_chunk)

    scores = validator.validate_sentence(
        sentence=original_claim,
        premise=real_chunk,
    )

    print_scores(
        "TEST 1 — REAL CHUNK",
        scores,
    )

    scores = validator.validate_sentence(
        sentence=original_claim,
        premise=clean_evidence,
    )

    print_scores(
        "TEST 2 — CLEAN SUPPORTING SPAN",
        scores,
    )

    # Simpler atomic version of the first fact.
    atomic_claim = (
        "Building the FAISS index on a single server "
        "takes 8.5 hours."
    )

    scores = validator.validate_sentence(
        sentence=atomic_claim,
        premise=real_chunk,
    )

    print_scores(
        "TEST 3 — REAL CHUNK + ATOMIC CLAIM",
        scores,
    )


if __name__ == "__main__":
    main()