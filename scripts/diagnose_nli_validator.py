from llm_rag.groundedness_validator import GroundednessValidator


def print_result(title: str, scores: dict) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    for key, value in scores.items():
        print(f"{key:15s}: {value:.6f}")


def main() -> None:
    validator = GroundednessValidator()

    # A deliberately simple entailment case.
    premise = (
        "Building the FAISS index on 21-million vectors "
        "on a single server takes 8.5 hours. "
        "Building an inverted index using Lucene takes "
        "about 30 minutes."
    )

    hypothesis = (
        "Building the FAISS index takes 8.5 hours, "
        "compared with 30 minutes for the Lucene inverted index."
    )

    scores = validator.validate_sentence(
        sentence=hypothesis,
        premise=premise,
    )

    print_result(
        "TEST 1 — CLEAN ENTAILMENT",
        scores,
    )

    # Explicit contradiction.
    contradiction = (
        "Building the FAISS index takes 30 minutes, "
        "while the Lucene index takes 8.5 hours."
    )

    scores = validator.validate_sentence(
        sentence=contradiction,
        premise=premise,
    )

    print_result(
        "TEST 2 — CLEAN CONTRADICTION",
        scores,
    )

    # Same entailment, but intentionally reverse NLI input order.
    scores = validator.validate_sentence(
        sentence=premise,
        premise=hypothesis,
    )

    print_result(
        "TEST 3 — REVERSED INPUT ORDER",
        scores,
    )


if __name__ == "__main__":
    main()