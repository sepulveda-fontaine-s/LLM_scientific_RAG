from llm_rag.citation_validator import validate_citations


def main() -> None:
    test_cases = [
        {
            "name": "single valid citation",
            "answer": "ColBERT uses late interaction [1].",
            "num_sources": 3,
            "expected": True,
        },
        {
            "name": "multiple valid citations",
            "answer": "The claim is supported by two sources [1][2].",
            "num_sources": 3,
            "expected": True,
        },
        {
            "name": "double brackets",
            "answer": "ColBERT uses late interaction [[3]].",
            "num_sources": 3,
            "expected": False,
        },
        {
            "name": "citation out of range",
            "answer": "The claim is supported [4].",
            "num_sources": 3,
            "expected": False,
        },
        {
            "name": "zero citation",
            "answer": "The claim is supported [0].",
            "num_sources": 3,
            "expected": False,
        },
        {
            "name": "sentence without citation",
            "answer": (
                "The first claim is supported [1]. "
                "The second claim has no citation."
            ),
            "num_sources": 3,
            "expected": False,
        },
        {
            "name": "malformed extra closing bracket",
            "answer": "The claim is supported [1]].",
            "num_sources": 3,
            "expected": False,
        },
        {
            "name": "malformed extra opening bracket",
            "answer": "The claim is supported [[1].",
            "num_sources": 3,
            "expected": False,
        },
    ]

    failures = 0

    for case in test_cases:
        actual = validate_citations(
            answer=case["answer"],
            num_sources=case["num_sources"],
        )

        passed = actual == case["expected"]

        status = "PASS" if passed else "FAIL"

        print(
            f"{status:4s} | "
            f"{case['name']:32s} | "
            f"expected={case['expected']} "
            f"actual={actual}"
        )

        if not passed:
            failures += 1

    if failures:
        raise SystemExit(
            f"\n{failures} citation validator test(s) failed."
        )

    print(
        f"\nAll {len(test_cases)} citation validator tests passed."
    )


if __name__ == "__main__":
    main()