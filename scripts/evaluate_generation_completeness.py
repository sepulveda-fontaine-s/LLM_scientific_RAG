import json
import re
from pathlib import Path

from llm_rag.groundedness_validator import GroundednessValidator


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "generation_oracle_results.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "generation_completeness_results.json"
)

ENTAILMENT_THRESHOLD = 0.70


def clean_answer(answer: str) -> str:
    """Remove citation markers before claim-coverage evaluation."""
    answer = re.sub(r"\[\[?\d+\]?\]", "", answer)
    return answer.strip()


def main() -> None:
    with RESULTS_PATH.open("r", encoding="utf-8") as file:
        oracle_results = json.load(file)

    validator = GroundednessValidator(
        entailment_threshold=ENTAILMENT_THRESHOLD,
    )

    evaluated_cases = []

    total_expected_claims = 0
    total_covered_claims = 0

    for index, case in enumerate(
        oracle_results["cases"],
        start=1,
    ):
        # Use the candidate generation, not the fail-closed message.
        generated_answer = clean_answer(
            case["answer_before_fail_closed"]
        )

        claim_results = []

        for expected_claim in case["expected_claims"]:
            scores = validator.validate_sentence(
                sentence=expected_claim,
                premise=generated_answer,
            )

            covered = (
                scores["entailment"]
                >= ENTAILMENT_THRESHOLD
            )

            total_expected_claims += 1
            total_covered_claims += int(covered)

            claim_results.append(
                {
                    "expected_claim": expected_claim,
                    "entailment_score": scores["entailment"],
                    "contradiction_score": scores["contradiction"],
                    "neutral_score": scores["neutral"],
                    "covered": covered,
                }
            )

        covered_count = sum(
            int(result["covered"])
            for result in claim_results
        )

        claim_count = len(claim_results)

        coverage = (
            covered_count / claim_count
            if claim_count
            else 0.0
        )

        evaluated_cases.append(
            {
                "query": case["query"],
                "generated_answer": generated_answer,
                "covered_claims": covered_count,
                "total_expected_claims": claim_count,
                "claim_coverage": coverage,
                "claim_results": claim_results,
            }
        )

        print("\n" + "=" * 100)
        print(f"CASE {index}")
        print("=" * 100)
        print(case["query"])

        print(
            f"\nExpected claim coverage: "
            f"{covered_count}/{claim_count} "
            f"({coverage:.4f})"
        )

        for claim_result in claim_results:
            status = (
                "COVERED"
                if claim_result["covered"]
                else "MISSING"
            )

            print(
                f"\n[{status}] "
                f"entailment="
                f"{claim_result['entailment_score']:.4f}"
            )
            print(
                claim_result["expected_claim"]
            )

    overall_coverage = (
        total_covered_claims / total_expected_claims
        if total_expected_claims
        else 0.0
    )

    aggregate = {
        "total_expected_claims": total_expected_claims,
        "total_covered_claims": total_covered_claims,
        "expected_claim_coverage": overall_coverage,
        "entailment_threshold": ENTAILMENT_THRESHOLD,
    }

    print("\n" + "=" * 100)
    print("GENERATION COMPLETENESS RESULTS")
    print("=" * 100)

    print(
        f"Covered expected claims: "
        f"{total_covered_claims}/"
        f"{total_expected_claims}"
    )

    print(
        f"Expected claim coverage: "
        f"{overall_coverage:.4f}"
    )

    output = {
        "mode": "oracle_generation_completeness",
        "aggregate": aggregate,
        "cases": evaluated_cases,
    }

    with OUTPUT_PATH.open(
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
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()