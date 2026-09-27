import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GENERATION_EVAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "generation_eval.jsonl"
)

AUDIT_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "generation_manual_audit.jsonl"
)

VALID_STATUSES = {
    "full",
    "partial",
    "missing",
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def main() -> None:
    evaluation_cases = load_jsonl(
        GENERATION_EVAL_PATH
    )

    audit_cases = load_jsonl(
        AUDIT_PATH
    )

    expected_by_query = {
        case["query"]: case["expected_claims"]
        for case in evaluation_cases
    }

    errors = []

    status_counts = {
        "full": 0,
        "partial": 0,
        "missing": 0,
    }

    seen_queries = set()

    for case_index, audit_case in enumerate(
        audit_cases,
        start=1,
    ):
        query = audit_case.get("query")

        if query in seen_queries:
            errors.append(
                f"Duplicate query in audit: {query}"
            )

        seen_queries.add(query)

        if query not in expected_by_query:
            errors.append(
                f"Unknown query in audit: {query}"
            )
            continue

        expected_claims = expected_by_query[query]
        claim_audit = audit_case.get(
            "claim_audit",
            [],
        )

        audited_claims = [
            item.get("expected_claim")
            for item in claim_audit
        ]

        if audited_claims != expected_claims:
            errors.append(
                f"Claim mismatch for query: {query}"
            )

        for item in claim_audit:
            status = item.get("status")
            reason = item.get("reason")

            if status not in VALID_STATUSES:
                errors.append(
                    f"Invalid status '{status}' "
                    f"for query: {query}"
                )
                continue

            status_counts[status] += 1

            if not reason:
                errors.append(
                    f"Missing reason for claim "
                    f"in query: {query}"
                )

    expected_queries = set(expected_by_query)

    if seen_queries != expected_queries:
        missing_queries = (
            expected_queries - seen_queries
        )

        for query in missing_queries:
            errors.append(
                f"Query missing from audit: {query}"
            )

    if errors:
        print(
            "INVALID GENERATION MANUAL AUDIT"
        )

        for error in errors:
            print(f"- {error}")

        raise SystemExit(1)

    total_claims = sum(
        status_counts.values()
    )

    print(
        f"Manual generation audit valid: "
        f"{len(audit_cases)} cases, "
        f"{total_claims} claims."
    )

    print()
    print("MANUAL CLAIM COVERAGE")
    print("=" * 50)

    print(
        f"Full:    {status_counts['full']}"
    )
    print(
        f"Partial: {status_counts['partial']}"
    )
    print(
        f"Missing: {status_counts['missing']}"
    )


if __name__ == "__main__":
    main()