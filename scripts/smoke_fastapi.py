import json
import time
import urllib.error
import urllib.request


BASE_URL = "http://127.0.0.1:8765"

QUERY = (
    "How does ColBERT compare with BERT-based rerankers "
    "in latency and computational cost?"
)


def wait_for_health(
    timeout_seconds: int = 180,
) -> None:
    """Wait until the FastAPI service is ready."""

    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"{BASE_URL}/health",
                timeout=5,
            ) as response:
                payload = json.loads(
                    response.read().decode("utf-8")
                )

            if payload.get("status") == "ok":
                print("HEALTH: PASS")
                return

        except (
            urllib.error.URLError,
            TimeoutError,
        ):
            time.sleep(2)

    raise TimeoutError(
        "FastAPI service did not become ready."
    )


def test_query() -> None:
    """Send one real POST request to the RAG API."""

    payload = json.dumps(
        {"query": QUERY}
    ).encode("utf-8")

    request = urllib.request.Request(
        f"{BASE_URL}/query",
        data=payload,
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(
        request,
        timeout=180,
    ) as response:
        result = json.loads(
            response.read().decode("utf-8")
        )

    print("\nQUERY: PASS")

    print("\nANSWER:")
    print(result["answer"])

    print("\nVALIDATION:")
    print(
        json.dumps(
            result["validation"],
            indent=2,
        )
    )

    print(
        "\nSOURCE COUNT:",
        len(result["sources"]),
    )


def main() -> None:
    wait_for_health()
    test_query()

    print("\nFASTAPI SMOKE TEST: PASS")


if __name__ == "__main__":
    main()