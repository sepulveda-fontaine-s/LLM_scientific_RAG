import pytest

from llm_rag.numeric_validator import (
    extract_numbers,
    validate_numeric_grounding,
)


def make_sources(*texts: str) -> list[dict]:
    """Build the minimal source structure required by the validator."""
    return [
        {
            "citation": f"[{index}]",
            "text": text,
        }
        for index, text in enumerate(texts, start=1)
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ColBERT is over 170× cheaper.", {"170×"}),
        ("The paper reports 13,900× fewer FLOPs.", {"13,900×"}),
        ("The latency is 10s.", {"10s"}),
        ("The model has 7B parameters.", {"7B"}),
        ("The experiment reports 97T operations.", {"97T"}),
    ],
)
def test_extract_numbers(
    text: str,
    expected: set[str],
) -> None:
    """Extract the numeric forms used by the RAG evaluation corpus."""

    assert extract_numbers(text) == expected


@pytest.mark.parametrize(
    ("answer", "sources", "expected"),
    [
        (
            "DPR processes 995.0 questions per second [1].",
            make_sources(
                "DPR processes 995.0 questions per second."
            ),
            True,
        ),
        (
            "DPR processes 995.0 questions per second [1].",
            make_sources(
                "DPR processes 23.7 questions per second."
            ),
            False,
        ),
        (
            "ColBERT is over 170× cheaper in latency [1].",
            make_sources(
                "ColBERT is over 170× cheaper in latency."
            ),
            True,
        ),
        (
            "The model uses 7B parameters [1].",
            make_sources(
                "The model uses 7B parameters."
            ),
            True,
        ),
        (
            "The answer contains no explicit measurement [1].",
            make_sources(
                "The source also contains no explicit measurement."
            ),
            True,
        ),
    ],
)
def test_validate_numeric_grounding(
    answer: str,
    sources: list[dict],
    expected: bool,
) -> None:
    """Check that explicit numbers occur in the cited evidence."""

    assert validate_numeric_grounding(
        answer=answer,
        sources=sources,
    ) is expected
