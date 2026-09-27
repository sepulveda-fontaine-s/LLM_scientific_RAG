import pytest

from llm_rag.citation_validator import validate_citations


@pytest.mark.parametrize(
    ("answer", "num_sources", "expected"),
    [
        (
            "ColBERT uses late interaction [1].",
            3,
            True,
        ),
        (
            "ColBERT uses late interaction [1] and pre-computed representations [2].",
            3,
            True,
        ),
        (
            "ColBERT uses late interaction [[1]].",
            3,
            False,
        ),
        (
            "ColBERT uses late interaction [4].",
            3,
            False,
        ),
        (
            "ColBERT uses late interaction [0].",
            3,
            False,
        ),
        (
            "ColBERT uses late interaction [1]. The model pre-computes document representations.",
            3,
            False,
        ),
        (
            "ColBERT uses late interaction [1]].",
            3,
            False,
        ),
        (
            "ColBERT uses late interaction [[1].",
            3,
            False,
        ),
    ],
)
def test_validate_citations(
    answer: str,
    num_sources: int,
    expected: bool,
) -> None:
    """Validate accepted and rejected citation formats."""

    assert validate_citations(
        answer=answer,
        num_sources=num_sources,
    ) is expected
