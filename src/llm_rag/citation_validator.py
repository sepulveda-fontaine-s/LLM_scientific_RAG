import re


VALID_CITATION_PATTERN = re.compile(
    r"(?<!\[)\[(\d+)\](?!\])"
)

DOUBLE_BRACKET_PATTERN = re.compile(
    r"\[\s*\[\s*\d+\s*\]\s*\]"
)


def extract_citations(text: str) -> list[int]:
    """Extract strictly formatted citations such as [1], [2], ..."""
    return [
        int(value)
        for value in VALID_CITATION_PATTERN.findall(text)
    ]


def validate_citations(
    answer: str,
    num_sources: int,
) -> bool:
    """
    Validate citation compliance.

    Requirements:
    - Citations must use exactly one pair of brackets: [n].
    - The answer must contain at least one citation.
    - Every citation must refer to an available source.
    - Every non-empty sentence must contain at least one citation.
    """

    # Reject malformed citation syntax such as [[3]].
    if DOUBLE_BRACKET_PATTERN.search(answer):
        return False

    citations = extract_citations(answer)

    if not citations:
        return False

    if not all(
        1 <= citation <= num_sources
        for citation in citations
    ):
        return False

    sentences = re.split(
        r"(?<=[.!?])\s+",
        answer.strip(),
    )

    for sentence in sentences:
        if sentence.strip() and not extract_citations(sentence):
            return False

    return True