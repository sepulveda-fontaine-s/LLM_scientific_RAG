import re


NUMBER_PATTERN = re.compile(
    r"(?<!\w)"
    r"\d[\d,]*(?:\.\d+)?"
    r"(?:K|M|B|T|ms|s|%|×|x)?"
    r"(?!\w)",
    re.IGNORECASE,
)


def extract_numbers(text: str) -> set[str]:
    """
    Extract explicit numeric expressions from text.

    Examples:
        13
        170×
        13,900×
        10s
        7B
        97T

    Lookarounds are used instead of a final word boundary because
    suffixes such as × and % are non-word characters. A trailing \b
    would otherwise stop the regex before those suffixes.
    """
    return set(
        NUMBER_PATTERN.findall(text)
    )


def validate_numeric_grounding(
    answer: str,
    sources: list[dict],
) -> bool:
    """
    Ensure every explicit number in the answer appears
    in at least one cited source for that sentence.
    """

    sentences = re.split(
        r"(?<=[.!?])\s+",
        answer.strip(),
    )

    for sentence in sentences:
        if not sentence.strip():
            continue

        # Citation identifiers such as [1] are metadata,
        # not factual numbers.
        sentence_without_citations = re.sub(
            r"\[\d+\]",
            "",
            sentence,
        )

        answer_numbers = extract_numbers(
            sentence_without_citations
        )

        if not answer_numbers:
            continue

        citation_numbers = [
            int(value)
            for value in re.findall(
                r"\[(\d+)\]",
                sentence,
            )
        ]

        cited_text = " ".join(
            sources[citation - 1]["text"]
            for citation in citation_numbers
            if 1 <= citation <= len(sources)
        )

        source_numbers = extract_numbers(
            cited_text
        )

        if not answer_numbers.issubset(
            source_numbers
        ):
            return False

    return True
