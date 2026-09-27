import re


def format_source_citations(
    text: str,
    num_sources: int,
) -> str:
    """
    Convert internal SOURCE_ID references into user-facing citations.

    Only citations originating from SOURCE_ID=n are considered valid.
    Raw numeric citations copied from source documents are rejected.

    Example:
        SOURCE_ID=2 -> [2]
        [88]        -> [INVALID_SOURCE]
    """

    protected_citations = {}

    def protect_source_id(match: re.Match) -> str:
        source_id = int(match.group(1))

        if not 1 <= source_id <= num_sources:
            return "[INVALID_SOURCE]"

        placeholder = f"__RAG_SOURCE_{source_id}__"

        protected_citations[placeholder] = (
            f"[{source_id}]"
        )

        return placeholder

    # Protect citations explicitly generated using our internal namespace.
    text = re.sub(
        r"SOURCE_ID\s*=\s*(\d+)",
        protect_source_id,
        text,
    )

    # Any remaining [n] was not generated from SOURCE_ID=n.
    # It may be a bibliographic reference copied from a paper.
    text = re.sub(
        r"\[\d+\]",
        "[INVALID_SOURCE]",
        text,
    )

    # Restore trusted RAG citations.
    for placeholder, citation in protected_citations.items():
        text = text.replace(
            placeholder,
            citation,
        )

    return text