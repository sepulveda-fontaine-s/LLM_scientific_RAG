import re


def clean_text(text: str) -> str:
    """
    Apply conservative text cleaning before chunking.

    We intentionally avoid aggressive de-hyphenation because
    a line-ending hyphen may represent either:
    - a word broken by PDF formatting, or
    - a genuine hyphenated expression such as "fine-grained".

    Preserving some formatting noise is preferable to changing
    the semantic content of the document.
    """

    # Replace repeated spaces or tabs with a single space.
    text = re.sub(r"[ \t]+", " ", text)

    # Remove spaces immediately before line breaks.
    text = re.sub(r" +\n", "\n", text)

    # Collapse three or more consecutive line breaks into two.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove leading and trailing whitespace.
    return text.strip()