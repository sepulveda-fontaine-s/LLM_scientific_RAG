from typing import Dict, List

from llm_rag.text_cleaner import clean_text


def chunk_text(
    text: str,
    chunk_size: int = 200,
    overlap: int = 40,
) -> List[str]:
    """
    Split text into overlapping word-based chunks.

    Parameters
    ----------
    text:
        Input text to split.

    chunk_size:
        Maximum number of words per chunk.

    overlap:
        Number of words shared between consecutive chunks.
    """

    # Basic validation.
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    # Simple baseline tokenization by whitespace.
    words = text.split()

    chunks = []

    # Example:
    # chunk_size = 200, overlap = 40
    # step = 160
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk_words = words[start:end]

        if not chunk_words:
            break

        chunks.append(" ".join(chunk_words))

        # Stop when the last chunk reaches the end of the text.
        if end >= len(words):
            break

    return chunks


def chunk_pages(
    pages: List[Dict],
    chunk_size: int = 200,
    overlap: int = 40,
) -> List[Dict]:
    """
    Clean and chunk PDF pages while preserving metadata.

    Each resulting chunk keeps:
    - source document
    - page number
    - unique chunk ID
    - chunk text
    """

    chunks = []

    for page in pages:
        # Clean the extracted PDF text first.
        cleaned_text = clean_text(page["text"])

        # Split the current page into overlapping chunks.
        page_chunks = chunk_text(
            cleaned_text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk_number, chunk_text_value in enumerate(
            page_chunks,
            start=1,
        ):
            # Human-readable identifier for traceability.
            chunk_id = (
                f"{page['source']}"
                f"_page_{page['page']}"
                f"_chunk_{chunk_number}"
            )

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "source": page["source"],
                    "page": page["page"],
                    "text": chunk_text_value,
                }
            )

    return chunks