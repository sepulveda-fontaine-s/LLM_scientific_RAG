from pathlib import Path

from llm_rag.chunker import chunk_pages
from llm_rag.document_loader import load_pdf_pages


# Project folders.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def main() -> None:
    """
    Load one PDF, split it into chunks, and inspect a few examples.

    This is a diagnostic step: before generating embeddings,
    we want to verify that chunk size, overlap, metadata, and
    text boundaries look reasonable.
    """

    # Use the first PDF alphabetically for this inspection.
    pdf_path = sorted(RAW_DATA_DIR.glob("*.pdf"))[0]

    # Extract text page by page.
    pages = load_pdf_pages(pdf_path)

    # Create our baseline word-based chunks.
    chunks = chunk_pages(
        pages,
        chunk_size=200,
        overlap=40,
    )

    print(f"PDF: {pdf_path.name}")
    print(f"Pages: {len(pages)}")
    print(f"Chunks generated: {len(chunks)}")

    # Inspect the first three chunks.
    for chunk in chunks[:3]:
        print("\n" + "=" * 80)
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Page: {chunk['page']}")
        print(f"Words: {len(chunk['text'].split())}")
        print("-" * 80)
        print(chunk["text"])


if __name__ == "__main__":
    main()