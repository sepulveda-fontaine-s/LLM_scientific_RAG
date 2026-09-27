import json
from pathlib import Path

from llm_rag.chunker import chunk_pages
from llm_rag.document_loader import load_pdf_pages


# Project directories.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_FILE = PROCESSED_DATA_DIR / "chunks.jsonl"


def main() -> None:
    """
    Load every PDF, generate baseline chunks,
    and save them as JSON Lines.

    JSONL stores one JSON object per line, which is convenient
    for later embedding, indexing, and experiment pipelines.
    """

    pdf_files = sorted(RAW_DATA_DIR.glob("*.pdf"))

    all_chunks = []

    for pdf_path in pdf_files:
        # Extract pages from the current PDF.
        pages = load_pdf_pages(pdf_path)

        # Generate baseline chunks.
        chunks = chunk_pages(
            pages,
            chunk_size=200,
            overlap=40,
        )

        all_chunks.extend(chunks)

        print(
            f"{pdf_path.name}: "
            f"{len(pages)} pages -> {len(chunks)} chunks"
        )

    # Ensure the output directory exists.
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save one chunk per line.
    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for chunk in all_chunks:
            file.write(
                json.dumps(chunk, ensure_ascii=False) + "\n"
            )

    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()