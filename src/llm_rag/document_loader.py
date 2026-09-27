from pathlib import Path

import pymupdf

# Root folder of the project:
# LLM_RAG/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Folder where the original PDF documents are stored.
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_pdf_pages(pdf_path: Path) -> list[dict]:
    """
    Extract text from a PDF page by page.

    Each page is stored as a dictionary containing:
    - source: original PDF filename
    - page: page number starting from 1
    - text: extracted text
    """

    pages = []

    # Open the PDF using PyMuPDF.
    with pymupdf.open(pdf_path) as document:

        # enumerate(..., start=1) makes page numbering human-readable.
        for page_number, page in enumerate(document, start=1):

            # Extract plain text from the current page.
            text = page.get_text("text")

            pages.append(
                {
                    "source": pdf_path.name,
                    "page": page_number,
                    "text": text,
                }
            )

    return pages


def main() -> None:
    # Find every PDF stored in data/raw.
    pdf_files = sorted(RAW_DATA_DIR.glob("*.pdf"))

    for pdf_path in pdf_files:
        pages = load_pdf_pages(pdf_path)

        # Simple diagnostic to verify that text extraction worked.
        total_characters = sum(len(page["text"]) for page in pages)

        print(f"\n{pdf_path.name}")
        print(f"Pages: {len(pages)}")
        print(f"Characters extracted: {total_characters:,}")


if __name__ == "__main__":
    main()