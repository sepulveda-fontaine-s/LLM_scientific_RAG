from pathlib import Path

import pymupdf

from llm_rag.text_cleaner import clean_text


# Project directories.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def main() -> None:
    # Take the first PDF only for this diagnostic.
    pdf_path = sorted(RAW_DATA_DIR.glob("*.pdf"))[0]

    with pymupdf.open(pdf_path) as document:
        # Inspect the first page.
        original_text = document[0].get_text("text")
        cleaned_text = clean_text(original_text)

    print(f"PDF: {pdf_path.name}")

    print("\n--- ORIGINAL ---\n")
    print(original_text[:2000])

    print("\n--- CLEANED ---\n")
    print(cleaned_text[:2000])


if __name__ == "__main__":
    main()