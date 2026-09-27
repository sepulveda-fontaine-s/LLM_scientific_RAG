import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect chunks manually for retrieval ground-truth annotation."
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Substring of the source PDF filename.",
    )

    parser.add_argument(
        "--start-page",
        type=int,
        required=True,
        help="First page to inspect.",
    )

    parser.add_argument(
        "--end-page",
        type=int,
        required=True,
        help="Last page to inspect.",
    )

    args = parser.parse_args()

    chunks = load_jsonl(CHUNKS_PATH)

    selected_chunks = [
        chunk
        for chunk in chunks
        if args.source.lower() in chunk["source"].lower()
        and args.start_page <= chunk["page"] <= args.end_page
    ]

    if not selected_chunks:
        print("No chunks found for the requested source/page range.")
        return

    print(
        f"Source filter: {args.source}\n"
        f"Pages: {args.start_page}-{args.end_page}\n"
        f"Chunks found: {len(selected_chunks)}"
    )

    for chunk in selected_chunks:
        print("\n" + "=" * 80)
        print("Chunk ID:", repr(chunk["chunk_id"]))
        print("Source:", chunk["source"])
        print("Page:", chunk["page"])
        print("\nText:")
        print(chunk["text"])


if __name__ == "__main__":
    main()