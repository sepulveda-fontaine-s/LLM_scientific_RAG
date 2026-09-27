import json
from pathlib import Path

import numpy as np

from llm_rag.embedder import Embedder


# Project folders.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "embeddings.npy"


def load_chunks() -> list[dict]:
    """
    Load processed chunks from the JSONL file.
    """

    chunks = []

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            chunks.append(json.loads(line))

    return chunks


def main() -> None:
    """
    Generate dense embeddings for all processed chunks
    and save them as a NumPy matrix.
    """

    chunks = load_chunks()

    # Extract only the text because the embedding model
    # operates on text, not on metadata.
    texts = [chunk["text"] for chunk in chunks]

    print(f"Chunks loaded: {len(chunks)}")

    # Load the local BGE embedding model.
    embedder = Embedder()

    # Generate one 384-dimensional vector per chunk.
    embeddings = embedder.encode(
        texts,
        batch_size=32,
    )

    print(f"Embeddings shape: {embeddings.shape}")

    # Save the matrix for later retrieval/indexing.
    np.save(OUTPUT_FILE, embeddings)

    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()