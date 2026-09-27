from pathlib import Path

import faiss
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EMBEDDINGS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "embeddings.npy"
)

INDEX_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "faiss.index"
)


def main() -> None:
    embeddings = np.ascontiguousarray(
        np.load(EMBEDDINGS_FILE),
        dtype=np.float32,
    )

    if embeddings.ndim != 2:
        raise ValueError(
            "Embeddings must be a 2D matrix."
        )

    number_of_vectors, dimension = embeddings.shape

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    if index.ntotal != number_of_vectors:
        raise ValueError(
            "FAISS index does not contain all embeddings."
        )

    faiss.write_index(
        index,
        str(INDEX_FILE),
    )

    print("FAISS INDEX BUILT")
    print(f"Vectors: {index.ntotal}")
    print(f"Dimension: {dimension}")
    print(f"Index type: {type(index).__name__}")
    print(f"Saved to: {INDEX_FILE}")


if __name__ == "__main__":
    main()