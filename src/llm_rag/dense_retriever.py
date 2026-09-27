import json
from pathlib import Path

import faiss
import numpy as np

from llm_rag.embedder import Embedder


# Project root:
# Advanced_LLM_RAG/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CHUNKS_FILE = (
    PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
)

DEFAULT_EMBEDDINGS_FILE = (
    PROJECT_ROOT / "data" / "processed" / "embeddings.npy"
)

DEFAULT_FAISS_INDEX_FILE = (
    PROJECT_ROOT / "data" / "processed" / "faiss.index"
)


class DenseRetriever:
    """
    Retrieve chunks using exact dense vector search with a persisted FAISS index.

    Pipeline:
        query text
            ↓
        query embedding
            ↓
        persisted FAISS IndexFlatIP
            ↓
        top-k most similar chunks

    The document and query embeddings are L2-normalized, so inner
    product is equivalent to cosine similarity.

    IndexFlatIP performs exact exhaustive search; it is not ANN.
    """

    def __init__(
        self,
        chunks_file: Path = DEFAULT_CHUNKS_FILE,
        embeddings_file: Path = DEFAULT_EMBEDDINGS_FILE,
        index_file: Path = DEFAULT_FAISS_INDEX_FILE,
    ) -> None:

        # Load chunk metadata and text.
        self.chunks = self._load_chunks(chunks_file)

        # Keep the embeddings available for validation, diagnostics,
        # and benchmark scripts.
        self.embeddings = np.ascontiguousarray(
            np.load(embeddings_file),
            dtype=np.float32,
        )

        if self.embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2D matrix."
            )

        if len(self.chunks) != self.embeddings.shape[0]:
            raise ValueError(
                "Number of chunks does not match number of embeddings."
            )

        # Load the FAISS index built offline.
        if not index_file.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_file}. "
                "Run scripts/build_faiss_index.py first."
            )

        self.index = faiss.read_index(
            str(index_file)
        )

        # Validate index cardinality and dimensionality.
        if self.index.ntotal != len(self.chunks):
            raise ValueError(
                "FAISS index size does not match number of chunks."
            )

        if self.index.d != self.embeddings.shape[1]:
            raise ValueError(
                "FAISS index dimension does not match embeddings."
            )

        # Load the same embedding model used to encode the chunks.
        self.embedder = Embedder()

    @staticmethod
    def _load_chunks(chunks_file: Path) -> list[dict]:
        """Load chunks stored in JSONL format."""

        chunks = []

        with chunks_file.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    chunks.append(json.loads(line))

        return chunks

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Retrieve the top-k chunks most semantically similar to the query.
        """

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        if not self.chunks:
            return []

        k = min(top_k, len(self.chunks))

        # Encode query in the same vector space used by the indexed chunks.
        query_embedding = self.embedder.encode(
            [query],
            batch_size=1,
        )[0]

        # FAISS expects a contiguous float32 matrix with shape (n_queries, d).
        query_matrix = np.ascontiguousarray(
            query_embedding.reshape(1, -1),
            dtype=np.float32,
        )

        scores, indices = self.index.search(
            query_matrix,
            k,
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            chunk = self.chunks[int(index)].copy()
            chunk["score"] = float(score)
            results.append(chunk)

        return results
