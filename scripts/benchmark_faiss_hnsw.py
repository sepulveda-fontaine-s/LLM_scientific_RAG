import json
import time
from pathlib import Path

import faiss
import numpy as np

from llm_rag.dense_retriever import DenseRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval_eval.jsonl"
)

TOP_K = 10

# HNSW hyperparameters.
HNSW_M = 16
EF_CONSTRUCTION = 40
EF_SEARCH = 32


def load_queries(path: Path) -> list[str]:
    """Load development queries."""

    queries = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                item = json.loads(line)
                queries.append(item["query"])

    return queries


def main() -> None:
    retriever = DenseRetriever()
    queries = load_queries(EVAL_FILE)

    # Precompute query embeddings so model inference does not
    # contaminate the index-search latency measurement.
    query_embeddings = retriever.embedder.encode(
        queries,
        batch_size=16,
    )

    query_embeddings = np.ascontiguousarray(
        query_embeddings,
        dtype=np.float32,
    )

    dimension = retriever.embeddings.shape[1]

    # Exact reference index.
    exact_index = faiss.IndexFlatIP(dimension)
    exact_index.add(retriever.embeddings)

    # Approximate HNSW index.
    hnsw_index = faiss.IndexHNSWFlat(
        dimension,
        HNSW_M,
        faiss.METRIC_INNER_PRODUCT,
    )

    hnsw_index.hnsw.efConstruction = EF_CONSTRUCTION
    hnsw_index.hnsw.efSearch = EF_SEARCH

    hnsw_index.add(retriever.embeddings)

    exact_start = time.perf_counter()

    exact_scores, exact_indices = exact_index.search(
        query_embeddings,
        TOP_K,
    )

    exact_time = time.perf_counter() - exact_start

    hnsw_start = time.perf_counter()

    hnsw_scores, hnsw_indices = hnsw_index.search(
        query_embeddings,
        TOP_K,
    )

    hnsw_time = time.perf_counter() - hnsw_start

    overlap_scores = []
    top1_matches = 0

    for exact_row, hnsw_row in zip(
        exact_indices,
        hnsw_indices,
    ):
        exact_set = set(exact_row.tolist())
        hnsw_set = set(hnsw_row.tolist())

        overlap = (
            len(exact_set & hnsw_set)
            / TOP_K
        )

        overlap_scores.append(overlap)

        if exact_row[0] == hnsw_row[0]:
            top1_matches += 1

    mean_overlap = float(np.mean(overlap_scores))

    top1_agreement = (
        top1_matches
        / len(queries)
    )

    print("=" * 80)
    print("FAISS HNSW BENCHMARK")
    print("=" * 80)

    print(f"Queries tested: {len(queries)}")
    print(f"Corpus vectors: {len(retriever.embeddings)}")
    print(f"Embedding dimension: {dimension}")

    print()
    print(f"HNSW M: {HNSW_M}")
    print(f"efConstruction: {EF_CONSTRUCTION}")
    print(f"efSearch: {EF_SEARCH}")

    print()
    print(
        f"Exact top-{TOP_K} overlap recovered: "
        f"{mean_overlap:.4f}"
    )

    print(
        f"Exact top-1 agreement: "
        f"{top1_agreement:.4f}"
    )

    print()
    print(
        f"Exact IndexFlatIP search time: "
        f"{exact_time * 1000:.4f} ms"
    )

    print(
        f"HNSW search time: "
        f"{hnsw_time * 1000:.4f} ms"
    )


if __name__ == "__main__":
    main()