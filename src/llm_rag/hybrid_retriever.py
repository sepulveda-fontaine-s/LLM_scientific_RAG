from collections import defaultdict

from llm_rag.dense_retriever import DenseRetriever
from llm_rag.sparse_retriever import SparseRetriever


class HybridRetriever:
    """
    Combine dense and sparse retrieval using Reciprocal Rank Fusion (RRF).

    Dense retrieval contributes semantic similarity.
    Sparse retrieval contributes lexical/BM25 relevance.

    RRF combines their rankings rather than their raw scores.
    """

    def __init__(
        self,
        rrf_k: int = 60,
    ) -> None:

        # Dense semantic retriever.
        self.dense_retriever = DenseRetriever()

        # Sparse lexical retriever.
        self.sparse_retriever = SparseRetriever(
            k1=1.5,
            b=0.75,
        )

        # RRF constant.
        # A larger value reduces the difference between nearby ranks.
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 10,
    ) -> list[dict]:
        """
        Retrieve candidates from dense and sparse search,
        then fuse both rankings using RRF.

        Parameters
        ----------
        query:
            Natural-language search query.

        top_k:
            Number of final hybrid results.

        candidate_k:
            Number of candidates retrieved independently
            from each retrieval method before fusion.
        """

        # Retrieve candidate lists independently.
        dense_results = self.dense_retriever.retrieve(
            query=query,
            top_k=candidate_k,
        )

        sparse_results = self.sparse_retriever.retrieve(
            query=query,
            top_k=candidate_k,
        )

        # Store accumulated RRF scores by chunk ID.
        rrf_scores = defaultdict(float)

        # Keep the chunk metadata so we can reconstruct
        # the final results after fusion.
        chunks_by_id = {}

        # Add dense ranking contribution.
        for rank, result in enumerate(dense_results, start=1):
            chunk_id = result["chunk_id"]

            rrf_scores[chunk_id] += 1 / (self.rrf_k + rank)
            chunks_by_id[chunk_id] = result

        # Add sparse ranking contribution.
        for rank, result in enumerate(sparse_results, start=1):
            chunk_id = result["chunk_id"]

            rrf_scores[chunk_id] += 1 / (self.rrf_k + rank)
            chunks_by_id[chunk_id] = result

        # Sort chunks by their combined RRF score.
        ranked_chunk_ids = sorted(
            rrf_scores,
            key=rrf_scores.get,
            reverse=True,
        )

        results = []

        for chunk_id in ranked_chunk_ids[:top_k]:
            chunk = chunks_by_id[chunk_id].copy()
            chunk["rrf_score"] = rrf_scores[chunk_id]

            results.append(chunk)

        return results