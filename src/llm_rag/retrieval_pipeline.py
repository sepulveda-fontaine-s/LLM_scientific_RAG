from llm_rag.hybrid_retriever import HybridRetriever
from llm_rag.reranker import Reranker


class RetrievalPipeline:
    """
    End-to-end retrieval pipeline.

    Flow:
        query
          ↓
        dense retrieval + BM25
          ↓
        RRF hybrid fusion
          ↓
        cross-encoder reranking
          ↓
        final top-k chunks

    This class hides the orchestration, but not the individual
    components: each retrieval stage still lives in its own module.
    """

    def __init__(
        self,
        rrf_k: int = 60,
        candidate_k: int = 25,
        rerank_k: int = 5,
    ) -> None:
        """
        Parameters
        ----------
        rrf_k:
            Constant used by Reciprocal Rank Fusion.

        candidate_k:
            Number of candidates retrieved from each branch
            before hybrid fusion.

        rerank_k:
            Number of final chunks returned after reranking.
        """

        self.candidate_k = candidate_k
        self.rerank_k = rerank_k

        # Hybrid retrieval:
        # dense semantic search + BM25 lexical search + RRF.
        self.hybrid_retriever = HybridRetriever(
            rrf_k=rrf_k,
        )

        # Cross-encoder used for final relevance scoring.
        self.reranker = Reranker()

    def retrieve(
        self,
        query: str,
    ) -> list[dict]:
        """
        Retrieve and rerank the most relevant chunks for a query.
        """

        # Stage 1:
        # retrieve a broader candidate set using hybrid retrieval.
        candidates = self.hybrid_retriever.retrieve(
            query=query,
            top_k=self.candidate_k,
            candidate_k=self.candidate_k,
        )

        # Stage 2:
        # rerank those candidates using the cross-encoder.
        final_results = self.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=self.rerank_k,
        )

        return final_results