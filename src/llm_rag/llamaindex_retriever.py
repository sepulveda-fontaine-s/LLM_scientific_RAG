from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from llm_rag.retrieval_pipeline import RetrievalPipeline


class LlamaIndexHybridRetriever(BaseRetriever):
    """
    LlamaIndex adapter around the project's existing retrieval pipeline.

    The underlying retrieval logic remains unchanged:

        dense FAISS
            +
        BM25
            ↓
        RRF fusion
            ↓
        cross-encoder reranking

    This adapter only translates the project's dictionaries into
    LlamaIndex NodeWithScore objects.
    """

    def __init__(
        self,
        rrf_k: int = 60,
        candidate_k: int = 25,
        rerank_k: int = 5,
    ) -> None:
        super().__init__()

        self.pipeline = RetrievalPipeline(
            rrf_k=rrf_k,
            candidate_k=candidate_k,
            rerank_k=rerank_k,
        )

    def _retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> list[NodeWithScore]:
        """
        Retrieve evidence for a LlamaIndex QueryBundle.
        """

        results = self.pipeline.retrieve(
            query=query_bundle.query_str,
        )

        nodes = []

        for result in results:
            node = TextNode(
                text=result["text"],
                id_=result["chunk_id"],
                metadata={
                    "source": result["source"],
                    "page": result["page"],
                    "chunk_id": result["chunk_id"],
                },
            )

            score = result.get(
                "reranker_score",
                result.get("score"),
            )

            nodes.append(
                NodeWithScore(
                    node=node,
                    score=score,
                )
            )

        return nodes
