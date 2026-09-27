from typing import Any

from llama_index.core.base.response.schema import Response
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever

from llm_rag.llamaindex_retriever import LlamaIndexHybridRetriever
from llm_rag.rag_pipeline import RAGPipeline


class LlamaIndexRAGQueryEngine(CustomQueryEngine):
    """
    LlamaIndex query engine that reuses the project's existing RAG pipeline.

    Flow:
        query
            ↓
        LlamaIndex BaseRetriever adapter
            ↓
        NodeWithScore objects
            ↓
        convert nodes back to project result dictionaries
            ↓
        RAGPipeline generation + guardrails
            ↓
        LlamaIndex Response

    Retrieval is performed only once by LlamaIndex. RAGPipeline receives
    those retrieved results through provided_results, so its lazy manual
    retriever is not instantiated.
    """

    retriever: BaseRetriever
    rag_pipeline: Any

    @classmethod
    def from_defaults(
        cls,
        rrf_k: int = 60,
        candidate_k: int = 25,
        rerank_k: int = 5,
    ) -> "LlamaIndexRAGQueryEngine":
        """
        Build the query engine using the project's default retrieval stack.
        """

        retriever = LlamaIndexHybridRetriever(
            rrf_k=rrf_k,
            candidate_k=candidate_k,
            rerank_k=rerank_k,
        )

        rag_pipeline = RAGPipeline(
            rrf_k=rrf_k,
            candidate_k=candidate_k,
            rerank_k=rerank_k,
        )

        return cls(
            retriever=retriever,
            rag_pipeline=rag_pipeline,
        )

    @staticmethod
    def _nodes_to_results(nodes) -> list[dict]:
        """
        Convert LlamaIndex NodeWithScore objects back into the dictionary
        structure expected by RAGPipeline.
        """

        results = []

        for item in nodes:
            metadata = item.node.metadata

            results.append(
                {
                    "text": item.node.get_content(),
                    "source": metadata["source"],
                    "page": metadata["page"],
                    "chunk_id": metadata["chunk_id"],
                    "reranker_score": item.score,
                }
            )

        return results

    def custom_query(
        self,
        query_str: str,
    ) -> Response:
        """
        Execute retrieval through LlamaIndex and reuse the project's
        generation, validation, and fail-closed logic.
        """

        nodes = self.retriever.retrieve(
            query_str,
        )

        results = self._nodes_to_results(
            nodes,
        )

        rag_result = self.rag_pipeline.answer(
            query=query_str,
            provided_results=results,
        )

        return Response(
            response=rag_result["answer"],
            source_nodes=nodes,
            metadata={
                "validation_status": rag_result["validation_status"],
                "hard_guardrails_passed": rag_result[
                    "hard_guardrails_passed"
                ],
                "citations_valid": rag_result["citations_valid"],
                "numeric_grounding_valid": rag_result[
                    "numeric_grounding_valid"
                ],
                "grounded": rag_result["grounded"],
                "answer_before_fail_closed": rag_result[
                    "answer_before_fail_closed"
                ],
            },
        )
