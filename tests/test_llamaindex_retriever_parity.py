import json
from pathlib import Path
import pytest
from llama_index.core.schema import QueryBundle

from llm_rag.llamaindex_retriever import (
    LlamaIndexHybridRetriever,
)
from llm_rag.retrieval_pipeline import (
    RetrievalPipeline,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval_eval.jsonl"
)

SCORE_TOLERANCE = 1e-6


def load_queries() -> list[str]:
    """Load retrieval development queries."""

    queries = []

    with EVAL_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                item = json.loads(line)
                queries.append(item["query"])

    return queries


@pytest.mark.integration
def test_llamaindex_retriever_preserves_manual_ranking() -> None:
    """
    Verify that the LlamaIndex adapter preserves ranking and scores
    from the project's manual RetrievalPipeline.
    """

    manual_retriever = RetrievalPipeline(
        rrf_k=60,
        candidate_k=25,
        rerank_k=5,
    )

    llama_retriever = LlamaIndexHybridRetriever(
        rrf_k=60,
        candidate_k=25,
        rerank_k=5,
    )

    for query in load_queries():
        manual_results = manual_retriever.retrieve(
            query=query,
        )

        llama_results = llama_retriever.retrieve(
            QueryBundle(
                query_str=query,
            )
        )

        manual_ids = [
            result["chunk_id"]
            for result in manual_results
        ]

        llama_ids = [
            result.node.node_id
            for result in llama_results
        ]

        assert llama_ids == manual_ids

        for manual_result, llama_result in zip(
            manual_results,
            llama_results,
        ):
            manual_score = manual_result.get(
                "reranker_score",
                manual_result.get("score"),
            )

            assert abs(
                manual_score - llama_result.score
            ) <= SCORE_TOLERANCE