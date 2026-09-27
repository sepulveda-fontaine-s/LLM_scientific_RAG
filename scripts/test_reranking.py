from llm_rag.hybrid_retriever import HybridRetriever
from llm_rag.reranker import Reranker


def main() -> None:
    """
    Test the complete retrieval stack so far:

        query
          ↓
        dense retrieval
          +
        BM25 retrieval
          ↓
        RRF hybrid fusion
          ↓
        cross-encoder reranking

    We use the same query as before so the rankings can be compared.
    """

    query = (
        "How does ColBERT reduce retrieval latency "
        "compared with BERT-based reranking?"
    )

    # First retrieve a broader candidate set using hybrid retrieval.
    hybrid_retriever = HybridRetriever(
        rrf_k=60,
    )

    candidates = hybrid_retriever.retrieve(
        query=query,
        top_k=10,
        candidate_k=15,
    )

    # Then re-score those candidates with the cross-encoder.
    reranker = Reranker()

    results = reranker.rerank(
        query=query,
        candidates=candidates,
        top_k=5,
    )

    print(f"\nQuery: {query}\n")

    # Display the final ranking after reranking.
    for rank, result in enumerate(results, start=1):
        print("=" * 80)
        print(f"Rank: {rank}")
        print(f"RRF score: {result['rrf_score']:.6f}")
        print(f"Reranker score: {result['reranker_score']:.6f}")
        print(f"Source: {result['source']}")
        print(f"Page: {result['page']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print("-" * 80)
        print(result["text"])
        print()


if __name__ == "__main__":
    main()