from llm_rag.hybrid_retriever import HybridRetriever


def main() -> None:
    """
    Run hybrid retrieval using the same query used previously
    for dense retrieval and BM25.

    This lets us compare:
    - dense ranking
    - sparse/BM25 ranking
    - fused RRF ranking
    """

    # Build the hybrid retriever.
    retriever = HybridRetriever(
        rrf_k=60,
    )

    # Same query used in previous tests.
    query = (
        "How does ColBERT reduce retrieval latency "
        "compared with BERT-based reranking?"
    )

    # Retrieve final hybrid results.
    results = retriever.retrieve(
        query=query,
        top_k=5,
        candidate_k=10,
    )

    print(f"\nQuery: {query}\n")

    # Display the final fused ranking.
    for rank, result in enumerate(results, start=1):
        print("=" * 80)
        print(f"Rank: {rank}")
        print(f"RRF score: {result['rrf_score']:.6f}")
        print(f"Source: {result['source']}")
        print(f"Page: {result['page']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print("-" * 80)
        print(result["text"])
        print()


if __name__ == "__main__":
    main()