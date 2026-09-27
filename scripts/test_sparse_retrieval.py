from llm_rag.sparse_retriever import SparseRetriever


def main() -> None:
    """
    Run a BM25 retrieval test using the same query
    that we used for dense retrieval.

    This lets us compare lexical retrieval against
    semantic retrieval under the same conditions.
    """

    # Build the BM25 retriever.
    retriever = SparseRetriever(
        k1=1.5,
        b=0.75,
    )

    # Same query used in the dense retrieval test.
    query = (
        "How does ColBERT reduce retrieval latency "
        "compared with BERT-based reranking?"
    )

    # Retrieve the top five chunks.
    results = retriever.retrieve(
        query=query,
        top_k=5,
    )

    print(f"\nQuery: {query}\n")

    # Display the ranked results.
    for rank, result in enumerate(results, start=1):
        print("=" * 80)
        print(f"Rank: {rank}")
        print(f"Score: {result['score']:.4f}")
        print(f"Source: {result['source']}")
        print(f"Page: {result['page']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print("-" * 80)
        print(result["text"])
        print()


if __name__ == "__main__":
    main()