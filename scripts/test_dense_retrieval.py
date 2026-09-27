from llm_rag.dense_retriever import DenseRetriever


def main() -> None:
    """
    Run a simple semantic-search test against the dense retriever.

    The goal is to verify that a natural-language query retrieves
    chunks that are actually relevant to the topic.
    """

    # Load the retriever.
    # This also loads:
    # - chunks.jsonl
    # - embeddings.npy
    # - the local embedding model
    retriever = DenseRetriever()

    # Test query.
    query = (
        "How does ColBERT reduce retrieval latency "
        "compared with BERT-based reranking?"
    )

    # Retrieve the five most similar chunks.
    results = retriever.retrieve(
        query=query,
        top_k=5,
    )

    print(f"\nQuery: {query}\n")

    # Display the retrieved chunks and their similarity scores.
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