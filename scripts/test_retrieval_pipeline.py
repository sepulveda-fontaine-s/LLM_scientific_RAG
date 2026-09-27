from llm_rag.retrieval_pipeline import RetrievalPipeline


def main() -> None:
    """
    Test the complete retrieval pipeline with one query.

    Flow:
        query
          ↓
        hybrid retrieval
          ↓
        reranking
          ↓
        final top-k chunks
    """

    query = (
        "How does ColBERT reduce retrieval latency "
        "compared with BERT-based reranking?"
    )

    # Build the complete retrieval pipeline.
    pipeline = RetrievalPipeline(
        rrf_k=60,
        candidate_k=15,
        rerank_k=5,
    )

    # Retrieve the final ranked chunks.
    results = pipeline.retrieve(query)

    print(f"\nQuery: {query}\n")

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