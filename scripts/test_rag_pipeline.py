from llm_rag.rag_pipeline import RAGPipeline


def main() -> None:
    query = (
        "How does ColBERT reduce retrieval latency "
        "compared with BERT-based reranking?"
    )

    pipeline = RAGPipeline(
        rrf_k=60,
        candidate_k=15,
        rerank_k=5,
    )

    result = pipeline.answer(
        query=query,
        max_new_tokens=256,
    )

    print("Query:")
    print(result["query"])

    print("\nAnswer:")
    print(result["answer"])
    print("\nCitations valid:")
    print(result["citations_valid"])
    print("\nNumeric grounding valid:")
    print(result["numeric_grounding_valid"])
    print("\nGrounded:")
    print(result["grounded"])
    print("\nHard guardrails passed:")
    print(result["hard_guardrails_passed"])
    print("\nValidation status:")
    print(result["validation_status"])

    print("\nGroundedness details:")
    for detail in result["groundedness_details"]:
        print(
            f'- grounded={detail["grounded"]} '
            f'| entailment={detail["entailment_score"]:.4f} '
            f'| contradiction={detail["contradiction_score"]:.4f} '
            f'| neutral={detail["neutral_score"]:.4f}'
        )
        print(f'  {detail["sentence"]}')

    print("\nSources:")
    for source in result["sources"]:
        print(
            f'{source["citation"]} '
            f'{source["source"]} | '
            f'page {source["page"]} | '
            f'reranker={source["reranker_score"]:.6f}'
        )


if __name__ == "__main__":
    main()