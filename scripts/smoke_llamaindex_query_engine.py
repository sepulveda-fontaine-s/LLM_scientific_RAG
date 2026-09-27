from llm_rag.llamaindex_query_engine import (
    LlamaIndexRAGQueryEngine,
)


QUERY = (
    "How does ColBERT compare with BERT-based rerankers "
    "in latency and computational cost?"
)


def main() -> None:
    engine = LlamaIndexRAGQueryEngine.from_defaults()

    response = engine.query(QUERY)

    print("=" * 100)
    print("LLAMAINDEX QUERY ENGINE SMOKE TEST")
    print("=" * 100)

    print("\nQUERY:")
    print(QUERY)

    print("\nANSWER:")
    print(str(response))

    print("\nSOURCE NODES:")
    print(len(response.source_nodes))

    for index, node in enumerate(
        response.source_nodes,
        start=1,
    ):
        print(
            f"{index}. "
            f"{node.node.metadata['source']} "
            f"| page={node.node.metadata['page']} "
            f"| score={node.score}"
        )

    print("\nVALIDATION:")
    print(
        "Status:",
        response.metadata["validation_status"],
    )
    print(
        "Hard guardrails:",
        response.metadata["hard_guardrails_passed"],
    )
    print(
        "Citations valid:",
        response.metadata["citations_valid"],
    )
    print(
        "Numeric grounding:",
        response.metadata["numeric_grounding_valid"],
    )
    print(
        "Grounded:",
        response.metadata["grounded"],
    )


if __name__ == "__main__":
    main()