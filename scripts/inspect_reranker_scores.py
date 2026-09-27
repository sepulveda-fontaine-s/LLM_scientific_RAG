import json
from pathlib import Path

from llm_rag.retrieval_pipeline import RetrievalPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval_eval.jsonl"
)

CANDIDATE_K = 15


TARGET_QUERIES = {
    "How is a retriever evaluated using top-k retrieval accuracy in open-domain question answering?",
    "How does Sentence-BERT evaluate sentence embeddings on semantic textual similarity tasks?",
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def main() -> None:
    evaluation_cases = load_jsonl(EVAL_PATH)

    pipeline = RetrievalPipeline(
        rrf_k=60,
        candidate_k=CANDIDATE_K,
        rerank_k=CANDIDATE_K,
    )

    for case in evaluation_cases:
        query = case["query"]

        if query not in TARGET_QUERIES:
            continue

        relevant_ids = set(case["relevant_chunk_ids"])

        candidates = pipeline.hybrid_retriever.retrieve(
            query=query,
            top_k=CANDIDATE_K,
            candidate_k=CANDIDATE_K,
        )

        reranked = pipeline.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=CANDIDATE_K,
        )

        print("\n" + "=" * 100)
        print("QUERY:", query)

        for rank, result in enumerate(reranked, start=1):
            marker = (
                "RELEVANT"
                if result["chunk_id"] in relevant_ids
                else ""
            )

            print(
                f"{rank:2d}. "
                f"score={result['reranker_score']:.6f} "
                f"{marker}"
            )

            print(
                "    ",
                repr(result["chunk_id"]),
            )


if __name__ == "__main__":
    main()