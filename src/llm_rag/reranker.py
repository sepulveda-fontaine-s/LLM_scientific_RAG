from pathlib import Path

from sentence_transformers import CrossEncoder


# Project root:
# Advanced_LLM_RAG/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "rerankers"
    / "ms-marco-MiniLM-L-6-v2"
)


class Reranker:
    """
    Re-rank retrieved candidates using a cross-encoder.

    Unlike the dense retriever, which embeds query and chunk
    separately, the cross-encoder processes the pair together:

        [query, chunk] -> Transformer -> relevance score

    This is more computationally expensive, so it is applied only
    to a small candidate set returned by the retriever.
    """

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
    ) -> None:
        # Load the reranking model from local disk.
        self.model = CrossEncoder(str(model_path))

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Re-rank candidate chunks according to cross-encoder relevance.

        Parameters
        ----------
        query:
            User query.

        candidates:
            Candidate chunks returned by the hybrid retriever.

        top_k:
            Number of final reranked chunks to return.
        """

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        if not candidates:
            return []

        # Build query-chunk pairs for the cross-encoder.
        pairs = [
            (query, candidate["text"])
            for candidate in candidates
        ]

        # Compute one relevance score per candidate.
        scores = self.model.predict(pairs)

        reranked = []

        for candidate, score in zip(candidates, scores):
            result = candidate.copy()
            result["reranker_score"] = float(score)
            reranked.append(result)

        # Sort from most relevant to least relevant.
        reranked.sort(
            key=lambda item: item["reranker_score"],
            reverse=True,
        )

        return reranked[:top_k]