import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


# Project root:
# Advanced_LLM_RAG/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CHUNKS_FILE = (
    PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
)


class SparseRetriever:
    """
    Retrieve chunks using BM25 lexical matching.

    Unlike dense retrieval, BM25 does not use embeddings.
    It scores documents based on term overlap, term frequency,
    and inverse document frequency.
    """

    def __init__(
        self,
        chunks_file: Path = DEFAULT_CHUNKS_FILE,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:

        # Load chunks from disk.
        self.chunks = self._load_chunks(chunks_file)

        # Tokenize every chunk once.
        self.tokenized_corpus = [
            self._tokenize(chunk["text"])
            for chunk in self.chunks
        ]

        # Build the BM25 index with explicit parameters.
        #
        # k1 controls term-frequency saturation.
        # b controls document-length normalization.
        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            k1=k1,
            b=b,
        )

    @staticmethod
    def _load_chunks(chunks_file: Path) -> list[dict]:
        """
        Load chunks from JSONL.
        """

        chunks = []

        with chunks_file.open("r", encoding="utf-8") as file:
            for line in file:
                chunks.append(json.loads(line))

        return chunks

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Very simple tokenizer for the BM25 baseline.

        Steps:
        - lowercase
        - keep alphanumeric word tokens
        """

        return re.findall(r"\b\w+\b", text.lower())

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Retrieve the top-k chunks according to BM25 score.
        """

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        # Tokenize the query using the same logic as the corpus.
        tokenized_query = self._tokenize(query)

        # Compute one BM25 score for every chunk.
        scores = self.bm25.get_scores(tokenized_query)

        # Sort chunk indices from highest score to lowest.
        top_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        results = []

        for index in top_indices:
            chunk = self.chunks[index].copy()
            chunk["score"] = float(scores[index])
            results.append(chunk)

        return results