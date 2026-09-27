from pathlib import Path
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer


# Project root:
# Advanced_LLM_RAG/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Local path of the embedding model downloaded from Hugging Face.
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "embeddings"
    / "bge-small-en-v1.5"
)


class Embedder:
    """
    Small wrapper around SentenceTransformer.

    Responsibilities:
    - load the embedding model from local disk;
    - encode one or more texts;
    - optionally normalize the embeddings.
    """

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_path = model_path
        self.normalize_embeddings = normalize_embeddings

        # Load the model from local storage.
        self.model = SentenceTransformer(str(model_path))

    def encode(
        self,
        texts: Iterable[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Convert texts into dense embedding vectors.

        Parameters
        ----------
        texts:
            Iterable of input strings.

        batch_size:
            Number of texts processed together.

        Returns
        -------
        np.ndarray
            Matrix with shape:
            (number_of_texts, embedding_dimension)
        """

        texts = list(texts)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        return embeddings