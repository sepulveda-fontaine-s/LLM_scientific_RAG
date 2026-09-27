import re
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "validators"
    / "nli-deberta-v3-small"
)


class GroundednessValidator:
    """Validate whether cited sources entail generated factual sentences."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        entailment_threshold: float = 0.70,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.entailment_threshold = entailment_threshold

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            local_files_only=True,
        )

        self.model.to(self.device)
        self.model.eval()

        # Resolve label index from model metadata instead of assuming its position.
        self.entailment_index = next(
            index
            for index, label in self.model.config.id2label.items()
            if label.lower() == "entailment"
        )

        self.contradiction_index = next(
            index
            for index, label in self.model.config.id2label.items()
            if label.lower() == "contradiction"
        )

        self.neutral_index = next(
            index
            for index, label in self.model.config.id2label.items()
            if label.lower() == "neutral"
        )

    def validate_sentence(
        self,
        sentence: str,
        premise: str,
    ) -> float:
        """Return entailment probability for one sentence/source pair."""

        inputs = self.tokenizer(
            premise,
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.inference_mode():
            logits = self.model(**inputs).logits[0]
            probabilities = torch.softmax(logits, dim=-1)

        return {
            "entailment": probabilities[self.entailment_index].item(),
            "contradiction": probabilities[self.contradiction_index].item(),
            "neutral": probabilities[self.neutral_index].item(),
        }

    
    def _build_premise_windows(
        self,
        text: str,
        window_size: int = 2,
    ) -> list[str]:
        """
        Split source text into small consecutive sentence windows.

        Small windows reduce irrelevant context for the NLI model
        while preserving nearby evidence that may span two sentences.
        """

        sentences = [
            sentence.strip()
            for sentence in re.split(
                r"(?<=[.!?])\s+",
                text.strip(),
            )
            if sentence.strip()
        ]

        if not sentences:
            return [text]

        windows = []

        for index in range(len(sentences)):
            window = " ".join(
                sentences[index:index + window_size]
            )

            if window:
                windows.append(window)

        return windows



    def validate_answer(
        self,
        answer: str,
        sources: list[dict],
    ) -> dict:
        """
        Validate every cited sentence against the chunks referenced by its citations.
        """

        sentences = re.split(r"(?<=[.!?])\s+", answer.strip())

        sentence_results = []

        for sentence in sentences:
            if not sentence.strip():
                continue

            citation_numbers = [
                int(value)
                for value in re.findall(r"\[(\d+)\]", sentence)
            ]

            if not citation_numbers:
                sentence_results.append(
                    {
                        "sentence": sentence,
                        "entailment_score": 0.0,
                        "grounded": False,
                    }
                )
                continue

            
            candidate_windows = []

            for citation in sorted(set(citation_numbers)):
                if 1 <= citation <= len(sources):
                    source_text = sources[citation - 1]["text"]

                    candidate_windows.extend(
                        self._build_premise_windows(
                            text=source_text,
                            window_size=2,
                        )
                    )

            # Remove citation markers before using the sentence as NLI hypothesis.
            hypothesis = re.sub(r"\[\d+\]", "", sentence).strip()

            if not candidate_windows:
                sentence_results.append(
                    {
                        "sentence": sentence,
                        "entailment_score": 0.0,
                        "contradiction_score": 0.0,
                        "neutral_score": 1.0,
                        "best_evidence_span": None,
                        "grounded": False,
                    }
                )
                continue

            best_scores = None
            best_window = None

            for window in candidate_windows:
                scores = self.validate_sentence(
                    sentence=hypothesis,
                    premise=window,
                )

                if (
                    best_scores is None
                    or scores["entailment"] > best_scores["entailment"]
                ):
                    best_scores = scores
                    best_window = window
            

            
            sentence_results.append(
                {
                    "sentence": sentence,
                    "entailment_score": best_scores["entailment"],
                    "contradiction_score": best_scores["contradiction"],
                    "neutral_score": best_scores["neutral"],
                    "best_evidence_span": best_window,
                    "grounded": (
                        best_scores["entailment"]
                        >= self.entailment_threshold
                    ),
                }
            )

        return {
            "grounded": all(
                result["grounded"]
                for result in sentence_results
            ),
            "sentence_results": sentence_results,
        }