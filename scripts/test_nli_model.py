from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "validators" / "nli-deberta-v3-small"


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print("Labels:", model.config.id2label)

    premise = (
        "ColBERT's query encoding and interaction consume only "
        "13 milliseconds of its total execution time."
    )

    hypotheses = [
        "ColBERT query encoding and interaction take 13 milliseconds.",
        "ColBERT query encoding and interaction take 100 milliseconds.",
    ]

    for hypothesis in hypotheses:
        inputs = tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
        ).to(device)

        with torch.inference_mode():
            logits = model(**inputs).logits[0]
            probabilities = torch.softmax(logits, dim=-1)

        print("\nHypothesis:", hypothesis)

        for index, probability in enumerate(probabilities):
            label = model.config.id2label[index]
            print(f"{label}: {probability.item():.4f}")


if __name__ == "__main__":
    main()