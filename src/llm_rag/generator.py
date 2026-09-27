from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "generation"
    / "Qwen2.5-3B-Instruct"
)


class Generator:
    """Local LLM generator using Qwen2.5-3B-Instruct."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device != "cuda":
            raise RuntimeError(
                "CUDA is required for this generator in the cluster environment."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            local_files_only=True,
        )

        self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
    ) -> str:
        """Generate a response from a plain-text prompt."""

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a technical assistant. "
                    "Answer using only the supplied context."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # Remove the input tokens so we decode only the generated answer.
        generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]

        return self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()