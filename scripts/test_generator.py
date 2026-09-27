from llm_rag.generator import Generator


def main() -> None:
    generator = Generator()

    prompt = """
Explain in 2-3 sentences what Retrieval-Augmented Generation (RAG) is.
"""

    answer = generator.generate(
        prompt=prompt,
        max_new_tokens=128,
    )

    print("Prompt:")
    print(prompt.strip())

    print("\nGenerated answer:")
    print(answer)


if __name__ == "__main__":
    main()