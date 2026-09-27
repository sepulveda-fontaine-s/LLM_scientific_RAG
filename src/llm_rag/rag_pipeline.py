import re

from llm_rag.citation_validator import validate_citations
from llm_rag.numeric_validator import validate_numeric_grounding
from llm_rag.generator import Generator
from llm_rag.groundedness_validator import GroundednessValidator
from llm_rag.retrieval_pipeline import RetrievalPipeline


class RAGPipeline:
    """End-to-end RAG pipeline: retrieve, rerank, build context, generate."""

    def __init__(
        self,
        rrf_k: int = 60,
        candidate_k: int = 25,
        rerank_k: int = 5,
    ):
        # Store retrieval configuration, but build the retrieval stack lazily.
        # This avoids loading a duplicate retriever when another framework
        # (for example LlamaIndex) supplies already-retrieved results.
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k
        self.rerank_k = rerank_k
        self.retriever: RetrievalPipeline | None = None

        self.generator = Generator()
        self.groundedness_validator = GroundednessValidator()


    def _get_retriever(self) -> RetrievalPipeline:
        """
        Build the manual retrieval pipeline only when it is actually needed.

        This keeps the original manual RAG path unchanged while allowing
        external retrievers, such as a LlamaIndex adapter, to inject results
        without loading a second dense/sparse/reranking stack.
        """
        if self.retriever is None:
            self.retriever = RetrievalPipeline(
                rrf_k=self.rrf_k,
                candidate_k=self.candidate_k,
                rerank_k=self.rerank_k,
            )

        return self.retriever

    @staticmethod
    def _sanitize_generation_text(text: str) -> str:
        """
        Remove bibliographic-style numeric citations from source text only
        for the generation prompt.

        Examples removed:
            [88]
            [2, 27, 28]
            [15-18]

        The original chunk text is preserved elsewhere for validators,
        provenance, debugging, and user-visible sources.
        """
        return re.sub(
            r"\[(?:\d+(?:\s*,\s*\d+)*)(?:\s*[-–]\s*\d+)?\]",
            "",
            text,
        )

    def _build_context(self, results: list[dict]) -> str:
        """
        Format retrieved chunks with unambiguous RAG source labels.

        Bibliographic references inside paper text are removed only from
        the generation context so they cannot be confused with RAG citations.
        """
        context_blocks = []

        for index, result in enumerate(results, start=1):
            clean_text = self._sanitize_generation_text(result["text"])

            context_blocks.append(
                f"""RAG_SOURCE=[{index}]
Source: {result["source"]}
Page: {result["page"]}
Text:
{clean_text}"""
            )

        return "\n\n".join(context_blocks)

    def _generate(
        self,
        prompt: str,
        max_new_tokens: int,
    ) -> str:
        """Generate one deterministic candidate answer."""
        return self.generator.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

    def answer(
        self,
        query: str,
        max_new_tokens: int = 256,
        provided_results: list[dict] | None = None,
    ) -> dict:
        """Retrieve evidence and generate a grounded answer with citations."""

        # Normal mode retrieves evidence automatically.
        # Oracle evaluation can inject manually annotated evidence.
        if provided_results is None:
            results = self._get_retriever().retrieve(query)
        else:
            results = provided_results

        context = self._build_context(results)

        prompt = f"""
Answer the question strictly from the provided context.

CITATION RULES:
- Cite RAG sources using square brackets exactly as shown in RAG_SOURCE, for example [1] or [2].
- Use only citation numbers from RAG_SOURCE labels that exist in the provided context.
- Every factual sentence must contain at least one RAG citation before its final punctuation.
- Do not invent citation numbers.
- Do not mention the text "RAG_SOURCE" in the final answer.

GROUNDING RULES:
- Use only facts explicitly stated in the context.
- Preserve numerical quantities exactly as stated in the supporting source.
- Do not infer, reverse, or invent numerical comparisons.
- Use the citation of the source that actually supports each claim.
- If the context does not support a claim, omit that claim.
- Answer in 2-4 concise factual sentences.

Example output:
ColBERT uses late interaction [2].
Its document representations can be precomputed offline [1].

Question:
{query}

Context:
{context}

Answer:
"""

        raw_answer = self._generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        answer = raw_answer

        citations_valid = validate_citations(
            answer=answer,
            num_sources=len(results),
        )

        # One bounded retry for citation-format/compliance failures.
        if not citations_valid:
            retry_prompt = f"""
Rewrite the answer using only the provided context.

The previous answer failed the citation contract.

STRICT CITATION RULES:
- Write 2-4 concise factual sentences.
- EVERY factual sentence must include at least one citation such as [1] before its final punctuation.
- Use ONLY numbers that appear in RAG_SOURCE labels in the context.
- Do not invent citation numbers.
- Do not omit citations.
- Do not mention RAG_SOURCE in the final answer.
- Return only the corrected answer.

GROUNDING RULES:
- Every cited source must directly support the claim in that sentence.
- Preserve numerical quantities exactly as written in the supporting source.
- Do not infer, reverse, or invent numerical relationships.
- Omit unsupported claims.

Question:
{query}

Context:
{context}

Answer:
"""

            raw_answer = self._generate(
                prompt=retry_prompt,
                max_new_tokens=max_new_tokens,
            )
            answer = raw_answer

            citations_valid = validate_citations(
                answer=answer,
                num_sources=len(results),
            )

        sources = [
            {
                "citation": f"[{index}]",
                "source": result["source"],
                "page": result["page"],
                "chunk_id": result["chunk_id"],
                "reranker_score": result.get("reranker_score"),
                "text": result["text"],
            }
            for index, result in enumerate(results, start=1)
        ]

        groundedness = self.groundedness_validator.validate_answer(
            answer=answer,
            sources=sources,
        )

        numeric_grounding_valid = validate_numeric_grounding(
            answer=answer,
            sources=sources,
        )

        # One bounded correction attempt for deterministic grounding failures.
        if not citations_valid or not numeric_grounding_valid:
            correction_prompt = f"""
Rewrite the answer so that it passes all deterministic grounding checks.

STRICT CITATION RULES:
- Write 2-4 concise factual sentences.
- EVERY factual sentence must include at least one citation such as [1] before its final punctuation.
- Use ONLY numbers that appear in RAG_SOURCE labels in the context.
- Do not invent citation numbers.
- Do not omit citations.
- Do not mention RAG_SOURCE in the final answer.

STRICT GROUNDING RULES:
- Use only facts explicitly supported by the context.
- Every citation must directly support the claim in that sentence.
- Every explicit number must appear in a source cited by that sentence.
- Preserve measurements and comparisons exactly.
- Do not infer, reverse, or invent numerical relationships.
- Omit unsupported claims.
- Return only the corrected answer.

Question:
{query}

Context:
{context}

Answer:
"""

            raw_answer = self._generate(
                prompt=correction_prompt,
                max_new_tokens=max_new_tokens,
            )
            answer = raw_answer

            citations_valid = validate_citations(
                answer=answer,
                num_sources=len(results),
            )

            numeric_grounding_valid = validate_numeric_grounding(
                answer=answer,
                sources=sources,
            )

            groundedness = self.groundedness_validator.validate_answer(
                answer=answer,
                sources=sources,
            )

        hard_guardrails_passed = (
            citations_valid
            and numeric_grounding_valid
        )

        # Preserve the final candidate for evaluation/debugging before fail-closed.
        answer_before_fail_closed = answer
        raw_answer_before_formatting = raw_answer

        # Fail closed: never expose an answer that failed deterministic checks.
        if not hard_guardrails_passed:
            answer = (
                "The generated answer could not be validated against "
                "the retrieved evidence."
            )

        if not hard_guardrails_passed:
            validation_status = "rejected"
        elif groundedness["grounded"]:
            validation_status = "validated"
        else:
            validation_status = "validated_with_soft_warning"

        return {
            "query": query,
            "answer": answer,
            "answer_before_fail_closed": answer_before_fail_closed,
            "raw_answer_before_formatting": raw_answer_before_formatting,
            "citations_valid": citations_valid,
            "numeric_grounding_valid": numeric_grounding_valid,
            "grounded": groundedness["grounded"],
            "groundedness_details": groundedness["sentence_results"],
            "sources": sources,
            "hard_guardrails_passed": hard_guardrails_passed,
            "validation_status": validation_status,
        }
