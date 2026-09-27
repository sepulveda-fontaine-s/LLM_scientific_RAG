import math


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Fraction of all relevant chunks retrieved in the top-k."""
    if not relevant_ids:
        return 0.0

    retrieved_at_k = set(retrieved_ids[:k])
    relevant_retrieved = retrieved_at_k & relevant_ids

    return len(relevant_retrieved) / len(relevant_ids)


def reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> float:
    """Reciprocal rank of the first relevant retrieved chunk."""
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Normalized Discounted Cumulative Gain with binary relevance."""

    dcg = 0.0

    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        relevance = 1.0 if chunk_id in relevant_ids else 0.0

        dcg += relevance / math.log2(rank + 1)

    ideal_relevant_count = min(len(relevant_ids), k)

    if ideal_relevant_count == 0:
        return 0.0

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_relevant_count + 1)
    )

    return dcg / idcg