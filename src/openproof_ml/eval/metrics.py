"""Evaluation metrics for tactic prediction."""

import math


def pass_at_k(n: int, c: int, k: int) -> float:
    """Compute pass@k metric.

    Args:
        n: total number of samples
        c: number of correct samples
        k: k in pass@k

    Returns:
        Estimated probability that at least one of k samples is correct.
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))
