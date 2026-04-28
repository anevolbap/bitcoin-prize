"""Jump table construction for Pollard's Kangaroo (r-adding walks).

The walk's birthday constant c — defined by `expected_ops = c * sqrt(N)` — is
governed by the *distribution* of jump sizes, not just their mean. Two
constructions:

  - powers_of_two: {2^0, 2^1, ..., 2^(r-1)}. Heavily skewed; the largest jump
    dominates. Empirical c ≈ 2.08 (Pollard's classical value).

  - teske: r distinct integer scalars sampled in a tight band around
    sqrt(N)/2, fixed-seed deterministic. Lower variance → c moves toward the
    theoretical lower bound for r-adding walks (Teske 2001 reports c → 1.53
    for properly tuned jump sets; the absolute floor for any random walk is
    sqrt(pi/2) ≈ 1.253).

Our `teske_jumps` is a deliberately simple "Teske-flavored" construction
(uniform sampling in [target/2, 3*target/2]). It's not a fully optimized
r-adding walk per Teske's paper — see test_jump_table.py for the empirically
measured c.
"""
import math
import random


def powers_of_two_jumps(jump_count: int) -> list[int]:
    """Baseline jump set {2^0, 2^1, ..., 2^(r-1)}. Mean = (2^r - 1)/r."""
    if jump_count <= 0:
        raise ValueError(f"jump_count must be positive, got {jump_count}")
    return [1 << i for i in range(jump_count)]


def teske_jumps(interval_width: int, jump_count: int, seed: int = 0xC0FFEE) -> list[int]:
    """r distinct positive scalars with mean ≈ sqrt(N)/2 and reduced variance.

    Sampled uniformly from [target/2, 3*target/2] where target = sqrt(N)/2.
    Deterministic given (interval_width, jump_count, seed).
    """
    if interval_width <= 0:
        raise ValueError(f"interval_width must be positive, got {interval_width}")
    if jump_count <= 0:
        raise ValueError(f"jump_count must be positive, got {jump_count}")

    target = max(jump_count, math.isqrt(interval_width) // 2)
    half_band = max(jump_count, target // 2)
    low = max(1, target - half_band)
    high = target + half_band  # inclusive upper bound below

    if high - low + 1 < jump_count:
        raise ValueError(
            f"sampling band [{low}, {high}] too narrow for {jump_count} "
            "distinct integers"
        )

    rng = random.Random(seed)
    seen: set[int] = set()
    while len(seen) < jump_count:
        seen.add(rng.randrange(low, high + 1))
    return sorted(seen)


def jump_stats(jumps: list[int]) -> dict:
    """Summary statistics for a jump set."""
    n = len(jumps)
    if n == 0:
        raise ValueError("empty jump set")
    s = sum(jumps)
    mean = s / n
    var = sum((j - mean) ** 2 for j in jumps) / n
    sd = math.sqrt(var)
    return {
        "count": n,
        "sum": s,
        "mean": mean,
        "variance": var,
        "stddev": sd,
        "cov": sd / mean if mean else float("inf"),  # coefficient of variation
        "min": min(jumps),
        "max": max(jumps),
    }
