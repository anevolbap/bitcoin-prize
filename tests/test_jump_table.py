"""Layer 4: Teske r-adding jump table.

Test groups:
  1. Isolation — sum/mean target, distinctness, bounds, determinism, lower
     variance than powers-of-2.
  2. Integration — solver still finds the key on small puzzles when run with
     a Teske jump set.
  3. Empirical birthday constant — measure c on 30-bit puzzles for both
     pow2 and Teske; assert Teske is better. Document the gap to Teske
     theoretical 1.53.
"""
import logging
import math
import random

import pytest

from kangaroo.curve import G, scalar_mult
from kangaroo.jump_table import jump_stats, powers_of_two_jumps, teske_jumps
from kangaroo.kangaroo import default_jump_count, solve

log = logging.getLogger(__name__)


def _make_puzzle(bits: int, seed: int):
    rng = random.Random(seed)
    k1, k2 = 1 << (bits - 1), 1 << bits
    d = rng.randrange(k1, k2)
    return d, scalar_mult(d, G), k1, k2


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bits,r", [(20, 12), (25, 14), (30, 17)])
def test_teske_count_and_distinctness(bits, r):
    interval = (1 << bits) - (1 << (bits - 1))
    js = teske_jumps(interval, r, seed=42)
    assert len(js) == r
    assert len(set(js)) == r
    assert js == sorted(js)


@pytest.mark.parametrize("bits,r", [(20, 12), (25, 14), (30, 17)])
def test_teske_positive_and_bounded(bits, r):
    interval = (1 << bits) - (1 << (bits - 1))
    js = teske_jumps(interval, r, seed=42)
    assert all(1 <= j < interval for j in js)


@pytest.mark.parametrize("bits", [20, 25, 30])
def test_teske_mean_near_target(bits):
    """Mean jump matches the kangaroo balance target sqrt(N)/2 within a factor of 2."""
    interval = (1 << bits) - (1 << (bits - 1))
    r = default_jump_count(interval)
    js = teske_jumps(interval, r, seed=42)
    target = math.sqrt(interval) / 2
    mean = sum(js) / len(js)
    log.info(
        "bits=%d r=%d target=%.0f mean=%.0f ratio=%.2f",
        bits, r, target, mean, mean / target,
    )
    assert 0.5 <= mean / target <= 2.0


@pytest.mark.parametrize("bits", [20, 25, 30])
def test_teske_variance_lower_than_pow2(bits):
    """Coefficient of variation (stddev/mean) is much lower for Teske than powers-of-2."""
    interval = (1 << bits) - (1 << (bits - 1))
    r = default_jump_count(interval)

    p2_stats = jump_stats(powers_of_two_jumps(r))
    tk_stats = jump_stats(teske_jumps(interval, r, seed=42))

    log.info(
        "bits=%d r=%d: pow2 cov=%.2f, teske cov=%.2f",
        bits, r, p2_stats["cov"], tk_stats["cov"],
    )
    assert tk_stats["cov"] < p2_stats["cov"]


def test_teske_deterministic():
    """Same seed → identical jump set; different seed → different set."""
    js1 = teske_jumps(1 << 24, 14, seed=42)
    js2 = teske_jumps(1 << 24, 14, seed=42)
    assert js1 == js2

    js3 = teske_jumps(1 << 24, 14, seed=43)
    assert js1 != js3


def test_teske_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        teske_jumps(0, 14)
    with pytest.raises(ValueError):
        teske_jumps(1 << 20, 0)


# ---------------------------------------------------------------------------
# Integration: solver works with Teske jumps
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bits,seed", [(20, 20), (25, 25), (30, 30)])
def test_teske_solves_small_puzzle(bits, seed):
    """Solver returns the correct d on a small puzzle when given Teske jumps."""
    d, Q, k1, k2 = _make_puzzle(bits, seed)
    interval = k2 - k1
    r = default_jump_count(interval)

    found = solve(
        Q, k1, k2,
        jumps=teske_jumps(interval, r, seed=42),
        max_steps=20 * int(math.sqrt(interval)),
    )
    assert found == d, f"got 0x{found:x}, expected 0x{d:x}"
    assert k1 <= found < k2
    assert scalar_mult(found, G) == Q


# ---------------------------------------------------------------------------
# Empirical birthday constant on 30-bit puzzles
# ---------------------------------------------------------------------------

def _measure_c(bits: int, jumps_factory, trials: int, seed_base: int) -> tuple[float, float]:
    """Solve `trials` puzzles, return (mean_c, mean_ops) where c = ops / sqrt(N)."""
    interval = (1 << bits) - (1 << (bits - 1))
    ops_list = []
    for i in range(trials):
        d, Q, k1, k2 = _make_puzzle(bits, seed_base + i)
        stats = {}
        found = solve(
            Q, k1, k2,
            jumps=jumps_factory(),
            max_steps=80 * int(math.sqrt(interval)),
            stats=stats,
        )
        assert found == d, f"solve failed at seed={seed_base + i}"
        # Each iteration steps both walkers; total point_adds = 2 * steps.
        ops_list.append(stats["steps"] * 2)
    mean_ops = sum(ops_list) / len(ops_list)
    return mean_ops / math.sqrt(interval), mean_ops


def test_empirical_constant_teske_better_than_pow2():
    """Measured c on 30-bit puzzles: Teske should beat powers-of-2.

    Powers-of-2 baseline: theoretical c ≈ 2.08 (Pollard).
    Teske theoretical: c → 1.53 (Teske 2001).
    Floor for any walk: sqrt(pi/2) ≈ 1.253.

    Our `teske_jumps` is a heuristic (uniform sampling, low variance) — not
    a fully tuned Teske walk. We expect c_teske < c_pow2, not necessarily
    1.53.

    Variance is high (~mean per trial); 8 trials gives ~35% stderr. The
    threshold is a one-sided "Teske wins"; logged values document the gap
    between observed c and the theoretical 1.53.
    """
    BITS = 30
    TRIALS = 8
    interval = (1 << BITS) - (1 << (BITS - 1))
    r = default_jump_count(interval)

    pow2 = powers_of_two_jumps(r)
    teske = teske_jumps(interval, r, seed=42)

    log.info(
        "pow2: mean=%.0f cov=%.2f",
        jump_stats(pow2)["mean"], jump_stats(pow2)["cov"],
    )
    log.info(
        "teske: mean=%.0f cov=%.2f",
        jump_stats(teske)["mean"], jump_stats(teske)["cov"],
    )

    c_pow2, ops_pow2 = _measure_c(BITS, lambda: pow2, TRIALS, seed_base=1000)
    c_teske, ops_teske = _measure_c(BITS, lambda: teske, TRIALS, seed_base=1000)

    log.info(
        "%d-bit, %d trials: c_pow2=%.2f, c_teske=%.2f, gain=%.2fx "
        "(theory: 2.08 -> 1.53 = 1.36x)",
        BITS, TRIALS, c_pow2, c_teske, c_pow2 / c_teske,
    )
    log.info(
        "ops: pow2 mean=%.0f, teske mean=%.0f, sqrt(N)=%.0f",
        ops_pow2, ops_teske, math.sqrt(interval),
    )

    assert c_teske < c_pow2, (
        f"Teske did not beat powers-of-2: c_pow2={c_pow2:.2f}, "
        f"c_teske={c_teske:.2f}"
    )
