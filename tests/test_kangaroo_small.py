"""End-to-end correctness for the CPU reference kangaroo on small puzzles.

Synthetic puzzles in the bit-width brackets of Bitcoin puzzles #20/#25/#30.
Synthetic over canonical because the published-answer pages weren't reachable
from this environment. The math the test exercises is identical: recover d
such that d*G = Q with d in [k1, k2). Whether Q comes from privatekeys.pw or
from `random.Random(seed)` doesn't change what's being validated. Canonical Q
values can drop in alongside if obtained later.
"""
import logging
import math
import random

import pytest

from kangaroo.curve import G, scalar_mult
from kangaroo.kangaroo import (
    default_dp_bits, default_jump_count, mean_jump, solve,
)

log = logging.getLogger(__name__)

PUZZLES = [(20, 20), (25, 25), (30, 30)]


def _make_puzzle(bits: int, seed: int):
    """Synthetic puzzle: random d in [2^(bits-1), 2^bits), Q = d*G."""
    rng = random.Random(seed)
    k1 = 1 << (bits - 1)
    k2 = 1 << bits
    d = rng.randrange(k1, k2)
    Q = scalar_mult(d, G)
    return d, Q, k1, k2


@pytest.mark.parametrize("bits,seed", PUZZLES)
def test_jump_table_sized_for_interval(bits, seed):
    """Default jump table has mean_jump within a factor of 4 of sqrt(N)/2.

    Fast precondition (milliseconds) — catches table misconfiguration before
    we waste minutes on a hung solve.
    """
    _, _, k1, k2 = _make_puzzle(bits, seed)
    interval = k2 - k1
    r = default_jump_count(interval)
    dp = default_dp_bits(interval)
    m = mean_jump(r)
    target = math.sqrt(interval) / 2
    ratio = m / target
    log.info(
        "bits=%d interval=%d r=%d dp_bits=%d mean_jump=%.3g target=%.3g ratio=%.3f",
        bits, interval, r, dp, m, target, ratio,
    )
    assert 0.25 < ratio < 4.0, (
        f"jump table miscalibrated: mean_jump={m:.0f}, sqrt(N)/2={target:.0f}, "
        f"ratio={ratio:.2f} (must be in [0.25, 4])"
    )


@pytest.mark.parametrize("bits,seed", PUZZLES)
def test_solve_recovers_key(bits, seed):
    """Solver finds d such that d*G == Q, d ∈ [k1, k2), and d matches the seed."""
    d_known, Q, k1, k2 = _make_puzzle(bits, seed)
    log.info("puzzle bits=%d d=0x%x", bits, d_known)

    interval = k2 - k1
    # Expected iterations ≈ sqrt(N) (each iter steps both walkers once).
    # Cap at 20× to absorb variance; hitting it would indicate a real bug.
    expected_iters = int(math.sqrt(interval))
    max_steps = 20 * expected_iters

    found = solve(Q, k1, k2, max_steps=max_steps)

    assert found is not None, f"solver hit max_steps={max_steps}"
    assert found == d_known, f"got 0x{found:x}, expected 0x{d_known:x}"
    assert k1 <= found < k2, f"d=0x{found:x} outside [0x{k1:x}, 0x{k2:x})"
    assert scalar_mult(found, G) == Q, "d*G != Q"
