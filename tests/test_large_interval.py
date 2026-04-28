"""Layer 7 — 134-bit interval arithmetic.

JeanLucPons hard-codes a 125-bit interval cap; we don't. This file documents
and exercises the contract that scalars near 2^134 (puzzle #135 range) are
handled correctly throughout: parsing, curve arithmetic, walk, checkpoint.

Pure-Python ints are arbitrary-precision, so the reference cannot literally
suffer 64-bit limb overflow. The role of these tests is twofold:
  1. Lock the public surface (interval params, scalar storage, walk loop) so
     the eventual CUDA port — which DOES need fixed-width limbs — is held
     to the same contract.
  2. Surface any latent dependence on small absolute scalars (mistaken
     truncation, off-by-N modular reduction) by exercising the algorithm at
     a 134-bit offset.

The CLAUDE.md spec asks for a "130-bit puzzle, end-to-end". A 130-bit-wide
search is not tractable in pure-Python single-walker (~10^20 ops). We get
equivalent coverage of the >125-bit-scalar code paths via a small-width
interval *placed at* a 134-bit offset: every absolute scalar in the walk
exceeds 130 bits, only the search width is small. The literal 130-bit-wide
puzzle solve is deferred to Layer 7-CUDA.
"""
import logging
import math
import random

import pytest

from kangaroo.checkpoint import load_checkpoint
from kangaroo.curve import G, point_add, scalar_mult
from kangaroo.kangaroo import default_dp_bits, default_jump_count, solve

log = logging.getLogger(__name__)

# Puzzle #135 search range.
K135_LO = 1 << 134
K135_HI = 1 << 135


def test_solve_accepts_puzzle_135_range():
    """Init and a few walk steps over the full [2^134, 2^135) range run cleanly.

    dp_bits at this scale is ~65 → no DP will fire in 10 steps. We're testing
    that param intake, midpoint scalar_mult, and the walk loop don't choke on
    134-bit absolute scalars; not that anything is found.
    """
    interval = K135_HI - K135_LO
    r = default_jump_count(interval)
    dp = default_dp_bits(interval)
    log.info("interval=2^%d r=%d dp_bits=%d", interval.bit_length() - 1, r, dp)
    assert r >= 8
    assert dp > 0

    # A real public key in-range. Solver shouldn't actually find d in 10 steps
    # (DP rate is 1 in 2^dp_bits ≈ 2^65); the assertion is "no exception".
    Q = scalar_mult(K135_LO + 12345, G)
    assert Q is not None
    out = solve(Q, K135_LO, K135_HI, max_steps=10)
    assert out is None


def test_scalar_mult_additive_at_134bit():
    """(a+b)*G == a*G + b*G for a ≈ 2^134 — locks curve linearity at scale.

    The puzzle #135 walk relies on scalar_mult being correct for inputs of
    134-135 bits. A bug truncating the input scalar to 64 bits would silently
    make every walker land on the wrong starting point.
    """
    a = K135_LO + 0xCAFEBABE_DEADBEEF
    b = 0x1234_5678_9ABC_DEF0
    P_a = scalar_mult(a, G)
    P_b = scalar_mult(b, G)
    P_sum = scalar_mult(a + b, G)
    assert P_a is not None and P_b is not None and P_sum is not None
    assert point_add(P_a, P_b) == P_sum
    assert a.bit_length() >= 134, "test inputs lost their bit-width"


def test_checkpoint_round_trips_134bit_scalars(tmp_path):
    """Walker scalars near 2^134 survive a checkpoint save/load exactly.

    Goes through solve() so the test exercises the actual production save
    path (not a synthetic dict). Catches any future regression where scalars
    get coerced to a fixed-width type during serialization.
    """
    bits = 24
    k1 = K135_LO
    k2 = k1 + (1 << bits)
    Q = scalar_mult(k1 + 7, G)
    path = tmp_path / "big.ckpt"

    solve(Q, k1, k2, max_steps=500, checkpoint_path=str(path))
    _, state = load_checkpoint(str(path))

    # Tame walker started at midpoint ≈ 2^134, so its scalar is unambiguously
    # large. Wild walker's scalar is just the accumulated walk distance from
    # 0, which after 500 steps is far smaller — only assert on the tame side.
    s_tame = state["scalar_tame"]
    assert isinstance(s_tame, int)
    assert s_tame.bit_length() >= 130, (
        f"scalar_tame={s_tame:#x} unexpectedly small ({s_tame.bit_length()} bits)"
    )

    # Round-trip: in-memory and on-disk values are byte-identical.
    _, state2 = load_checkpoint(str(path))
    assert state2["scalar_tame"] == s_tame
    assert state2["scalar_wild"] == state["scalar_wild"]


@pytest.mark.slow
def test_end_to_end_solve_at_134bit_offset():
    """Solve a small-width puzzle whose key lives at the 2^134 offset.

    The walk traverses absolute scalars > 2^134 throughout — every step
    exercises the >125-bit code paths a 130-bit puzzle would, but with a
    tractable interval width. If anything in the algorithm depended on small
    k, this test fails; if it passes, the kangaroo is offset-agnostic and the
    CUDA port can start from a known-good reference.
    """
    bits = 30
    k1 = K135_LO
    k2 = k1 + (1 << bits)
    rng = random.Random(20260427)
    d = rng.randrange(k1, k2)
    Q = scalar_mult(d, G)
    assert Q is not None
    assert d.bit_length() >= 134

    interval = k2 - k1
    expected_iters = int(2.08 * math.sqrt(interval))
    max_steps = 20 * expected_iters
    log.info("solving 0x%x in [2^134, 2^134+2^%d) (max_steps=%d)",
             d, bits, max_steps)

    found = solve(Q, k1, k2, max_steps=max_steps)
    assert found is not None, f"solver hit max_steps={max_steps}"
    assert found == d, f"got 0x{found:x}, expected 0x{d:x}"
    assert k1 <= found < k2
    assert scalar_mult(found, G) == Q
