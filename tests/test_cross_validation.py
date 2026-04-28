"""Layer 6 — cross-validate the CPU reference against JeanLucPons/Kangaroo.

Both implementations must recover the same key on a custom-generated puzzle.
JLP is a separate C++/CUDA project; this test skips locally when its binary
isn't present, with a message pointing at $JLP_KANGAROO_BIN. The intended
runtime environment is Colab (CUDA build available there).

Bit-width is capped at 40 here because our pure-Python single-walker reference
doesn't reach 50-bit in test time. Layer 7 will add 50/60/70-bit cases to this
same file, comparing our CUDA kernel vs JLP-CUDA — the structure below is
designed for those rows to slot in alongside without rework.

Op-count parity (the spec's "within 2× of each other") is intentionally NOT
asserted: JLP runs ~1024 walkers per CPU thread by default, ours runs 1, so
absolute step counts differ by ~3 orders of magnitude even when both
implementations are correct. Group-op totals are roughly equal, but JLP
doesn't expose a stable, version-independent total to scrape from stdout.
Wall times and our step count are logged for visibility; the load-bearing
assertion is key agreement.
"""
import logging
import math
import random
import time

import pytest

import jlp_runner

from kangaroo.curve import G, scalar_mult
from kangaroo.kangaroo import default_dp_bits, solve

log = logging.getLogger(__name__)

pytestmark = pytest.mark.slow

PUZZLES = [(32, 32), (36, 36), (40, 40)]


def _make_puzzle(bits: int, seed: int):
    rng = random.Random(seed)
    k1 = 1 << (bits - 1)
    k2 = 1 << bits
    d = rng.randrange(k1, k2)
    Q = scalar_mult(d, G)
    assert Q is not None
    return d, Q, k1, k2


@pytest.fixture(scope="module")
def jlp_bin():
    path = jlp_runner.find_binary()
    if path is None:
        pytest.skip(
            "JeanLucPons binary not found. Set $JLP_KANGAROO_BIN, put one on "
            "PATH, or build into tools/Kangaroo/. Test is intended for Colab "
            "where the CUDA build is available."
        )
    log.info("using JLP binary: %s", path)
    return path


@pytest.mark.parametrize("bits,seed", PUZZLES)
def test_cross_validate(jlp_bin, bits, seed):
    d_known, Q, k1, k2 = _make_puzzle(bits, seed)
    interval = k2 - k1
    log.info("puzzle bits=%d d=0x%x", bits, d_known)

    # ---- our CPU reference ----
    expected_iters = int(2.08 * math.sqrt(interval))
    max_steps = 20 * expected_iters
    stats: dict = {}
    t0 = time.monotonic()
    our_d = solve(
        Q, k1, k2,
        dp_bits=default_dp_bits(interval),
        max_steps=max_steps,
        stats=stats,
    )
    our_elapsed = time.monotonic() - t0
    assert our_d is not None, f"our solver hit max_steps={max_steps}"
    assert scalar_mult(our_d, G) == Q, "our recovered key fails d*G == Q"
    assert our_d == d_known, f"ours wrong: 0x{our_d:x} != 0x{d_known:x}"
    log.info(
        "ours: d=0x%x in %.2fs, iters=%d (≈ %.2g group ops, theoretical=%.2g)",
        our_d, our_elapsed, stats["steps"],
        2 * stats["steps"], 2.08 * math.sqrt(interval),
    )

    # ---- JLP ----
    # Don't pin -d; let JLP choose its own dp_bits for its many-walker model.
    # Generous timeout — JLP-CPU on a 40-bit puzzle finishes in ~seconds, but
    # initial-startup overhead on cold Colab can spike the first run.
    t0 = time.monotonic()
    res = jlp_runner.run(jlp_bin, k1, k2, Q, timeout=300.0)
    jlp_elapsed = time.monotonic() - t0
    log.info("jlp: d=0x%x in %.2fs (rc=%d)", res["key"], jlp_elapsed, res["returncode"])

    # ---- agreement ----
    assert res["key"] == d_known, f"JLP wrong: 0x{res['key']:x} != 0x{d_known:x}"
    assert res["key"] == our_d, "JLP and our reference disagree"
