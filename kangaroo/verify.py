"""Triple-check a recovered private key before declaring success.

A bug anywhere in the GPU solver pipeline can produce a "recovered key" that
doesn't actually satisfy d*G == Q, or that lies outside the puzzle's interval.
This module is the always-on guard: nothing should declare success without
calling verify_solution().

Per the project's "Critical correctness lesson":
    A buggy Kangaroo does not gracefully degrade — it silently fails forever.

Wrong-key paths to catch:
  - Sign-tracking bug → key returned is n - d (off-curve check would miss it)
  - Off-by-one in interval → d outside [k1, k2)
  - Stale state from checkpoint → d points at a different puzzle's pubkey
"""
from __future__ import annotations

from .curve import G, scalar_mult


def serialize_compressed(point: tuple[int, int]) -> str:
    """SEC1 compressed: 02/03 prefix + 32-byte big-endian x. Lowercase hex."""
    x, y = point
    return ("02" if (y & 1) == 0 else "03") + f"{x:064x}"


def verify_solution(
    d: int, pubkey_hex: str, k1: int, k2: int,
) -> bool:
    """Return True iff d is a valid private key for `pubkey_hex` in [k1, k2).

    Raises ValueError with a specific message on each failure mode so the
    caller can log the *kind* of mismatch — useful when the bug is in our
    pipeline (off-by-one) vs. the solver's (sign-tracking).
    """
    if not (k1 <= d < k2):
        raise ValueError(
            f"d=0x{d:x} outside interval [0x{k1:x}, 0x{k2:x}) "
            f"({d.bit_length()} bits)"
        )
    Q = scalar_mult(d, G)
    if Q is None:
        raise ValueError(f"d=0x{d:x} produces the identity point (d ≡ 0 mod n)")
    derived = serialize_compressed(Q)
    expected = pubkey_hex.lower()
    if derived != expected:
        raise ValueError(
            f"d*G != Q\n  d         = 0x{d:x}\n"
            f"  d*G       = {derived}\n  expected  = {expected}"
        )
    return True
