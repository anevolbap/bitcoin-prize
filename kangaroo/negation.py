"""Negation-map canonicalization: collapse {P, -P} to a single representative.

The canonical representative is the one with the smaller y-coordinate. Halving
the effective walk state space gives a sqrt(2) ≈ 1.41x speedup in expected
time-to-collision.

This module is intentionally self-contained: it only depends on the field
prime P from curve.py. The integration with the kangaroo walk (sign tracking
on flip) lives in kangaroo.kangaroo. Getting the sign tracking wrong yields
n - d_correct instead of d_correct — verification at the end of solve()
catches that.
"""
from .curve import P as FIELD_P


def canonical(point):
    """Return the canonical representative of {P, -P}: the one with smaller y.

    Identity (None) is its own canonical form.

    Note: secp256k1's field prime is odd, so y == FIELD_P - y has no integer
    solution; ties are impossible.
    """
    if point is None:
        return None
    x, y = point
    if y < FIELD_P - y:
        return point
    return (x, FIELD_P - y)


def canonical_with_flag(point):
    """As canonical(P), but also returns whether the result is -P.

    Returns (canonical_point, flipped). flipped is True iff the input had to
    be negated to reach canonical form. The kangaroo walk uses this flag to
    keep its scalar tracker in sync with the canonicalized point.
    """
    if point is None:
        return None, False
    x, y = point
    if y < FIELD_P - y:
        return point, False
    return (x, FIELD_P - y), True
