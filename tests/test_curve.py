"""Validate kangaroo.curve against coincurve as the reference oracle."""
import secrets

import pytest
from coincurve import PrivateKey
from coincurve.keys import PublicKey

from kangaroo.curve import (
    P, N, G,
    field_inv, is_on_curve,
    point_add, point_double, point_neg, scalar_mult,
)


# ---------------------------------------------------------------------------
# coincurve adapters
# ---------------------------------------------------------------------------

def _to_uncompressed(point) -> bytes:
    x, y = point
    return b"\x04" + x.to_bytes(32, "big") + y.to_bytes(32, "big")


def _from_pubkey(pk: PublicKey):
    raw = pk.format(compressed=False)
    assert raw[0] == 0x04
    x = int.from_bytes(raw[1:33], "big")
    y = int.from_bytes(raw[33:65], "big")
    return (x, y)


def _ref_kG(k: int):
    """k*G via coincurve. Returns None for k ≡ 0 (mod N)."""
    k_mod = k % N
    if k_mod == 0:
        return None
    return _from_pubkey(PrivateKey.from_int(k_mod).public_key)


def _ref_add(p1, p2):
    """p1 + p2 via coincurve. Caller must guarantee p1 != -p2."""
    if p1 is None:
        return p2
    if p2 is None:
        return p1
    pk1 = PublicKey(_to_uncompressed(p1))
    pk2 = PublicKey(_to_uncompressed(p2))
    return _from_pubkey(PublicKey.combine_keys([pk1, pk2]))


# ---------------------------------------------------------------------------
# Field arithmetic
# ---------------------------------------------------------------------------

def test_field_inv_property():
    """a * inv(a) ≡ 1 (mod P) for 1000 random nonzero a."""
    for _ in range(1000):
        a = secrets.randbelow(P - 1) + 1
        assert (a * field_inv(a)) % P == 1


def test_field_inv_handles_unreduced_input():
    """inv accepts a not yet reduced mod P."""
    a = secrets.randbelow(P - 1) + 1
    assert field_inv(a) == field_inv(a + P)
    assert field_inv(a) == field_inv(a + 7 * P)


def test_field_inv_zero_raises():
    with pytest.raises(ZeroDivisionError):
        field_inv(0)
    with pytest.raises(ZeroDivisionError):
        field_inv(P)  # 0 mod P


def test_field_add_mul_consistent():
    """Sanity: (a+b) mod P and (a*b) mod P agree with reduce-then-op."""
    for _ in range(1000):
        a = secrets.randbelow(P)
        b = secrets.randbelow(P)
        assert (a + b) % P == ((a % P) + (b % P)) % P
        assert (a * b) % P == ((a % P) * (b % P)) % P


# ---------------------------------------------------------------------------
# Curve membership and the generator
# ---------------------------------------------------------------------------

def test_generator_on_curve():
    assert is_on_curve(G)


def test_identity_on_curve():
    assert is_on_curve(None)


def test_random_kG_on_curve():
    for _ in range(50):
        k = secrets.randbelow(N - 1) + 1
        assert is_on_curve(_ref_kG(k))


# ---------------------------------------------------------------------------
# Scalar multiplication vs coincurve
# ---------------------------------------------------------------------------

def test_scalar_mult_one_is_generator():
    assert scalar_mult(1, G) == G


def test_scalar_mult_matches_coincurve():
    """k*G matches coincurve for 100 random k in [1, N-1]."""
    for _ in range(100):
        k = secrets.randbelow(N - 1) + 1
        assert scalar_mult(k, G) == _ref_kG(k)


def test_scalar_mult_reduces_mod_n():
    """k*G == (k mod N)*G for k >= N."""
    for _ in range(20):
        # k in [N, 2^256)
        k = N + secrets.randbelow(2**256 - N)
        assert scalar_mult(k, G) == scalar_mult(k % N, G)


def test_scalar_mult_zero_is_infinity():
    assert scalar_mult(0, G) is None
    assert scalar_mult(N, G) is None


def test_scalar_mult_arbitrary_base_point():
    """k*(j*G) == (k*j)*G."""
    for _ in range(20):
        j = secrets.randbelow(N - 1) + 1
        k = secrets.randbelow(N - 1) + 1
        base = _ref_kG(j)
        assert scalar_mult(k, base) == _ref_kG(j * k)


# ---------------------------------------------------------------------------
# Point addition and doubling vs coincurve
# ---------------------------------------------------------------------------

def test_point_add_matches_coincurve():
    """P + Q matches coincurve for random distinct P, Q."""
    for _ in range(100):
        a = secrets.randbelow(N - 1) + 1
        b = secrets.randbelow(N - 1) + 1
        if (a + b) % N == 0 or a == b:
            continue  # skip P + (-P) and P + P (covered separately)
        Pa = _ref_kG(a)
        Pb = _ref_kG(b)
        assert point_add(Pa, Pb) == _ref_add(Pa, Pb)


def test_point_add_same_point_doubles():
    """point_add(P, P) dispatches to doubling and matches 2*k*G."""
    for _ in range(50):
        k = secrets.randbelow(N - 1) + 1
        Pk = _ref_kG(k)
        assert point_add(Pk, Pk) == _ref_kG(2 * k)


def test_point_double_matches_coincurve():
    """2P == (2k)*G for P = k*G."""
    for _ in range(50):
        k = secrets.randbelow(N - 1) + 1
        Pk = _ref_kG(k)
        assert point_double(Pk) == _ref_kG(2 * k)


def test_point_double_of_infinity():
    assert point_double(None) is None


# ---------------------------------------------------------------------------
# Identity and negation
# ---------------------------------------------------------------------------

def test_identity_is_additive_neutral():
    Pk = _ref_kG(12345)
    assert point_add(None, Pk) == Pk
    assert point_add(Pk, None) == Pk
    assert point_add(None, None) is None


def test_negation_formula():
    """-P = (x, P - y)."""
    for _ in range(100):
        k = secrets.randbelow(N - 1) + 1
        Pk = _ref_kG(k)
        assert Pk is not None
        x, y = Pk
        assert point_neg(Pk) == (x, (P - y) % P)


def test_negation_of_infinity():
    assert point_neg(None) is None


def test_inverse_addition_yields_infinity():
    """P + (-P) = O."""
    for _ in range(50):
        k = secrets.randbelow(N - 1) + 1
        Pk = _ref_kG(k)
        assert point_add(Pk, point_neg(Pk)) is None


def test_negation_consistent_with_scalar():
    """-P == (N-1)*P + ... actually -P == (-1 mod N)*G when P = G."""
    Pk = _ref_kG(7)
    assert point_neg(Pk) == _ref_kG(N - 7)
