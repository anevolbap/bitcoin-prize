"""verify_solution() catches each failure mode it's meant to.

The function is the last line of defense before broadcasting a claim
transaction; coverage here must be tight.
"""
import pytest

from kangaroo.curve import G, scalar_mult
from kangaroo.puzzle_135 import K1, K2, PUBKEY_HEX
from kangaroo.verify import serialize_compressed, verify_solution


def _pk(d: int) -> str:
    Q = scalar_mult(d, G)
    assert Q is not None
    return serialize_compressed(Q)


def test_passes_on_valid_solution():
    d = 0xC0FFEE
    assert verify_solution(d, _pk(d), 1, 1 << 32)


def test_rejects_d_below_interval():
    d = 50
    with pytest.raises(ValueError, match="outside interval"):
        verify_solution(d, _pk(d), 100, 1000)


def test_rejects_d_above_interval():
    d = 5000
    with pytest.raises(ValueError, match="outside interval"):
        verify_solution(d, _pk(d), 1, 1000)


def test_rejects_wrong_pubkey():
    """Mismatch between scalar and target Q (the sign-tracking bug surfaces here)."""
    d = 0xCAFEBABE
    wrong_pubkey = _pk(d + 1)
    with pytest.raises(ValueError, match="d\\*G != Q"):
        verify_solution(d, wrong_pubkey, 1, 1 << 40)


def test_rejects_d_against_puzzle_135_pubkey():
    """A guessed scalar that isn't the real puzzle key fails verification."""
    d = K1  # exact lower bound, definitely not the answer
    with pytest.raises(ValueError, match="d\\*G != Q"):
        verify_solution(d, PUBKEY_HEX, K1, K2)


def test_pubkey_case_insensitive():
    """Solvers may emit upper-case hex; we normalize."""
    d = 0xDEAD_BEEF
    assert verify_solution(d, _pk(d).upper(), 1, 1 << 40)
