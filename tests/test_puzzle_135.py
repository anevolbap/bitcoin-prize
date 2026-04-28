"""Self-consistency checks for the puzzle #135 constants.

Catches a fat-finger in `kangaroo/puzzle_135.py` before any solver run wastes
GPU-years on the wrong target.
"""
import hashlib

from kangaroo.curve import P, is_on_curve
from kangaroo.puzzle_135 import (
    ADDRESS, K1, K2, PUBKEY_BYTES, PUBKEY_HEX, PUBKEY_X, PUBKEY_Y_PARITY,
)


def _hash160(b: bytes) -> bytes:
    return hashlib.new("ripemd160", hashlib.sha256(b).digest()).digest()


def _b58check(payload: bytes) -> str:
    alpha = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    raw = payload + hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    n = int.from_bytes(raw, "big")
    out: list[str] = []
    while n > 0:
        n, r = divmod(n, 58)
        out.append(alpha[r])
    leading = 0
    for b in raw:
        if b == 0:
            leading += 1
        else:
            break
    return "1" * leading + "".join(reversed(out))


def test_pubkey_hashes_to_address():
    """HASH160(pubkey) base58-checks to the published puzzle #135 address."""
    assert _b58check(b"\x00" + _hash160(PUBKEY_BYTES)) == ADDRESS


def test_pubkey_is_on_curve():
    """Decompressed pubkey is a valid secp256k1 point."""
    assert PUBKEY_BYTES[0] in (0x02, 0x03)
    # Solve y^2 = x^3 + 7 (mod P), pick the root matching the parity prefix.
    rhs = (pow(PUBKEY_X, 3, P) + 7) % P
    # secp256k1's prime ≡ 3 (mod 4), so the modular sqrt is rhs^((P+1)/4).
    y = pow(rhs, (P + 1) // 4, P)
    if (y & 1) != PUBKEY_Y_PARITY:
        y = P - y
    assert (y * y - rhs) % P == 0, "decompression failed: not a quadratic residue"
    assert is_on_curve((PUBKEY_X, y))


def test_range_brackets_134_bits():
    """Puzzle #135's range is exactly [2^134, 2^135)."""
    assert K1 == 1 << 134
    assert K2 == 1 << 135
    assert (K2 - K1).bit_length() == 135


def test_pubkey_hex_well_formed():
    assert len(PUBKEY_HEX) == 66
    assert PUBKEY_HEX[:2] in ("02", "03")
    assert all(c in "0123456789abcdef" for c in PUBKEY_HEX[2:])
