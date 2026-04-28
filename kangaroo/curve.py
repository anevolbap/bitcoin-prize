"""secp256k1 affine arithmetic — pure Python reference.

Point representation:
    (x, y)  with coordinates in [0, P)   for finite points
    None                                 for the identity (point at infinity)

This is the slow, obviously-correct reference. Every group op performs a field
inversion. All optimized implementations (Jacobian, batched, CUDA) must match
this on the same inputs.
"""

# secp256k1 domain parameters (SEC 2 v2, §2.4.1)
P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
A = 0
B = 7
GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
G = (GX, GY)
INFINITY = None


def field_inv(a: int) -> int:
    """Modular inverse of a in F_P. Raises ZeroDivisionError if a == 0 mod P."""
    a %= P
    if a == 0:
        raise ZeroDivisionError("inverse of 0 in F_P")
    return pow(a, -1, P)


def is_on_curve(point) -> bool:
    """Check y^2 == x^3 + 7 (mod P). Identity is on the curve by convention."""
    if point is None:
        return True
    x, y = point
    return (y * y - x * x * x - B) % P == 0


def point_neg(point):
    """-P = (x, -y mod P). Identity is its own inverse."""
    if point is None:
        return None
    x, y = point
    return (x, (-y) % P)


def point_double(point):
    """2P via the affine doubling formula."""
    if point is None:
        return None
    x, y = point
    if y == 0:
        # tangent is vertical → 2P = O. (Not reachable on secp256k1 with non-id P,
        # since the curve has no point of order 2, but we handle it anyway.)
        return None
    s = (3 * x * x * field_inv(2 * y)) % P
    x3 = (s * s - 2 * x) % P
    y3 = (s * (x - x3) - y) % P
    return (x3, y3)


def point_add(p1, p2):
    """P1 + P2 via the affine addition formula. Dispatches to double if P1 == P2."""
    if p1 is None:
        return p2
    if p2 is None:
        return p1
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2:
        if (y1 + y2) % P == 0:
            return None  # P + (-P) = O
        return point_double(p1)  # P1 == P2
    s = ((y2 - y1) * field_inv(x2 - x1)) % P
    x3 = (s * s - x1 - x2) % P
    y3 = (s * (x1 - x3) - y1) % P
    return (x3, y3)


def scalar_mult(k: int, point=G):
    """k * point via right-to-left double-and-add. k may be any integer."""
    if point is None:
        return None
    k %= N
    if k == 0:
        return None
    result = None
    addend = point
    while k:
        if k & 1:
            result = point_add(result, addend)
        addend = point_double(addend)
        k >>= 1
    return result
