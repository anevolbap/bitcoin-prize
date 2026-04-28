#!/usr/bin/env python3
"""Generate a Kangaroo-256/JLP-format puzzle file with a known answer.

Used by smoke tests on Colab to validate that the GPU solver builds correctly
and recovers the right key on a small synthetic puzzle before being pointed
at the real target.

Outputs two files at <out_path>:
  <out_path>          JLP input — three hex lines: start, end, pubkey
  <out_path>.json     manifest — known d, k1, k2, pubkey, bits, seed

The manifest is what the smoke test compares against the solver's recovered
key. Don't ship this manifest alongside any real-target run — it would
trivialize the problem.

Usage:
    cd <repo_root>
    python3 scripts/make_synthetic_puzzle.py --bits 60 --seed 42 --out /tmp/p.txt
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

# Allow running as a script from any directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from kangaroo.curve import G, scalar_mult  # noqa: E402


def serialize_compressed(point: tuple[int, int]) -> str:
    """SEC1 compressed: 02/03 prefix + 32-byte big-endian x."""
    x, y = point
    return ("02" if (y & 1) == 0 else "03") + f"{x:064x}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--bits", type=int, required=True,
                   help="puzzle bit width; key range is [2^(bits-1), 2^bits)")
    p.add_argument("--seed", type=int, default=None,
                   help="seed for the RNG; omit for nondeterministic")
    p.add_argument("--out", required=True,
                   help="path to write the JLP input file (manifest at <out>.json)")
    args = p.parse_args()

    if args.bits < 2:
        print("--bits must be >= 2", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    k1 = 1 << (args.bits - 1)
    k2 = 1 << args.bits
    d = rng.randrange(k1, k2)
    Q = scalar_mult(d, G)
    assert Q is not None, "scalar_mult returned identity for nonzero d"
    pubkey_hex = serialize_compressed(Q)

    with open(args.out, "w") as f:
        # Hex without 0x prefix; uppercase for readability of the range bounds.
        f.write(f"{k1:X}\n{k2:X}\n{pubkey_hex}\n")

    manifest = {
        "bits": args.bits,
        "seed": args.seed,
        "d_hex": f"{d:x}",
        "k1_hex": f"{k1:x}",
        "k2_hex": f"{k2:x}",
        "pubkey_hex": pubkey_hex,
    }
    with open(args.out + ".json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"wrote {args.out} (bits={args.bits}, d=0x{d:x})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
