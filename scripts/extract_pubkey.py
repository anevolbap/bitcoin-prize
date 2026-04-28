#!/usr/bin/env python3
"""Extract a Bitcoin puzzle's public key from its first outgoing transaction.

When a P2PKH address spends, the input scriptSig contains the spender's public
key (since pay-to-public-key-hash requires revealing the key to satisfy the
hash check). Bitcoin puzzle addresses that have been spent from at least once
therefore expose their public key on the chain — that's why puzzle #135 is
attackable via Pollard's Kangaroo (O(sqrt(N))) instead of brute-force hash
search (O(N)).

This script:
  1. Pulls the tx list for an address from Blockstream's public API.
  2. Finds the first input that spends FROM that address (P2PKH or P2WPKH).
  3. Extracts the public key from the scriptSig or witness.
  4. Verifies HASH160(pubkey) base58-encodes back to the input address.

The verification step is non-negotiable: feeding the wrong public key to the
solver would silently target a different ECDLP problem, with the GPU happily
running for years producing nothing.

Usage:
    python scripts/extract_pubkey.py
    python scripts/extract_pubkey.py 16RGFo6hjq9ym6Pj7N5H7L1NR1rVPJyw2v

Output goes to stdout as a single hex string (the public key); diagnostics go
to stderr so the script is pipeline-friendly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request

PUZZLE_135_ADDRESS = "16RGFo6hjq9ym6Pj7N5H7L1NR1rVPJyw2v"
BLOCKSTREAM_API = "https://blockstream.info/api"

_BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _http_get_json(url: str, *, timeout: float = 30.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def fetch_address_txs(address: str) -> list[dict]:
    """All transactions touching `address`, paged. Newest first.

    Blockstream returns up to 25 confirmed txs per page; subsequent pages use
    /chain/<last_seen_txid>. We don't filter by direction here.
    """
    base = f"{BLOCKSTREAM_API}/address/{address}/txs"
    out: list[dict] = []
    last = None
    while True:
        url = base if last is None else f"{base}/chain/{last}"
        page = _http_get_json(url)
        if not page:
            return out
        out.extend(page)
        if len(page) < 25:
            return out
        last = page[-1]["txid"]


def find_first_spend_from(txs: list[dict], address: str) -> tuple[dict, int, dict]:
    """First (tx, vin_index, vin_dict) where `vin` spends from `address`.

    Iterates oldest-first so the result is reproducible — picking "newest" or
    "any" would mean the chosen tx changes whenever the address sees new
    activity. Raises LookupError if no spend exists.
    """
    for tx in reversed(txs):
        for i, vin in enumerate(tx.get("vin", [])):
            prev = vin.get("prevout") or {}
            if prev.get("scriptpubkey_address") == address:
                return tx, i, vin
    raise LookupError(
        f"no input spending from {address} in {len(txs)} txs — pubkey not yet exposed"
    )


def _read_pushes(script: bytes) -> list[bytes]:
    """Decode a Bitcoin script as a sequence of pushed data items.

    Only data-push opcodes (0x01..0x4b, 0x4c, 0x4d, 0x4e). Any non-push opcode
    raises — a P2PKH unlock script must be exactly two pushes (sig, pubkey).
    """
    out: list[bytes] = []
    i = 0
    while i < len(script):
        op = script[i]
        i += 1
        if 1 <= op <= 0x4B:
            out.append(script[i : i + op]); i += op
        elif op == 0x4C:  # OP_PUSHDATA1
            n = script[i]; i += 1
            out.append(script[i : i + n]); i += n
        elif op == 0x4D:  # OP_PUSHDATA2
            n = int.from_bytes(script[i : i + 2], "little"); i += 2
            out.append(script[i : i + n]); i += n
        elif op == 0x4E:  # OP_PUSHDATA4
            n = int.from_bytes(script[i : i + 4], "little"); i += 4
            out.append(script[i : i + n]); i += n
        else:
            raise ValueError(f"non-push opcode 0x{op:02x} at byte {i - 1}")
    return out


def extract_pubkey_from_vin(vin: dict) -> bytes:
    """Pull the public key from a P2PKH or P2WPKH input."""
    scriptsig_hex = vin.get("scriptsig") or ""
    witness = vin.get("witness") or []

    if scriptsig_hex:
        pushes = _read_pushes(bytes.fromhex(scriptsig_hex))
        if len(pushes) != 2:
            raise ValueError(
                f"scriptSig has {len(pushes)} pushes; expected 2 (sig, pubkey)"
            )
        return pushes[1]
    if len(witness) >= 2:
        # P2WPKH witness: <sig> <pubkey>
        return bytes.fromhex(witness[1])
    raise ValueError("no scriptSig and witness has < 2 items — unsupported input type")


def hash160(data: bytes) -> bytes:
    return hashlib.new("ripemd160", hashlib.sha256(data).digest()).digest()


def base58_check_encode(payload: bytes) -> str:
    checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    raw = payload + checksum
    n = int.from_bytes(raw, "big")
    out: list[str] = []
    while n > 0:
        n, r = divmod(n, 58)
        out.append(_BASE58_ALPHABET[r])
    leading = 0
    for b in raw:
        if b == 0:
            leading += 1
        else:
            break
    return "1" * leading + "".join(reversed(out))


def pubkey_to_p2pkh_address(pubkey: bytes) -> str:
    return base58_check_encode(b"\x00" + hash160(pubkey))


def main() -> int:
    p = argparse.ArgumentParser(
        description="Extract the public key for a Bitcoin P2PKH/P2WPKH address "
        "from its first outgoing transaction."
    )
    p.add_argument(
        "address", nargs="?", default=PUZZLE_135_ADDRESS,
        help=f"target address (default: puzzle #135 {PUZZLE_135_ADDRESS})",
    )
    args = p.parse_args()

    print(f"querying {BLOCKSTREAM_API} for txs of {args.address} ...", file=sys.stderr)
    try:
        txs = fetch_address_txs(args.address)
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code}: {e.read().decode(errors='replace')[:200]}", file=sys.stderr)
        return 2
    print(f"  {len(txs)} txs", file=sys.stderr)

    tx, vin_idx, vin = find_first_spend_from(txs, args.address)
    print(f"first spend: {tx['txid']} vin[{vin_idx}]", file=sys.stderr)

    pubkey = extract_pubkey_from_vin(vin)
    encoding = (
        "compressed" if len(pubkey) == 33 and pubkey[0] in (2, 3)
        else "uncompressed" if len(pubkey) == 65 and pubkey[0] == 4
        else f"unknown ({len(pubkey)} bytes)"
    )
    print(f"pubkey: {encoding}", file=sys.stderr)

    derived = pubkey_to_p2pkh_address(pubkey)
    if derived != args.address:
        print(
            f"FATAL: HASH160(pubkey) → {derived}, expected {args.address}\n"
            f"  refusing to emit a pubkey that doesn't match the input address",
            file=sys.stderr,
        )
        return 3
    print(f"verified: HASH160(pubkey) → {derived}", file=sys.stderr)

    print(pubkey.hex())
    return 0


if __name__ == "__main__":
    sys.exit(main())
