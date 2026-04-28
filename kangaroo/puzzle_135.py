"""Bitcoin puzzle #135 constants.

Extracted via `scripts/extract_pubkey.py` from the address's first spend
transaction (vin[14] of 17e4e323cfbc68d7f0071cad09364e8193eedf8fefbcbd8a21b4b65717a4b3d3).
The verification — HASH160(PUBKEY) → ADDRESS — runs in tests/test_puzzle_135.py
and at no point should this file be edited without re-running the extractor.

Wrong constants here would be catastrophic: the solver would silently target a
different ECDLP problem and never find the actual puzzle key.
"""

ADDRESS = "16RGFo6hjq9ym6Pj7N5H7L1NR1rVPJyw2v"

# Compressed SEC1 encoding: 02/03 prefix + 32-byte big-endian x.
# Even y → 0x02 prefix here.
PUBKEY_HEX = "02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16"
PUBKEY_BYTES = bytes.fromhex(PUBKEY_HEX)
PUBKEY_X = int(PUBKEY_HEX[2:], 16)
PUBKEY_Y_PARITY = int(PUBKEY_HEX[:2], 16) & 1  # 0 = even, 1 = odd

# Search range: 2^(N-1) ≤ d < 2^N for puzzle #N.
K1 = 1 << 134
K2 = 1 << 135

# Provenance — for audit / re-verification.
SOURCE_TXID = "17e4e323cfbc68d7f0071cad09364e8193eedf8fefbcbd8a21b4b65717a4b3d3"
SOURCE_VIN_INDEX = 14
