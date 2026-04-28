"""Layer 5: checkpoint round-trip — save, load, resume, verify state equality.

The most consequential test in the project: a Colab session crash mid-walk
must not erase progress. Round-trip equality is the definition of "the
checkpoint actually captures the full walker state".
"""
import hashlib
import logging
import os
import pickle
import random

import pytest

from kangaroo.checkpoint import (
    CorruptedCheckpoint, load_checkpoint, save_checkpoint,
)
from kangaroo.curve import G, scalar_mult
from kangaroo.kangaroo import solve

log = logging.getLogger(__name__)


def _make_puzzle(bits: int, seed: int):
    rng = random.Random(seed)
    k1, k2 = 1 << (bits - 1), 1 << bits
    d = rng.randrange(k1, k2)
    return d, scalar_mult(d, G), k1, k2


# ---------------------------------------------------------------------------
# Round-trip determinism
# ---------------------------------------------------------------------------

def test_round_trip_state_equality(tmp_path):
    """20k continuous run state == (10k save + 10k resume) state.

    Uses a 40-bit puzzle so 20k steps reliably don't collide (expected ops
    ≈ 1.5M). dp_bits=8 forces a non-trivial DP table to accumulate, so
    dict-equality isn't vacuous.
    """
    _, Q, k1, k2 = _make_puzzle(40, seed=42)

    ref_path = tmp_path / "ref.ckpt"
    mid_path = tmp_path / "mid.ckpt"
    final_path = tmp_path / "final.ckpt"

    # Reference: 20k continuous → save final state.
    solve(Q, k1, k2, max_steps=20_000, dp_bits=8,
          checkpoint_path=str(ref_path))
    ref_params, ref_state = load_checkpoint(str(ref_path))
    assert ref_state["step_count"] == 20_000

    # Round-trip: 10k → save → load → 10k more → save.
    solve(Q, k1, k2, max_steps=10_000, dp_bits=8,
          checkpoint_path=str(mid_path))
    mid_params, mid_state = load_checkpoint(str(mid_path))
    assert mid_state["step_count"] == 10_000

    solve(Q, k1, k2, max_steps=20_000, dp_bits=8,
          resume_path=str(mid_path), checkpoint_path=str(final_path))
    final_params, final_state = load_checkpoint(str(final_path))

    assert final_state["step_count"] == 20_000
    assert final_params == ref_params

    # Full state equality — this is the determinism guarantee.
    assert final_state == ref_state
    log.info(
        "20k run: tame_dps=%d wild_dps=%d (round-trip matches continuous)",
        len(ref_state["tame_dps"]), len(ref_state["wild_dps"]),
    )


def test_resume_finds_same_key(tmp_path):
    """Solver from a mid-run checkpoint finds the same d as a continuous run."""
    d, Q, k1, k2 = _make_puzzle(20, seed=20)

    found_continuous = solve(Q, k1, k2, max_steps=50_000)
    assert found_continuous == d

    mid_path = tmp_path / "mid.ckpt"
    # Short prefix run that almost certainly hasn't collided yet.
    solve(Q, k1, k2, max_steps=200, checkpoint_path=str(mid_path))
    found_resumed = solve(Q, k1, k2, max_steps=50_000, resume_path=str(mid_path))
    assert found_resumed == d


# ---------------------------------------------------------------------------
# File-size linearity (no memory/serialization leak)
# ---------------------------------------------------------------------------

def test_checkpoint_size_grows_linearly_with_dps(tmp_path):
    """File size scales linearly with stored DPs; per-DP overhead is bounded."""
    _, Q, k1, k2 = _make_puzzle(40, seed=7)

    sizes = []
    dp_counts = []
    for n_steps in [5_000, 10_000, 15_000, 20_000]:
        path = tmp_path / f"size_{n_steps}.ckpt"
        solve(Q, k1, k2, max_steps=n_steps, dp_bits=8,
              checkpoint_path=str(path))
        size = os.path.getsize(path)
        _, state = load_checkpoint(str(path))
        dps = len(state["tame_dps"]) + len(state["wild_dps"])
        sizes.append(size)
        dp_counts.append(dps)
        log.info("n_steps=%d size=%d dps=%d", n_steps, size, dps)

    # DPs should accumulate monotonically.
    assert dp_counts == sorted(dp_counts)
    assert dp_counts[-1] > dp_counts[0], "DPs did not grow across runs"

    # Per-DP overhead in pickle bytes is ~30-100 (two ints in a dict entry).
    # Wide tolerance to absorb dict-resize discontinuities.
    bytes_per_dp = (sizes[-1] - sizes[0]) / (dp_counts[-1] - dp_counts[0])
    log.info("inferred bytes per DP: %.1f", bytes_per_dp)
    assert 20 < bytes_per_dp < 5000, f"unexpected per-DP overhead {bytes_per_dp}"


# ---------------------------------------------------------------------------
# Corruption detection — silently starting fresh would erase progress
# ---------------------------------------------------------------------------

def test_corrupted_payload_raises(tmp_path):
    """Flipping a byte in the pickled payload is detected via checksum."""
    _, Q, k1, k2 = _make_puzzle(20, seed=1)
    path = tmp_path / "good.ckpt"
    solve(Q, k1, k2, max_steps=500, checkpoint_path=str(path))

    # Sanity: clean load works.
    load_checkpoint(str(path))

    # Flip a byte 100 bytes past the digest line.
    raw = path.read_bytes()
    nl = raw.index(b"\n")
    target = nl + 1 + 100
    corrupted = raw[:target] + bytes([raw[target] ^ 0xFF]) + raw[target + 1:]
    path.write_bytes(corrupted)

    with pytest.raises(CorruptedCheckpoint, match="checksum mismatch"):
        load_checkpoint(str(path))


def test_truncated_file_raises(tmp_path):
    """A truncated file is rejected (checksum mismatch or unpickle error)."""
    _, Q, k1, k2 = _make_puzzle(20, seed=1)
    path = tmp_path / "trunc.ckpt"
    solve(Q, k1, k2, max_steps=500, checkpoint_path=str(path))

    raw = path.read_bytes()
    path.write_bytes(raw[: len(raw) // 2])

    with pytest.raises(CorruptedCheckpoint):
        load_checkpoint(str(path))


def test_corrupt_digest_line_raises(tmp_path):
    """Malformed digest line (not 64 hex chars) is rejected."""
    _, Q, k1, k2 = _make_puzzle(20, seed=1)
    path = tmp_path / "bad_digest.ckpt"
    solve(Q, k1, k2, max_steps=500, checkpoint_path=str(path))

    raw = path.read_bytes()
    nl = raw.index(b"\n")
    # Replace digest with garbage of wrong length.
    path.write_bytes(b"not-a-real-hex-digest\n" + raw[nl + 1 :])

    with pytest.raises(CorruptedCheckpoint, match="invalid digest"):
        load_checkpoint(str(path))


def test_wrong_magic_raises(tmp_path):
    """A properly-checksummed file with the wrong magic is still rejected."""
    path = tmp_path / "wrong_magic.ckpt"
    payload = pickle.dumps(
        {"magic": "EVIL", "version": 1, "params": {}, "state": {}},
        protocol=4,
    )
    digest = hashlib.sha256(payload).hexdigest()
    with open(path, "wb") as f:
        f.write(digest.encode("ascii"))
        f.write(b"\n")
        f.write(payload)

    with pytest.raises(CorruptedCheckpoint, match="bad magic"):
        load_checkpoint(str(path))


def test_unsupported_version_raises(tmp_path):
    """Future or unknown versions are rejected, not silently ignored."""
    path = tmp_path / "future.ckpt"
    payload = pickle.dumps(
        {"magic": "KGRO", "version": 99, "params": {}, "state": {}},
        protocol=4,
    )
    digest = hashlib.sha256(payload).hexdigest()
    with open(path, "wb") as f:
        f.write(digest.encode("ascii"))
        f.write(b"\n")
        f.write(payload)

    with pytest.raises(CorruptedCheckpoint, match="version"):
        load_checkpoint(str(path))


def test_missing_file_raises(tmp_path):
    with pytest.raises(CorruptedCheckpoint, match="cannot read"):
        load_checkpoint(str(tmp_path / "nope.ckpt"))


# ---------------------------------------------------------------------------
# Param-mismatch on resume
# ---------------------------------------------------------------------------

def test_resume_with_mismatched_pubkey_raises(tmp_path):
    """Resuming with a different Q must raise — would corrupt the run."""
    _, Q1, k1, k2 = _make_puzzle(20, seed=1)
    _, Q2, _, _ = _make_puzzle(20, seed=2)
    assert Q1 != Q2

    path = tmp_path / "p.ckpt"
    solve(Q1, k1, k2, max_steps=200, checkpoint_path=str(path))

    with pytest.raises(ValueError, match="checkpoint params"):
        solve(Q2, k1, k2, max_steps=200, resume_path=str(path))


def test_resume_with_mismatched_dp_bits_raises(tmp_path):
    """Different dp_bits → different DP semantics → resume must abort."""
    _, Q, k1, k2 = _make_puzzle(20, seed=1)
    path = tmp_path / "p.ckpt"
    solve(Q, k1, k2, max_steps=200, dp_bits=4, checkpoint_path=str(path))

    with pytest.raises(ValueError, match="checkpoint params"):
        solve(Q, k1, k2, max_steps=200, dp_bits=8, resume_path=str(path))


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------

def test_save_is_atomic(tmp_path):
    """No .tmp file should remain after a successful save."""
    path = tmp_path / "atomic.ckpt"
    save_checkpoint(str(path), params={"a": 1}, state={"b": 2})
    assert path.exists()
    assert not (tmp_path / "atomic.ckpt.tmp").exists()
    p, s = load_checkpoint(str(path))
    assert p == {"a": 1}
    assert s == {"b": 2}
