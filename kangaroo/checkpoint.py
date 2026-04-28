"""Save/load kangaroo solver state for checkpointed long runs.

The Colab session limit (12 hours) is shorter than the expected solve time
for puzzle #135. Without checkpointing, every session boundary destroys
progress. With checkpointing, a session loads the previous DP table, walks
for ~11.5 hours, saves, and exits cleanly.

File format:
    line 1: 64-character ASCII hex SHA256 of the rest of the file
    rest:   pickle-serialized payload dict {magic, version, params, state}

Integrity is checked on load. A mismatch raises CorruptedCheckpoint —
silently starting fresh would erase progress, which is worse than aborting.
Atomic writes (.tmp + os.replace) prevent half-written files from a crash
mid-save destroying the previous good checkpoint.
"""
import hashlib
import os
import pickle


class CorruptedCheckpoint(Exception):
    """Raised when a checkpoint file fails any integrity check on load."""


_MAGIC = "KGRO"
_VERSION = 1


def save_checkpoint(path: str, *, params: dict, state: dict) -> None:
    """Atomically write a checkpoint with SHA256 integrity tag.

    `params` should contain the solver configuration (Q, k1, k2, jumps,
    dp_bits, negation) — anything that affects determinism. `state` is the
    walker state at the moment of save.
    """
    payload = {
        "magic": _MAGIC,
        "version": _VERSION,
        "params": params,
        "state": state,
    }
    raw = pickle.dumps(payload, protocol=4)
    digest = hashlib.sha256(raw).hexdigest()

    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        f.write(digest.encode("ascii"))
        f.write(b"\n")
        f.write(raw)
    os.replace(tmp_path, path)


def load_checkpoint(path: str) -> tuple[dict, dict]:
    """Load and verify a checkpoint. Raises CorruptedCheckpoint on any issue.

    Returns (params, state).
    """
    try:
        with open(path, "rb") as f:
            digest_line = f.readline()
            raw = f.read()
    except OSError as e:
        raise CorruptedCheckpoint(f"cannot read {path}: {e}") from e

    digest_str = digest_line.rstrip(b"\n").decode("ascii", errors="replace")
    if len(digest_str) != 64 or any(c not in "0123456789abcdef" for c in digest_str):
        raise CorruptedCheckpoint(f"invalid digest line: {digest_str!r}")

    expected = hashlib.sha256(raw).hexdigest()
    if digest_str != expected:
        raise CorruptedCheckpoint(
            f"checksum mismatch: stored {digest_str}, computed {expected}"
        )

    try:
        payload = pickle.loads(raw)
    except Exception as e:
        raise CorruptedCheckpoint(f"unpickling failed: {e}") from e

    if not isinstance(payload, dict):
        raise CorruptedCheckpoint(f"payload is not a dict: {type(payload).__name__}")
    if payload.get("magic") != _MAGIC:
        raise CorruptedCheckpoint(f"bad magic: {payload.get('magic')!r}")
    if payload.get("version") != _VERSION:
        raise CorruptedCheckpoint(
            f"version {payload.get('version')!r} not supported (expected {_VERSION})"
        )

    return payload["params"], payload["state"]
