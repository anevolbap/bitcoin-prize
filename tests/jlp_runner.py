"""Adapter for invoking the JeanLucPons/Kangaroo binary from tests.

JLP is the reference implementation we cross-validate against. It's a separate
C++/CUDA project; this module locates the binary, drives it on a single
puzzle, and parses the recovered key from stdout.

Resolution order for the binary:
    1. $JLP_KANGAROO_BIN
    2. `kangaroo` / `Kangaroo` on $PATH
    3. <repo_root>/tools/Kangaroo/{kangaroo,Kangaroo}

Returns None if absent, and the test layer turns that into pytest.skip — we
do NOT silently treat "binary missing" as a passing test.

Input file format JLP expects (one entry per line, hex without 0x prefix):
    <start>
    <stop>
    <pubkey>            # compressed (66 chars) or uncompressed (130 chars)
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


def find_binary() -> str | None:
    env = os.environ.get("JLP_KANGAROO_BIN")
    if env:
        if os.path.isfile(env) and os.access(env, os.X_OK):
            return env
        return None
    for name in ("kangaroo", "Kangaroo"):
        p = shutil.which(name)
        if p:
            return p
    repo_root = Path(__file__).resolve().parent.parent
    for cand in (
        repo_root / "tools" / "Kangaroo" / "kangaroo",
        repo_root / "tools" / "Kangaroo" / "Kangaroo",
    ):
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    return None


def serialize_compressed(point: tuple[int, int]) -> str:
    """SEC1 compressed encoding: 02/03 prefix + 32-byte big-endian x."""
    x, y = point
    prefix = "02" if (y & 1) == 0 else "03"
    return prefix + f"{x:064x}"


# Match `Priv: 0x<hex>` or `Priv: <hex>` — JLP versions vary on the prefix.
_PRIV_RE = re.compile(r"Priv\s*:\s*(?:0x)?([0-9A-Fa-f]+)")


def run(
    binary: str,
    k1: int,
    k2: int,
    Q: tuple[int, int],
    *,
    timeout: float = 120.0,
    extra_args: list[str] | None = None,
) -> dict:
    """Run JLP on one puzzle, return {'key': int, 'stdout': str, 'returncode': int}.

    Raises RuntimeError if the run times out or no Priv: line is parsed.
    """
    pubkey = serialize_compressed(Q)
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        in_file = td_path / "puzzle.txt"
        in_file.write_text(f"{k1:X}\n{k2:X}\n{pubkey}\n")
        cmd = [binary, "-t", "1"]
        if extra_args:
            cmd += extra_args
        cmd += [str(in_file)]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(td_path),
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"JLP timed out after {timeout}s on {k2 - k1:#x}-wide interval. "
                f"stdout so far:\n{e.stdout or ''}"
            ) from e
        out = (proc.stdout or "") + (proc.stderr or "")
        m = _PRIV_RE.search(out)
        if not m:
            raise RuntimeError(
                f"JLP produced no Priv: line (returncode={proc.returncode}). "
                f"Output:\n{out}"
            )
        return {
            "key": int(m.group(1), 16),
            "stdout": out,
            "returncode": proc.returncode,
        }
