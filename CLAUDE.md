# Bitcoin Puzzle #135 — Kangaroo Solver Project

## Who I am

- PhD mathematician, Lead Data Scientist at Muun (Bitcoin/Lightning Network wallet)
- Primary language: Python
- Editor: Emacs
- OS: Debian Linux
- Laptop: Lenovo ThinkPad 2024 — **GPU model still unknown**, run
  `lspci | grep -E "VGA|3D|Display"` and `nvidia-smi` to determine if discrete NVIDIA exists
- Preference: **design and explain plan before writing code**

---

## Project goal

Implement an optimized Pollard's Kangaroo ECDLP solver for secp256k1, targeting
**Bitcoin Puzzle #135**, running on **a single free Google Colab T4 instance** with
checkpointing to Google Drive across the 12-hour session limit.

This is treated as an **engineering exercise with a lottery-ticket payoff**, not a
realistic path to the BTC reward. The expected solve time on one T4 is ~6,300 years.
The point is to build a correct, well-tested implementation and let it run for free.

---

## Bitcoin puzzle background

In 2015 an anonymous creator funded 160 Bitcoin addresses where puzzle #N has a private
key `d` satisfying `2^(N-1) <= d < 2^N`. As of early 2026, puzzles #1–#70 and all
multiples of 5 up to #130 are solved (82 total).

### Why #135 and not #71

| Puzzle | Public key known? | Best algorithm | Expected work |
|--------|------------------|----------------|---------------|
| #71 | No | Brute force (hash match) | O(2^70) — ~3,100 GPU-years |
| #135 | **Yes** | Kangaroo | O(2^67) — ~290 GPU-years |

Despite #135 having a larger key, it's ~10× cheaper because the public key is known
and Kangaroo gives O(√N).

### Why lattice/HNP attacks don't apply

Lattice attacks (LLL, Hidden Number Problem, Boneh-Venkatesan) require ECDSA signatures
with biased nonces. Bitcoin puzzles have **no signatures** — just a public key and an
interval. The problem is pure ECDLP-in-an-interval, which Kangaroo solves directly.
Lattice methods are irrelevant here.

### Puzzle #135 parameters

```
Private key range : [2^134, 2^135)
Hex range start   : 0x40000000000000000000000000000000000 (134-bit)
Hex range end     : 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
Bitcoin address   : 16RGFo6hjq9ym6Pj7N5H7L1NR1rVPJyw2v
Reward            : ~13.5 BTC
Public key        : 02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16
Source            : tx 17e4e323cfbc68d7f0071cad09364e8193eedf8fefbcbd8a21b4b65717a4b3d3 (vin 14)
```

The public key must be extracted from the blockchain before starting (see
`scripts/extract_pubkey.py`).

---

## Algorithm: Pollard's Kangaroo (lambda method)

### Core idea

Solve `Q = d*G` where `d ∈ [k1, k2]` in O(√(k2-k1)) group operations.

Two herds of pseudo-random walkers on the elliptic curve:
- **Tame herd**: starts from `T_i = (k1 + N/2 + a_i) * G` for random offsets `a_i`
- **Wild herd**: starts from `W_i = Q + b_i * G` for random offsets `b_i`

Each kangaroo steps deterministically: `P -> P + s(P)` where `s(P)` is chosen
from a precomputed jump table based on a hash of `P.x`.

When tame and wild kangaroos collide on the same point, they follow identical
deterministic paths and both reach the same Distinguished Point (DP). At that
collision: `d = a_tame - b_wild + N/2` (mod n).

### Distinguished Points (DP) method

Store only points whose `x` coordinate has `dp_bits` leading zero bits. These are
rare (probability `1/2^dp_bits`) and serve as collision detectors.

Optimal `dp_bits` ≈ `(log2(N) / 2) - log2(K)` where K = number of kangaroos.
For puzzle #135 with K ≈ 1024 (single T4): `dp_bits ≈ 67 - 10 ≈ 57`.

### Average operation count

`E[ops] = c * sqrt(N)` where `c ≈ 2.08` in JeanLucPons.
For #135: `2.08 * 2^67 ≈ 3 × 10^20` operations.

---

## Planned improvements over JeanLucPons

Treat these as **engineering exercises**. They reduce expected time from ~6,300 years
to ~3,000 years on one T4 — still a lottery, but the implementation work is the point.

### 1. Negation map / ±P folding (~1.41× speedup)

Canonicalize every point to the one with smaller y-coordinate (`min(y, p-y)`). Halves
effective search space.

**Critical**: must apply identically on tame and wild sides. Wrong sign → returns `n - d`.

### 2. Teske r-adding walks (~1.36× speedup)

Jump set distributed so expected jump = N/2 with minimized variance. Reduces birthday
constant from 2.08 toward theoretical minimum ~1.53.

### 3. 125-bit cap removal (mandatory blocker)

JeanLucPons hard-limits to 125-bit intervals. Puzzle #135 needs 134-bit.
Practical Linux-buildable fork: **ZenulAbidin/Kangaroo-256** (up to 256 bits,
inherits JLP's input-file format and work-file checkpointing). Etarkangaroo
is *Windows-only PureBasic* — not usable on Colab; ignore CLAUDE.md drafts
that listed it as a reference.

### 4. Adaptive dp_bits

Compute `dp_bits` at startup based on actual K, not a hardcoded value.

### 5. Batched affine point addition (deferred)

GPU optimization — Montgomery's trick for ~1.20× throughput. Implement only after the
basic CUDA kernel is correct.

---

## Critical correctness lesson

**A buggy Kangaroo does not gracefully degrade — it silently fails forever.**

Failure modes:
- Walk function bug → tame/wild paths never converge → infinite loop, looks healthy
- DP detection bug → collisions never recorded → infinite loop
- Negation map bug → zero collision rate, OR wrong key returned
- Offset tracking bug → wrong key, may pass `d*G == Q` only by coincidence
- 64-bit limb overflow on >125-bit interval → silent wraparound, searching wrong range

The correctness testing strategy below is **mandatory**, not optional. There is no way
to detect these bugs from "watching it run" — the GPU stays at 100% utilization
producing nothing.

---

## Testing strategy — DO NOT SKIP

Every component must pass the tests below before being trusted. Tests live in `tests/`
and run with `pytest`.

### Layer 0: Field arithmetic (`test_curve.py`)

Test the secp256k1 prime field and group operations against a reference (`coincurve`
or `ecdsa` Python libraries):

- [ ] Field addition, multiplication, inversion match reference for 1000 random inputs
- [ ] Point addition `P + Q` matches reference for random `P, Q`
- [ ] Point doubling `2P` matches reference
- [ ] Scalar multiplication `k*G` matches reference for k in [1, 2^256)
- [ ] Identity element handled correctly (`P + (-P) = O`)
- [ ] Negation: `-P = (P.x, p - P.y)` and `(-P) + P = O`

If these don't pass, nothing else can possibly work.

### Layer 1: Algorithm correctness on toy puzzles (`test_kangaroo_small.py`)

Solve **known** puzzles and verify the recovered key matches the published answer.
Pull the canonical answers from privatekeys.pw before encoding into tests:

| Test puzzle | Range | Bits |
|---|---|---|
| Puzzle #20 | [2^19, 2^20) | 20 |
| Puzzle #25 | [2^24, 2^25) | 25 |
| Puzzle #30 | [2^29, 2^30) | 30 |
| Puzzle #35 | [2^34, 2^35) | 35 |
| Puzzle #40 | [2^39, 2^40) | 40 |

For each:
- [ ] Solver returns a key `d`
- [ ] `d` matches the known answer exactly
- [ ] `d * G == Q` (the public key from the puzzle)
- [ ] `d ∈ [k1, k2]` (within the declared interval)
- [ ] Time-to-solve within 10× of theoretical `2.08 * sqrt(N)` operations

### Layer 2: Walk function determinism (`test_walk_determinism.py`)

The walk MUST be deterministic — given the same starting point, two runs produce
identical paths.

- [ ] Run walk from same start point twice, compare full point sequences for first 1000 steps
- [ ] Two kangaroos that land on the same point must produce identical next 100 points
- [ ] Jump table indices: `hash(P.x) mod r` is stable across runs and across processes

### Layer 3: Negation map correctness (`test_negation.py`)

Implement and test in **isolation** before integrating:

- [ ] Canonicalize: for random P, `canonical(P) == canonical(-P)`
- [ ] Canonicalize is idempotent: `canonical(canonical(P)) == canonical(P)`
- [ ] Solve a 25-bit toy puzzle WITH negation map → key matches known answer
- [ ] Solve same puzzle WITHOUT negation map → key matches known answer
- [ ] Empirically measure: collision rate ratio (with/without negation) ≈ 2.0

If the recovered key from the negation-map version equals `n - known_answer` instead
of `known_answer`, the sign-recovery logic is wrong.

### Layer 4: Teske jump table (`test_jump_table.py`)

- [ ] Sum of jump sizes / r ≈ N/2 (the balance constraint)
- [ ] All jump sizes are positive integers < N
- [ ] Distribution: variance of jump sizes minimized vs powers-of-2 baseline
- [ ] Empirical constant `c` measured on 30-bit puzzle: should be < 2.08 (target ~1.53)

### Layer 5: Checkpoint round-trip (`test_checkpoint.py`)

The single most important test for the Colab use case.

- [ ] Run solver for 10,000 steps, save checkpoint
- [ ] Load checkpoint into fresh process, run 10,000 more steps
- [ ] Final hashtable state must equal: 20,000-step continuous run state
- [ ] Solver from checkpoint must find the same key as continuous run on a toy puzzle
- [ ] Checkpoint file size scales linearly with number of stored DPs (no leak)
- [ ] Corrupted checkpoint detected and rejected with clear error
      (don't silently start fresh — that would erase your progress)

### Layer 6: Cross-validation against JeanLucPons (`test_cross_validation.py`)

For a 50-bit test puzzle (custom-generated, not from the puzzle series):

- [ ] Both implementations find the same key
- [ ] Operation counts within 2× of each other
- [ ] If our implementation is faster (due to negation/Teske), confirm by ratio

### Layer 7: 134-bit interval arithmetic (`test_large_interval.py`)

Specifically test that the 125-bit cap removal is correct:

- [ ] Interval boundary `k1 = 2^134`, `k2 = 2^135 - 1` parsed correctly
- [ ] Random walks from `k = 2^134 + ε` produce points matching reference scalar mult
- [ ] No 64-bit limb overflow: scalar offsets up to 2^135 stored and compared correctly
- [ ] Solve a custom 130-bit puzzle (generate the keypair yourself) end-to-end

### Layer 8: End-to-end integration test

Final smoke test before pointing it at #135:

- [ ] Generate a random 50-bit keypair `(d, Q)` where `d ∈ [2^49, 2^50)`
- [ ] Run solver against `Q` with a 6-hour timeout
- [ ] Verify recovered key equals `d`
- [ ] Repeat for 60-bit, then 70-bit
- [ ] If 70-bit solves correctly within expected time, the implementation is trusted

### Always-on runtime sanity checks

In production code (not just tests), enforce:

```python
def verify_solution(d: int, Q_bytes: bytes, k1: int, k2: int) -> bool:
    """Triple-check before declaring success."""
    if not (k1 <= d < k2):
        raise ValueError(f"Recovered key {d:x} outside interval [{k1:x}, {k2:x})")
    Q_recovered = scalar_mult(d, G)
    if serialize_point(Q_recovered) != Q_bytes:
        raise ValueError("Recovered key does not produce target public key")
    return True
```

This must run before any "FOUND IT!" log line, before any Bitcoin transaction is built,
before any celebration.

### Continuous health metrics during long runs

Even with all tests passing, log these metrics every checkpoint to detect regressions:

- DPs collected per hour (should be roughly constant)
- Walk steps per second (should match GPU benchmark)
- DP collision rate (drops to ~0 → walk is broken or negation map asymmetric)
- Hashtable size growth (linear with time → healthy; flat → DPs not being recorded)
- Memory usage (flat → healthy; growing without bound → leak)

---

## Architecture (single Colab instance)

### Directory structure

```
kangaroo/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── kangaroo/
│   ├── __init__.py
│   ├── curve.py               # secp256k1 field & group arithmetic
│   ├── kangaroo.py            # core algorithm (CPU reference)
│   ├── jump_table.py          # Teske jump set construction
│   ├── dp_table.py            # distinguished points hashtable
│   ├── negation.py            # ±P canonicalization
│   ├── checkpoint.py          # save/load to Drive
│   └── verify.py              # key verification (always-on sanity checks)
├── gpu/
│   └── kangaroo.cu            # CUDA kernel (post-CPU validation)
├── colab/
│   └── solver.ipynb           # mount Drive, load checkpoint, run, save, exit
├── tests/
│   ├── test_curve.py
│   ├── test_kangaroo_small.py
│   ├── test_walk_determinism.py
│   ├── test_negation.py
│   ├── test_jump_table.py
│   ├── test_checkpoint.py
│   ├── test_cross_validation.py
│   ├── test_large_interval.py
│   └── test_integration.py
└── scripts/
    └── extract_pubkey.py      # extract puzzle #135 public key from blockchain
```

### Implementation order (strict)

Each step requires its tests to pass before moving on:

1. **`curve.py`** — pure Python secp256k1, validated against `coincurve`
   → `test_curve.py` passes
2. **CPU reference Kangaroo** — no optimizations, simplest possible code
   → `test_kangaroo_small.py` passes for puzzles #20, #25, #30
3. **Checkpoint system** — save/load hashtable + walker state to disk
   → `test_checkpoint.py` passes
4. **Negation map** — added to CPU reference, tested in isolation
   → `test_negation.py` passes, recovered keys still correct
5. **Teske jump table** — measure constant improvement on 30-bit puzzles
   → `test_jump_table.py` passes
6. **134-bit interval support** — extend integer width throughout
   → `test_large_interval.py` passes
7. **CUDA kernel** — port logic from JeanLucPons with negation + Teske + 134-bit support
   → `test_cross_validation.py` passes against CPU reference
8. **Colab notebook** — Drive mount, checkpoint lifecycle, idle session detection
   → manual test: run for 30 min, kill kernel, restart, verify resume works
9. **Point at #135** — only after all of the above

### Single-instance Colab lifecycle

```
Session start:
  ├─ Mount Google Drive
  ├─ Check for existing /drive/MyDrive/kangaroo_135/checkpoint.work
  ├─ If exists: load it (DPs hashtable + per-walker state)
  └─ If not: initialize fresh from puzzle #135 public key

Run loop (every batch):
  ├─ Execute walk steps on GPU
  ├─ Detect collisions, recover candidate key
  ├─ verify_solution() — triple-check
  ├─ Every 20 minutes: save checkpoint to Drive
  └─ On idle/timeout signal: save final checkpoint, exit cleanly

Session end:
  ├─ Final checkpoint save
  └─ Drive close
```

No work file merging, no multi-account orchestration, no contention.

---

## Key references

- **JeanLucPons/Kangaroo** — gold standard CUDA implementation, 125-bit limit
- **ZenulAbidin/Kangaroo-256** — Linux-buildable JLP fork extended to 256-bit
  intervals; the practical drop-in for puzzle #135. Build with
  `make gpu=1 ccap=75 all` (T4). Same input/work-file format as JLP.
- ~~Etarkangaroo~~ — Windows-only PureBasic; not usable on Colab
- **Teske (2001)** — "On random walks for Pollard's rho method"
- **Wiener & Zuccherato (1998)** — negation map for ECDLP
- **van Oorschot & Wiener (1996)** — parallel collision search
- **`coincurve`** Python library — reference for secp256k1 arithmetic in tests

---

## Open questions

- [ ] Confirm laptop GPU model (irrelevant if Colab-only, but useful to know)
- [x] Extract puzzle #135 public key from blockchain (`scripts/extract_pubkey.py`)
      → constants in `kangaroo/puzzle_135.py`, verified by `tests/test_puzzle_135.py`
- [ ] Decide: pure Python CPU reference, or use `coincurve` directly for the reference?
      (Pure Python is slower but easier to debug and ports cleanly to CUDA logic;
       `coincurve` is faster for testing but adds an opaque dependency)
- [ ] CUDA approach: Numba/CuPy from Python, or write `.cu` files and call via ctypes?
      (Numba is faster to iterate; raw CUDA is faster to run)

---

## Repo conventions (read before adding code)

Patterns established by what's already in the repo. Following them avoids a
round-trip of "why doesn't my X work".

- **Slow-test marker.** Minute-scale tests carry `@pytest.mark.slow`. Default
  `pytest` excludes them via `addopts = "-m 'not slow'"` in `pyproject.toml`.
  Opt in with `pytest -m slow` or per-file. Tag any new test expected to run
  longer than ~10 s.
- **`coincurve` is a dev dep, not a runtime dep.** Only `tests/test_curve.py`
  imports it (it's the independent reference for our pure-Python field
  arithmetic). If it's missing, run the rest with
  `pytest --ignore=tests/test_curve.py`; don't reflex-install. Production
  code under `kangaroo/` is pure stdlib by design — flag any new third-party
  import there before adding it.
- **Test helpers in `tests/`.** No `__init__.py`. Pytest's rootdir prepend
  means helper modules placed there import as bare names
  (`import jlp_runner`), not `from tests.x import …`.
- **Checkpoint schema versioning.** Any change to the `state` dict shape
  saved by `solve()` requires bumping `_VERSION` in
  `kangaroo/checkpoint.py`. Old files are then refused with
  `CorruptedCheckpoint` rather than silently misread.
- **Solver speed budget.** Pure-Python reference runs ~3–5 k iter/sec.
  Tractable test widths: <30 bits (seconds), <40 bits (a few minutes). A
  130-bit *width* puzzle is wall-clock infeasible; for >125-bit-scalar
  coverage, place a small-width interval at a 134-bit offset — see
  `tests/test_large_interval.py`.
- **Negation map default.** `negation=False` is the safe path. `negation=True`
  is the cycle-prone single-walker route (~60% seed convergence; the solver
  emits a runtime warning). Reserve it for negation-specific tests.
- **JLP cross-validation runs on Colab, not locally.** `tests/jlp_runner.py`
  skips cleanly when `$JLP_KANGAROO_BIN` is unset. Don't try to build
  JeanLucPons on the dev box (no CUDA here).
- **Logging, not print.** Pytest is configured with `log_cli=true` at INFO.
  Use module-level `log = logging.getLogger(__name__)`; `print()` calls
  bypass the captured-output flow and clutter test runs.

---

## Reality check (re-read this when motivation flags)

- One free T4 → ~6,300 years expected solve time
- Optimizations cut to ~3,000 years — still a lottery
- Probability of solving in 1 year: ~0.03%
- The point is **the implementation**, not the prize

Front-running risk on claim: real (puzzle #66 lost ~10%). Use Mara Slipstream or
similar private relay if a key is ever found.

Colab ToS: prohibits cryptocurrency mining. Whether Kangaroo on a published puzzle
counts is genuinely ambiguous. Single account, no automation gymnastics, low risk of
flagging.
