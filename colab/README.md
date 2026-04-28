# Colab notebooks

Notebooks intended to run on Google Colab against a T4 GPU runtime.

## `smoke.ipynb` — build + smoke test (item #2 in the gap list)

Verifies the full toolchain end-to-end on a tiny synthetic puzzle:

1. confirms a T4 is attached
2. clones and builds **JeanLucPons/Kangaroo** with `ccap=75` (Makefile path
   overrides for Colab's CUDA 12 / g++ 11 layout)
3. generates a 60-bit synthetic puzzle via `scripts/make_synthetic_puzzle.py`
4. runs the GPU solver against it
5. asserts the recovered key matches the known answer in the manifest

### Why plain JLP and not Kangaroo-256?

K-256 is the eventual production target — puzzle #135 needs a 134-bit interval
which JLP can't handle (125-bit cap). But on Colab T4, K-256 misbehaves on
narrow intervals: GPU runs at ~600 MK/s indefinitely without finding the key
(verified across multiple bit widths and `dp_bits` values; pubkey generation
checked against `coincurve` and matches).

Plain JLP is the gold standard with mature defaults; using it for the smoke
test validates the rest of the pipeline (build, puzzle gen, invocation, key
verification) while the K-256 issue is investigated separately.

**TODO** before puzzle-#135 attempt: either find a working >125-bit fork on
T4, fix K-256 (likely a narrow-interval edge case in its 256-bit arithmetic
or auto-DP picker), or fork JLP and patch the cap ourselves.

### How to run

The notebook needs the project tree at `PROJECT_DIR` (set in cell 3, default
`/content/bitcoin-prize`). Three options to put it there:

- **git clone** (cleanest, once the repo is pushed somewhere reachable):
  ```python
  !git clone https://github.com/<you>/bitcoin-prize /content/bitcoin-prize
  ```
- **Files panel upload**: drag the project folder into Colab's Files sidebar.
- **Drive mount**: mount Drive, then point `PROJECT_DIR` at e.g.
  `/content/drive/MyDrive/bitcoin-prize`.

Then run cells top to bottom. The final cell asserts a `PASS` if the recovered
private key matches the synthetic-puzzle manifest.

### Why a synthetic puzzle, not a real puzzle from the series

A 60-bit synthetic key is small enough to solve in seconds. Solving a real
puzzle from the series (#65+) on a smoke test would take minutes-to-hours and
the recovered key has economic value — neither is what you want from a
"is the toolchain working" check.

## Future: `solver.ipynb`

Targets the real puzzle #135 (constants in `kangaroo/puzzle_135.py`). Adds:

- Drive mount + checkpoint round-trip across the Colab 12-hour session limit
- work-file save every ~20 min (`-w state.work -wi 1200`)
- resume detection (`-i state.work` if exists)
- clean exit on idle/timeout signal
- a kill-switch for the unlikely event a real key is found, so the recovered
  key isn't broadcast over an untrusted channel

Not yet written.
