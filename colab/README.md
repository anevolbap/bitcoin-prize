# Colab notebooks

Notebooks intended to run on Google Colab against a T4 GPU runtime.

## `smoke.ipynb` — build + smoke test (item #2 in the gap list)

Verifies the full toolchain end-to-end on a tiny synthetic puzzle:

1. confirms a T4 is attached
2. clones and builds **RetiredC/RCKangaroo** (`make CUDA_PATH=/usr/local/cuda`;
   sm_75 is in the default flags)
3. generates a 60-bit synthetic puzzle via `scripts/make_synthetic_puzzle.py`
4. runs the GPU solver against it (`-gpu 0 -dp 14 -range 59 -start 8…0 -pubkey 03…`)
5. parses `RESULTS.TXT` (RCKangaroo writes the recovered key here) and asserts
   it matches the known answer in the manifest

### Why RCKangaroo

- **170-bit max range** (covers puzzle #135's 134-bit need)
- **SOTA equivalence-class negation** → ~1.15·√n ops vs JLP's 2.08·√n; ~1.8×
  faster, materially improves the lottery odds
- 130 ⭐, GPL-3.0, active (v3.1 Nov 2025)
- Linux/CUDA buildable, T4 (sm_75) included in default `NVCCFLAGS`
- Used by puzzle-hunting community; PSCKangaroo (a fork) explicitly cites
  puzzle #135's pubkey in its examples

### What we tried first

- **JLP** — gold standard, smoke-passed on T4, but 125-bit cap means it can't
  ever attempt puzzle #135.
- **Kangaroo-256 (ZenulAbidin)** — 256-bit but hangs on Colab T4 across all
  bit widths and `dp_bits` we tried (pubkey checked against `coincurve`, so
  the hang isn't on our side). Likely an edge case in its 256-bit arithmetic
  or auto-DP picker; not worth debugging when RCKangaroo just works.
- **Etarkangaroo (Etayson)** — Windows-only PureBasic, can't build on Linux.

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
