# Colab notebooks

Notebooks intended to run on Google Colab against a T4 GPU runtime.

## `smoke.ipynb` — build + smoke test (item #2 in the gap list)

Verifies the full toolchain end-to-end on a tiny synthetic puzzle:

1. confirms a T4 is attached
2. clones and builds Kangaroo-256 with `ccap=75`
3. generates a 60-bit synthetic puzzle via `scripts/make_synthetic_puzzle.py`
   (~1 second on T4)
4. runs the GPU solver against it
5. asserts the recovered key matches the known answer in the manifest

If this passes, the build path is good and we can move on to the production
solver notebook (Drive mount, work-file checkpointing, idle-clean exit).

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
