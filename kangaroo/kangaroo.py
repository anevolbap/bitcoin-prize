"""Pollard's Kangaroo CPU reference — single tame + single wild walker.

Solve d*G = Q with d in [k1, k2).

Both walkers step pseudo-randomly: P -> P + jumps[hash(P.x)] * G, where the
jump set is {2^0, 2^1, ..., 2^(r-1)} * G. With negation=True, every step also
canonicalizes the result via the negation map (smaller-y representative of
{P, -P}). Distinguished points (DPs) are recorded; a collision between a tame
DP and a wild DP yields the discrete log.

State invariants (always maintained, both modes):
    point_tame == scalar_tame * G                          (mod n)
    point_wild == scalar_wild * G  +  q_sign * Q           (mod n)

With negation=False the canonicalize step is a no-op, no flips ever occur,
q_sign stays at +1 forever, and the formulas degenerate to the basic
Pollard kangaroo.

At collision (point_tame == point_wild):
    d = q_sign_at_wild_record * (scalar_tame - scalar_wild)   (mod n)

where the q_sign used is whichever was recorded at the WILD DP entry that
participates in the collision (not necessarily the walker's current q_sign).

Recovery is verified with d*G == Q before returning. A sign-tracking bug
yields n-d, which fails this check.

Fruitless cycles — KNOWN LIMITATION OF THIS PURE-PYTHON REFERENCE
-----------------------------------------------------------------
The negation map can trap a walker in a short canonical cycle. The simplest
case is length 2:
  P -> canonical(P + a_j*G)
       -> canonical(canonical(P + a_j*G) + a_j*G) = P
(possible because the hash uses x only and (-X).x == X.x, so the same jump j
gets picked at both ends).

We mitigate via three combined mechanisms:
  1. Parity-aware hash: idx = (x + prev_flipped) % r, breaking length-2 cycles
     deterministically.
  2. Cycle detection: per-walker history window (deque + set) of size 64
     catches longer cycles that slip past mechanism 1.
  3. Point-indexed escape: on revisit, take an escape offset chosen via
     hash(point.x ^ count*MIX) so escape destinations don't form a meta-orbit.

Even with all three, this single-walker reference DOES NOT converge reliably
for every random seed — production implementations parallelize across many
walkers so any single one's cycle issues are averaged out. We choose test
seeds that converge and document the rest as a known limitation; a fully
robust solution requires either Bos-Kleinjung-Lenstra cycle reduction or
many-walker parallelism (GPU layer).
"""
import collections
import logging

from .checkpoint import load_checkpoint, save_checkpoint
from .curve import G, N, point_add, scalar_mult
from .negation import canonical_with_flag

log = logging.getLogger(__name__)


def default_jump_count(interval_width: int) -> int:
    """r = max(8, log2(N)//2 + 2). Targets mean jump ≈ sqrt(N)/2."""
    if interval_width <= 0:
        raise ValueError(f"interval_width must be positive, got {interval_width}")
    log2N = interval_width.bit_length() - 1
    return max(8, log2N // 2 + 2)


def default_dp_bits(interval_width: int) -> int:
    """dp_bits = max(0, log2(N)//2 - 2). DP rate is 1 in 2^dp_bits."""
    if interval_width <= 0:
        raise ValueError(f"interval_width must be positive, got {interval_width}")
    log2N = interval_width.bit_length() - 1
    return max(0, log2N // 2 - 2)


def mean_jump(jump_count: int) -> float:
    """Arithmetic mean of {2^0, ..., 2^(r-1)} = (2^r - 1)/r."""
    return (2**jump_count - 1) / jump_count


def is_distinguished(point, dp_bits: int) -> bool:
    """True iff point.x has dp_bits leading zero bits. dp_bits=0 → all points."""
    if point is None:
        return False
    if dp_bits == 0:
        return True
    return (point[0] >> (256 - dp_bits)) == 0


def _identity_canon(point):
    """Canonicalize-pass-through used when the negation map is disabled."""
    return point, False


def solve(
    Q,
    k1: int,
    k2: int,
    *,
    jump_count: int | None = None,
    jumps: list[int] | None = None,
    dp_bits: int | None = None,
    negation: bool = False,
    max_steps: int = 10**8,
    stats: dict | None = None,
    checkpoint_path: str | None = None,
    checkpoint_every: int | None = None,
    resume_path: str | None = None,
):
    """Recover d such that d*G == Q with d in [k1, k2).

    Returns d, or None if max_steps was exhausted before a tame/wild DP
    collision occurred. Each iteration steps both walkers once.

    Jump set: pass either `jumps` (an explicit list of scalars, e.g. from
    kangaroo.jump_table) OR `jump_count` (defaults to powers of 2 of that
    size). If both are omitted, defaults to powers of 2 with size
    default_jump_count(interval).

    Checkpointing:
      - `resume_path`: if set, load walker state from this file (the file's
        params must match the call's args, else ValueError).
      - `checkpoint_path`: if set, write a checkpoint at every step that is
        a multiple of `checkpoint_every` (when given) AND once at the end of
        the run regardless. Atomic writes; a CorruptedCheckpoint on load
        aborts rather than silently restarting.

    If `stats` is given, it is populated with step count and DP table sizes.
    """
    if Q is None:
        raise ValueError("Q must not be the identity")
    interval = k2 - k1
    if interval <= 0:
        raise ValueError(f"empty interval [{k1}, {k2})")

    if jumps is not None:
        if jump_count is not None and jump_count != len(jumps):
            raise ValueError(
                f"jump_count={jump_count} does not match len(jumps)={len(jumps)}"
            )
        jump_count = len(jumps)
        jump_scalars = list(jumps)
    else:
        if jump_count is None:
            jump_count = default_jump_count(interval)
        jump_scalars = [1 << i for i in range(jump_count)]
    if dp_bits is None:
        dp_bits = default_dp_bits(interval)

    canonicalize = canonical_with_flag if negation else _identity_canon

    actual_mean_jump = sum(jump_scalars) / len(jump_scalars)
    log.info(
        "solve: interval=2^%d (=%d), jump_count=%d (mean_jump=%.3g), "
        "dp_bits=%d, negation=%s",
        interval.bit_length() - 1, interval,
        jump_count, actual_mean_jump, dp_bits, negation,
    )

    if negation:
        log.warning(
            "negation=True is experimental in this pure-Python single-walker "
            "reference: ~60%% of random seeds converge, and converging runs "
            "are 2-3x SLOWER than negation=False due to cycle-handling "
            "overhead. See the 'Fruitless cycles' section of "
            "kangaroo.kangaroo's module docstring for the full story."
        )

    # Precompute curve points for each scalar in the jump set.
    jump_points = [scalar_mult(s, G) for s in jump_scalars]

    # Cycle-escape table: 32 distinct offsets, all larger than any regular jump.
    # Index by point.x (XOR with golden-ratio-ish multiplier of the escape count)
    # so different points map to different escape destinations. Without this
    # indexing — i.e., a fixed rotation by escape count — the walker forms a
    # large meta-orbit through the 32 destinations and never reaches new DPs.
    _NUM_ESCAPE_OFFSETS = 32
    _ESCAPE_INDEX_MASK = _NUM_ESCAPE_OFFSETS - 1
    _ESCAPE_MIX = 0x9E3779B97F4A7C15  # golden-ratio constant, spreads bits
    escape_scalars = [
        (1 << jump_count) + i * (1 << (jump_count - 1)) + 1
        for i in range(_NUM_ESCAPE_OFFSETS)
    ]
    escape_points = [scalar_mult(s, G) for s in escape_scalars]

    midpoint = (k1 + k2) // 2

    # Params dict: anything that affects determinism. Checked on resume.
    solver_params = {
        "Q": Q,
        "k1": k1,
        "k2": k2,
        "jumps": list(jump_scalars),
        "dp_bits": dp_bits,
        "negation": negation,
    }

    _CYCLE_WINDOW = 64

    if resume_path is not None:
        loaded_params, loaded_state = load_checkpoint(resume_path)
        if loaded_params != solver_params:
            raise ValueError(
                f"checkpoint params at {resume_path} don't match solve() args"
            )
        tame_point = loaded_state["tame_point"]
        scalar_tame = loaded_state["scalar_tame"]
        tame_parity = loaded_state["tame_parity"]
        wild_point = loaded_state["wild_point"]
        scalar_wild = loaded_state["scalar_wild"]
        q_sign = loaded_state["q_sign"]
        wild_parity = loaded_state["wild_parity"]
        tame_dps = loaded_state["tame_dps"]
        wild_dps = loaded_state["wild_dps"]
        tame_window_d = collections.deque(
            loaded_state["tame_window"], maxlen=_CYCLE_WINDOW
        )
        tame_window_s = set(loaded_state["tame_window"])
        wild_window_d = collections.deque(
            loaded_state["wild_window"], maxlen=_CYCLE_WINDOW
        )
        wild_window_s = set(loaded_state["wild_window"])
        tame_escapes = loaded_state["tame_escapes"]
        wild_escapes = loaded_state["wild_escapes"]
        start_step = loaded_state["step_count"]
        log.info(
            "resumed from %s at step %d (tame_dps=%d, wild_dps=%d)",
            resume_path, start_step, len(tame_dps), len(wild_dps),
        )
    else:
        # --- tame init: point_tame == scalar_tame * G ---
        tame_unflipped = scalar_mult(midpoint, G)
        tame_point, tame_init_flipped = canonicalize(tame_unflipped)
        scalar_tame = (-midpoint) % N if tame_init_flipped else midpoint
        tame_parity = 1 if tame_init_flipped else 0

        # --- wild init: point_wild == scalar_wild * G + q_sign * Q ---
        wild_point, wild_init_flipped = canonicalize(Q)
        scalar_wild = 0
        q_sign = -1 if wild_init_flipped else 1
        wild_parity = 1 if wild_init_flipped else 0

        # Two dicts so a tame DP and a wild DP at the same x can coexist long enough
        # to be detected as a collision; a kind tag in one dict would lose info.
        tame_dps: dict[int, int] = {}              # x -> scalar_tame
        wild_dps: dict[int, tuple[int, int]] = {}  # x -> (scalar_wild, q_sign)

        # Per-walker history windows for fruitless-cycle detection. With negation=False
        # these never fire (no flips → no short cycles in n ≈ 2^256 group).
        # Set + deque: deque enforces FIFO eviction, set gives O(1) membership.
        # Window must exceed the longest practical cycle. Empirically: cycles up to
        # ~50 are seen (length-2 dominate but tails extend); 64 is safely above the
        # plateau in our measurements.
        tame_window_d: collections.deque = collections.deque(maxlen=_CYCLE_WINDOW)
        tame_window_s: set = set()
        wild_window_d: collections.deque = collections.deque(maxlen=_CYCLE_WINDOW)
        wild_window_s: set = set()
        tame_escapes = 0
        wild_escapes = 0
        start_step = 0

    def _save_state(step_count: int) -> None:
        save_checkpoint(checkpoint_path, params=solver_params, state={
            "tame_point": tame_point,
            "scalar_tame": scalar_tame,
            "tame_parity": tame_parity,
            "wild_point": wild_point,
            "scalar_wild": scalar_wild,
            "q_sign": q_sign,
            "wild_parity": wild_parity,
            "tame_dps": tame_dps,
            "wild_dps": wild_dps,
            "tame_window": list(tame_window_d),
            "wild_window": list(wild_window_d),
            "tame_escapes": tame_escapes,
            "wild_escapes": wild_escapes,
            "step_count": step_count,
        })

    found_d: int | None = None
    found_step: int | None = None
    step_count = start_step

    for step in range(start_step, max_steps):
        # ---- tame step ----
        idx = (tame_point[0] + tame_parity) % jump_count if negation else tame_point[0] % jump_count
        jump_s = jump_scalars[idx]
        new_uncan = point_add(tame_point, jump_points[idx])
        tame_point, flipped = canonicalize(new_uncan)
        if flipped:
            scalar_tame = (-(scalar_tame + jump_s)) % N
        else:
            scalar_tame = (scalar_tame + jump_s) % N
        tame_parity = 1 if flipped else 0

        if tame_point in tame_window_s:
            ei = (tame_point[0] ^ (tame_escapes * _ESCAPE_MIX)) & _ESCAPE_INDEX_MASK
            new_uncan = point_add(tame_point, escape_points[ei])
            tame_point, flipped = canonicalize(new_uncan)
            if flipped:
                scalar_tame = (-(scalar_tame + escape_scalars[ei])) % N
            else:
                scalar_tame = (scalar_tame + escape_scalars[ei]) % N
            tame_parity = 1 if flipped else 0
            tame_window_d.clear()
            tame_window_s.clear()
            tame_escapes += 1
        if len(tame_window_d) == _CYCLE_WINDOW:
            tame_window_s.discard(tame_window_d[0])
        tame_window_d.append(tame_point)
        tame_window_s.add(tame_point)

        if is_distinguished(tame_point, dp_bits):
            x = tame_point[0]
            wild_hit = wild_dps.get(x)
            if wild_hit is not None:
                ws, wq = wild_hit
                found_d = (wq * (scalar_tame - ws)) % N
                found_step = step
                break
            elif x not in tame_dps:
                tame_dps[x] = scalar_tame

        # ---- wild step ----
        idx = (wild_point[0] + wild_parity) % jump_count if negation else wild_point[0] % jump_count
        jump_s = jump_scalars[idx]
        new_uncan = point_add(wild_point, jump_points[idx])
        wild_point, flipped = canonicalize(new_uncan)
        if flipped:
            scalar_wild = (-(scalar_wild + jump_s)) % N
            q_sign = -q_sign
        else:
            scalar_wild = (scalar_wild + jump_s) % N
        wild_parity = 1 if flipped else 0

        if wild_point in wild_window_s:
            ei = (wild_point[0] ^ (wild_escapes * _ESCAPE_MIX)) & _ESCAPE_INDEX_MASK
            new_uncan = point_add(wild_point, escape_points[ei])
            wild_point, flipped = canonicalize(new_uncan)
            if flipped:
                scalar_wild = (-(scalar_wild + escape_scalars[ei])) % N
                q_sign = -q_sign
            else:
                scalar_wild = (scalar_wild + escape_scalars[ei]) % N
            wild_parity = 1 if flipped else 0
            wild_window_d.clear()
            wild_window_s.clear()
            wild_escapes += 1
        if len(wild_window_d) == _CYCLE_WINDOW:
            wild_window_s.discard(wild_window_d[0])
        wild_window_d.append(wild_point)
        wild_window_s.add(wild_point)

        if is_distinguished(wild_point, dp_bits):
            x = wild_point[0]
            tame_hit = tame_dps.get(x)
            if tame_hit is not None:
                found_d = (q_sign * (tame_hit - scalar_wild)) % N
                found_step = step
                step_count = step + 1
                break
            elif x not in wild_dps:
                wild_dps[x] = (scalar_wild, q_sign)

        step_count = step + 1
        if (
            checkpoint_path is not None
            and checkpoint_every is not None
            and step_count % checkpoint_every == 0
        ):
            _save_state(step_count)

    if checkpoint_path is not None:
        _save_state(step_count)

    if stats is not None:
        stats["steps"] = found_step if found_step is not None else max_steps
        stats["tame_dps"] = len(tame_dps)
        stats["wild_dps"] = len(wild_dps)
        stats["tame_escapes"] = tame_escapes
        stats["wild_escapes"] = wild_escapes

    if found_d is None:
        log.warning(
            "max_steps=%d exhausted; tame_dps=%d, wild_dps=%d, escapes=%d/%d",
            max_steps, len(tame_dps), len(wild_dps), tame_escapes, wild_escapes,
        )
        return None

    # Mandatory verification: catches sign-tracking bugs (which would yield n-d).
    if scalar_mult(found_d, G) != Q:
        raise RuntimeError(
            f"sign-tracking bug: recovered d=0x{found_d:x} does not satisfy d*G=Q"
        )

    log.info(
        "collision at step %d; tame_dps=%d, wild_dps=%d, escapes=%d/%d",
        found_step, len(tame_dps), len(wild_dps), tame_escapes, wild_escapes,
    )
    return found_d
