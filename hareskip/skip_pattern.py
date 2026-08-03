"""Pure stochastic skip-pattern generation for HareSkip.

Stdlib only (math / random / hashlib / dataclasses). No Forge, torch or
gradio imports, so this module is fully unit-testable without a model.

The pattern is generated for all steps at once (the max-streak constraint
needs the whole picture). A user-configurable **skip window** in the
progress domain (``progress_i = idx / (num_steps - 1)``, 0-based) decides
which steps are eligible to be skipped; steps outside the window are forced
full (``p = 0.0``, ``skip = False``). Skip decisions for eligible steps are
drawn from a seeded RNG for reproducibility; a zone-based max-skip-streak
constraint (3 zones: danger/middle/safe, with user-configurable boundaries)
then trims runs deterministically in a single left-to-right pass, applying
each step's own zone limit to the running consecutive-skip counter
(2026-08-03; see apply_max_streak_constraint). End-of-generation moderation
is handled by the skip-probability taper (probability_models.py), not a zone.

Skip window semantics (2026-07-10 user decision, WYSIWYG)
========================================================
The skip window replaces the former automatic ~5% guard rule. There is NO
hidden last-step safety net: if the user sets the window to ``(0.0, 1.0)``
every step (including the last) is eligible. The default ``(0.05, 0.95)``
reproduces the old ``max(1, round(0.05 * N))`` guards for 30 steps (idx
0, 1, 28, 29 forced full). The first step remains full in practice via the
shared first_call / missing_residual force-full in patcher.py — that is an
inference necessity, not a UI guard, and is independent of this window.

Naming discipline: only ``skip_probability`` / ``skip_density`` style names
are used. The rejected score-based and top-K selection concepts from the
design spec are deliberately not part of this module.
"""

import hashlib
import math
import random
from dataclasses import dataclass

from . import probability_models

# Default trajectory-coordinate zone boundaries (3-zone max-skip-streak model;
# see docs/SPEC-alpha.md for the 2026-07-10 removal of the former "final"
# zone -- end-of-generation moderation is handled by the skip-probability
# taper in probability_models.py instead). Boundaries are parameterized
# (see zone_from_z); with the default (low=-4.0, high=0.0):
#   danger: z < low   (-> streak 1)
#   middle: low <= z < high  (-> streak 2)
#   safe:   z >= high  (-> streak 3)
DEFAULT_ZONE_BOUNDARIES = (-4.0, 0.0)
DEFAULT_SKIP_WINDOW = (0.05, 0.95)

ZONE_MAX_STREAK = {
    "danger": 1,
    "middle": 2,
    "safe": 3,
}


def logsnr_proxy_from_t_now(t_now, eps=1e-6):
    """Map ``t_now`` to the logSNR proxy z = 2*ln((1 - t)/t).

    ``t_now`` is clamped to ``[eps, 1 - eps]`` to avoid singularities.
    """
    t = max(eps, min(1.0 - eps, t_now))
    return 2.0 * math.log((1.0 - t) / t)


def zone_from_z(z, boundaries=DEFAULT_ZONE_BOUNDARIES):
    """Classify a trajectory coordinate into a rough zone (3-zone model).

    ``boundaries`` is a ``(low, high)`` pair (default ``(-4.0, 0.0)``):
      z < low         -> "danger"
      low <= z < high -> "middle"
      z >= high       -> "safe"
    """
    low, high = boundaries
    if z < low:
        return "danger"
    if z < high:
        return "middle"
    return "safe"


def progress_for_step(idx, num_steps):
    """0-based progress of a step, ``idx / (num_steps - 1)``.

    A single-step generation (num_steps <= 1) yields 0.0.
    """
    if num_steps <= 1:
        return 0.0
    return idx / float(num_steps - 1)


def derive_skip_seed(image_seed, offset):
    """Deterministically derive a skip-sampler seed from the image seed.

    Uses SHA-256 (NOT the builtin ``hash()``, which is salted per process
    and would break reproducibility). Returns a value in ``[0, 2**63)``.
    """
    digest = hashlib.sha256(
        f"{int(image_seed)}|hareskip|{int(offset)}".encode("utf-8")
    ).hexdigest()
    return int(digest, 16) % (2 ** 63)


@dataclass
class SkipPattern:
    """Full generated skip pattern plus diagnostics for logging."""

    skip: list  # list[bool], per step
    z_by_step: list  # list[float]
    p_by_step: list  # list[float], skip probability per step (excluded = 0.0)
    params: dict
    skip_window: tuple  # (window_start, window_end) in progress domain
    zone_boundaries: tuple  # (low, high) in z domain
    guarded_steps: int  # count of window-excluded steps (for logging)
    skip_seed: int
    expected_skips_before_streak: float
    skip_count: int
    skipped_steps: list  # 1-based step numbers that are skipped
    probability_model: str
    aggressiveness: float
    num_steps: int


def apply_max_streak_constraint(
    skip, z_by_step, p_by_step, zone_boundaries=DEFAULT_ZONE_BOUNDARIES
):
    """Trim skip runs in place so no step exceeds its own zone's streak.

    Deterministic (no RNG), single left-to-right pass, ``skip`` is modified
    in place. A running counter holds the number of consecutive skips
    immediately preceding the current step. For each candidate step ``k``
    (``skip[k]`` is True), the step's *own* zone
    (``zone_from_z(z_by_step[k], zone_boundaries)``) supplies the limit: if
    the counter has already reached ``ZONE_MAX_STREAK[zone]`` the step is
    flipped back to full and the counter resets to 0; otherwise the skip is
    kept and the counter increments. Steps that were already full reset the
    counter to 0.

    ``p_by_step`` is accepted for signature compatibility (and callers pass
    it) but the per-step rule needs no tie-breaking, so it is unused.

    2026-08-03: this replaces the former run-level rule, which applied the
    *minimum* ``ZONE_MAX_STREAK`` over all zones present in a maximal run
    and greedily flipped the ``(p, z, index)``-argmin step. At high skip
    probability every step joined one run, so a single danger step imposed
    streak 1 on the whole schedule and the skip count collapsed (3 skips at
    p = 1.0 on the 30-step reference schedule, against 21 for the per-step
    trim, or 19 under the default skip window). See docs/SPEC-alpha.md §4.5.
    """
    counter = 0
    for k in range(len(skip)):
        if not skip[k]:
            counter = 0
            continue
        allowed = ZONE_MAX_STREAK[zone_from_z(z_by_step[k], zone_boundaries)]
        if counter >= allowed:
            skip[k] = False
            counter = 0
        else:
            counter += 1


def _draw_pattern(t_now_by_step, params, skip_window, num_steps, model, rng):
    """Draw one raw pattern (pre-constraint) and its z/p arrays.

    A step is eligible for skipping iff its progress lies within
    ``skip_window`` (inclusive on both ends). Excluded steps get ``p = 0.0``
    and ``skip = False`` (forced full).
    """
    window_start, window_end = skip_window
    z_by_step = []
    p_by_step = []
    skip = []
    for idx, t_now in enumerate(t_now_by_step):
        progress = progress_for_step(idx, num_steps)
        eligible = window_start <= progress <= window_end
        z = logsnr_proxy_from_t_now(t_now)
        p = model.skip_probability(z, params) if eligible else 0.0
        z_by_step.append(z)
        p_by_step.append(p)
        skip.append((rng.random() < p) if eligible else False)
    return skip, z_by_step, p_by_step


def _guarded_steps_count(num_steps, skip_window):
    """Number of steps excluded by the skip window (forced full)."""
    window_start, window_end = skip_window
    count = 0
    for idx in range(num_steps):
        progress = progress_for_step(idx, num_steps)
        if not (window_start <= progress <= window_end):
            count += 1
    return count


def generate_skip_pattern(
    t_now_by_step,
    aggressiveness,
    skip_seed,
    probability_model="sigmoid_band_v0.1",
    skip_window=DEFAULT_SKIP_WINDOW,
    zone_boundaries=DEFAULT_ZONE_BOUNDARIES,
    exact_target=None,
    max_resample=100,
):
    """Generate a stochastic skip pattern for all steps.

    Steps are 0-based for the progress/window logic. A step is eligible for
    skipping iff ``window_start <= progress_i <= window_end`` where
    ``progress_i = idx / (num_steps - 1)``. Excluded steps get ``p = 0.0``
    and ``skip = False``. Eligible steps skip when ``rng.random() < p``. The
    zone-based max-streak constraint (3-zone model, ``zone_boundaries``) is
    then applied.

    ``expected_skips_before_streak`` is the sum of ``p`` over eligible steps
    (deterministic, independent of the RNG draws); excluded steps carry
    ``p == 0.0`` so summing all ``p`` is equivalent.

    If ``exact_target`` is not None, the whole pattern is re-rolled up to
    ``max_resample`` times. Each attempt uses ``random.Random(skip_seed +
    attempt)`` so attempts are deterministic. The first attempt whose
    post-constraint ``skip_count == exact_target`` is accepted; otherwise
    the attempt minimising ``abs(skip_count - target)`` (first found on
    ties) is kept. Impossible targets never raise.
    """
    num_steps = len(t_now_by_step)
    model = probability_models.get_model(probability_model)
    params = model.params_from_aggressiveness(aggressiveness)

    def build(rng):
        skip, z_by_step, p_by_step = _draw_pattern(
            t_now_by_step, params, skip_window, num_steps, model, rng
        )
        apply_max_streak_constraint(skip, z_by_step, p_by_step, zone_boundaries)
        return skip, z_by_step, p_by_step

    if exact_target is None:
        rng = random.Random(skip_seed)
        skip, z_by_step, p_by_step = build(rng)
    else:
        best = None  # (distance, skip, z_by_step, p_by_step)
        chosen = None
        for attempt in range(max_resample):
            rng = random.Random(skip_seed + attempt)
            cand_skip, cand_z, cand_p = build(rng)
            cand_count = sum(1 for s in cand_skip if s)
            if cand_count == exact_target:
                chosen = (cand_skip, cand_z, cand_p)
                break
            distance = abs(cand_count - exact_target)
            if best is None or distance < best[0]:
                best = (distance, cand_skip, cand_z, cand_p)
        if chosen is not None:
            skip, z_by_step, p_by_step = chosen
        else:
            _dist, skip, z_by_step, p_by_step = best

    # expected_skips_before_streak: sum of p over eligible steps. Excluded
    # steps carry p == 0.0 so summing all p is equivalent and robust.
    expected = sum(p_by_step)
    skipped_steps = [idx + 1 for idx, s in enumerate(skip) if s]
    skip_count = len(skipped_steps)
    guarded_steps = _guarded_steps_count(num_steps, skip_window)

    return SkipPattern(
        skip=skip,
        z_by_step=z_by_step,
        p_by_step=p_by_step,
        params=params,
        skip_window=tuple(skip_window),
        zone_boundaries=tuple(zone_boundaries),
        guarded_steps=guarded_steps,
        skip_seed=skip_seed,
        expected_skips_before_streak=expected,
        skip_count=skip_count,
        skipped_steps=skipped_steps,
        probability_model=probability_model,
        aggressiveness=aggressiveness,
        num_steps=num_steps,
    )
