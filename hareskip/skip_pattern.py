"""Pure stochastic skip-pattern generation for HareSkip.

Stdlib only (math / random / hashlib / dataclasses). No Forge, torch or
gradio imports, so this module is fully unit-testable without a model.

The pattern is generated for all steps at once (the max-streak constraint
needs the whole picture). Guards force the first/last ~5% of steps to full
computation. Skip decisions are drawn from a seeded RNG for reproducibility;
a zone-based max-skip-streak constraint (3 zones: danger/middle/safe) then
trims runs deterministically. End-of-generation moderation is handled by
the skip-probability taper (probability_models.py), not by a zone.

Naming discipline: only ``skip_probability`` / ``skip_density`` style names
are used. The rejected score-based and top-K selection concepts from the
design spec are deliberately not part of this module.
"""

import hashlib
import math
import random
from dataclasses import dataclass

from . import probability_models

# Trajectory-coordinate zone boundaries (3-zone max-skip-streak model;
# see docs/SPEC-alpha.md for the 2026-07-10 removal of the former "final"
# zone -- end-of-generation moderation is handled by the skip-probability
# taper in probability_models.py instead):
#   danger: z < -4
#   middle: -4 <= z < 0
#   safe:    z >= 0
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


def zone_from_z(z):
    """Classify a trajectory coordinate into a rough zone (3-zone model)."""
    if z < -4.0:
        return "danger"
    if z < 0.0:
        return "middle"
    return "safe"


def guard_count_for(num_steps):
    """Number of forced-full steps at each end (~5%, at least 1)."""
    return max(1, round(num_steps * 0.05))


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
    p_by_step: list  # list[float], skip probability per step (guards = 0.0)
    params: dict
    guard_count: int
    skip_seed: int
    expected_skips_before_streak: float
    skip_count: int
    skipped_steps: list  # 1-based step numbers that are skipped
    probability_model: str
    aggressiveness: float
    num_steps: int


def apply_max_streak_constraint(skip, z_by_step, p_by_step):
    """Trim skip runs in place so no run exceeds its allowed streak.

    Deterministic (no RNG). For each maximal run of consecutive skipped
    steps, the allowed streak is the most conservative
    ``ZONE_MAX_STREAK`` over the zones present in the run. While a run is
    too long, flip the step with the lowest ``(p, z, index)`` back to full
    and re-scan (a flip may split a run into shorter sub-runs). Terminates
    because each flip strictly reduces the number of skips.
    """
    while True:
        # Find the first maximal run that violates its allowed streak.
        n = len(skip)
        i = 0
        violated = None  # (start, end_exclusive, allowed)
        while i < n:
            if not skip[i]:
                i += 1
                continue
            j = i
            while j < n and skip[j]:
                j += 1
            # run is [i, j)
            allowed = min(
                ZONE_MAX_STREAK[zone_from_z(z_by_step[k])] for k in range(i, j)
            )
            if (j - i) > allowed:
                violated = (i, j, allowed)
                break
            i = j

        if violated is None:
            return

        start, end, _allowed = violated
        # argmin over the run by (p, z, index).
        best = None
        for k in range(start, end):
            key = (p_by_step[k], z_by_step[k], k)
            if best is None or key < best[0]:
                best = (key, k)
        skip[best[1]] = False
        # Loop again: re-scan from scratch so newly created sub-runs are
        # correctly re-evaluated.


def _draw_pattern(t_now_by_step, params, guard_count, num_steps, model, rng):
    """Draw one raw pattern (pre-constraint) and its z/p arrays."""
    z_by_step = []
    p_by_step = []
    skip = []
    for idx, t_now in enumerate(t_now_by_step):
        step_no = idx + 1
        is_guard = step_no <= guard_count or step_no > num_steps - guard_count
        z = logsnr_proxy_from_t_now(t_now)
        p = 0.0 if is_guard else model.skip_probability(z, params)
        z_by_step.append(z)
        p_by_step.append(p)
        skip.append(False if is_guard else (rng.random() < p))
    return skip, z_by_step, p_by_step


def generate_skip_pattern(
    t_now_by_step,
    aggressiveness,
    skip_seed,
    probability_model="sigmoid_band_v0.1",
    exact_target=None,
    max_resample=100,
):
    """Generate a stochastic skip pattern for all steps.

    Steps are 1-based for guard logic. Guard steps get ``p = 0.0`` and
    ``skip = False``. Non-guard steps skip when ``rng.random() < p``. The
    zone-based max-streak constraint (3-zone model) is then applied.

    ``expected_skips_before_streak`` is the sum of ``p`` over non-guard
    steps (deterministic, independent of the RNG draws).

    If ``exact_target`` is not None, the whole pattern is re-rolled up to
    ``max_resample`` times. Each attempt uses ``random.Random(skip_seed +
    attempt)`` so attempts are deterministic. The first attempt whose
    post-constraint ``skip_count == exact_target`` is accepted; otherwise
    the attempt minimising ``abs(skip_count - target)`` (first found on
    ties) is kept. Impossible targets never raise.
    """
    num_steps = len(t_now_by_step)
    guard_count = guard_count_for(num_steps)
    model = probability_models.get_model(probability_model)
    params = model.params_from_aggressiveness(aggressiveness)

    def build(rng):
        skip, z_by_step, p_by_step = _draw_pattern(
            t_now_by_step, params, guard_count, num_steps, model, rng
        )
        apply_max_streak_constraint(skip, z_by_step, p_by_step)
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

    # expected_skips_before_streak: sum of p over non-guard steps. Guard
    # steps carry p == 0.0 so summing all p is equivalent and robust.
    expected = sum(p_by_step)
    skipped_steps = [idx + 1 for idx, s in enumerate(skip) if s]
    skip_count = len(skipped_steps)

    return SkipPattern(
        skip=skip,
        z_by_step=z_by_step,
        p_by_step=p_by_step,
        params=params,
        guard_count=guard_count,
        skip_seed=skip_seed,
        expected_skips_before_streak=expected,
        skip_count=skip_count,
        skipped_steps=skipped_steps,
        probability_model=probability_model,
        aggressiveness=aggressiveness,
        num_steps=num_steps,
    )
