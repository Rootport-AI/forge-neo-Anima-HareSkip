"""Unit tests for hareskip.skip_pattern (pure stdlib + pytest)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hareskip import skip_pattern as sp


# --- fixtures ---------------------------------------------------------------


def _t_now_schedule(n=30):
    """Realistic descending t_now schedule in (0, 1).

    Linearly spaced from 0.999 down to 0.003 (descending). z therefore
    increases as generation progresses, matching the design doc.
    """
    hi, lo = 0.999, 0.003
    if n == 1:
        return [hi]
    return [hi + (lo - hi) * i / (n - 1) for i in range(n)]


def _t_now_from_z(z):
    """Inverse of logsnr_proxy_from_t_now: t = 1 / (1 + exp(z / 2))."""
    import math

    return 1.0 / (1.0 + math.exp(z / 2.0))


# --- coordinate helpers -----------------------------------------------------


def test_logsnr_proxy_clamped_and_monotone():
    # Descending t_now -> increasing z.
    sched = _t_now_schedule(30)
    zs = [sp.logsnr_proxy_from_t_now(t) for t in sched]
    assert all(zs[i] < zs[i + 1] for i in range(len(zs) - 1))
    # Clamp protects extremes.
    assert sp.logsnr_proxy_from_t_now(0.0) > 0.0  # t clamped to eps
    assert sp.logsnr_proxy_from_t_now(1.0) < 0.0


@pytest.mark.parametrize(
    "z,zone",
    [
        (-4.0001, "danger"),
        (-4.0, "middle"),
        (-0.0001, "middle"),
        (0.0, "safe"),
        (3.9999, "safe"),
        (4.0, "safe"),
        (10.0, "safe"),
        (-100.0, "danger"),
    ],
)
def test_zone_boundaries_default(z, zone):
    # Default boundaries (low=-4.0, high=0.0).
    assert sp.zone_from_z(z) == zone


@pytest.mark.parametrize(
    "z,zone",
    [
        # boundaries (low=-2.0, high=2.0):
        (-3.0, "danger"),   # z < low
        (-2.0, "middle"),   # z == low -> middle
        (-1.0, "middle"),
        (1.0, "middle"),    # low <= z < high
        (2.0, "safe"),      # z == high -> safe
        (3.0, "safe"),      # z >= high
    ],
)
def test_zone_boundaries_custom(z, zone):
    assert sp.zone_from_z(z, boundaries=(-2.0, 2.0)) == zone


def test_zone_max_streak_table():
    assert sp.ZONE_MAX_STREAK == {
        "danger": 1,
        "middle": 2,
        "safe": 3,
    }


def test_progress_for_step():
    assert sp.progress_for_step(0, 30) == 0.0
    assert sp.progress_for_step(29, 30) == pytest.approx(1.0)
    assert sp.progress_for_step(0, 1) == 0.0  # single-step guard


# --- skip window (replaces the former ~5% guard rule) -----------------------


def test_default_window_reproduces_old_30_step_guards():
    # Default window (0.05, 0.95) with progress = idx / 29 reproduces the old
    # max(1, round(0.05 * 30)) = 2 guards: idx 0, 1, 28, 29 are forced full
    # with p == 0.0 and never skipped.
    sched = _t_now_schedule(30)
    pat = sp.generate_skip_pattern(
        sched, aggressiveness=1.0, skip_seed=7, probability_model="sigmoid_band_v0.1",
    )
    assert pat.skip_window == (0.05, 0.95)
    assert pat.guarded_steps == 4  # idx 0, 1, 28, 29
    for idx in (0, 1, 28, 29):
        assert pat.skip[idx] is False
        assert pat.p_by_step[idx] == 0.0
    # Interior boundary steps ARE eligible (nonzero p in the band).
    for idx in (2, 27):
        assert pat.p_by_step[idx] > 0.0


def test_full_window_makes_every_step_eligible():
    # WYSIWYG (2026-07-10 user decision): window (0.0, 1.0) means NO hidden
    # safety net — every step, including the last, is eligible for skipping.
    # There are zero window-excluded (guarded) steps.
    sched = _t_now_schedule(30)
    pat = sp.generate_skip_pattern(
        sched, aggressiveness=1.0, skip_seed=0, skip_window=(0.0, 1.0),
        probability_model="sigmoid_band_v0.1",
    )
    assert pat.skip_window == (0.0, 1.0)
    assert pat.guarded_steps == 0
    # Contrast with the default window, which forces the ends full: under
    # (0.0, 1.0) the first and last steps are NOT zeroed by a window guard,
    # so their p equals the raw band value (idx 0's z is far below the band,
    # so its p is ~0 by the band, not by a guard). The eligibility mask is
    # what changed: no step is force-zeroed by the window.
    default = sp.generate_skip_pattern(
        sched, aggressiveness=1.0, skip_seed=0, probability_model="sigmoid_band_v0.1",
    )
    # Default window zeroes idx 1 (progress 0.0345 < 0.05) even though its raw
    # band p is large; the full window leaves it at the raw band value.
    assert default.p_by_step[1] == 0.0
    assert pat.p_by_step[1] > 0.0

    # The last step CAN be skipped when its z falls inside the probability
    # band. Build a schedule whose final t_now puts z near the band centre so
    # a suitable seed skips the last step (proving it is genuinely eligible,
    # not forced full by the window). z is swept from -3.0 up to 4.5, keeping
    # the final step comfortably inside the band (z_exit=5.2 at a=1.0).
    band_sched = [_t_now_from_z(-3.0 + 7.5 * i / 29) for i in range(30)]
    last_z = sp.logsnr_proxy_from_t_now(band_sched[-1])
    assert 4.0 < last_z < 5.0  # inside the band
    last_can_skip = False
    for seed in range(200):
        p = sp.generate_skip_pattern(
            band_sched, aggressiveness=1.0, skip_seed=seed, skip_window=(0.0, 1.0),
            probability_model="sigmoid_band_v0.1",
        )
        if p.skip[-1]:
            last_can_skip = True
            break
    assert last_can_skip, "last step must be skippable under window (0.0, 1.0)"


def test_narrow_window_excludes_more_steps():
    sched = _t_now_schedule(30)
    pat = sp.generate_skip_pattern(
        sched, aggressiveness=1.0, skip_seed=1, skip_window=(0.3, 0.7),
        probability_model="sigmoid_band_v0.1",
    )
    # progress in [0.3, 0.7] -> idx 9..20 inclusive eligible (12 steps),
    # so 18 guarded.
    eligible = [i for i in range(30) if pat.p_by_step[i] > 0.0 or pat.skip[i]]
    for idx in eligible:
        prog = idx / 29.0
        assert 0.3 <= prog <= 0.7
    assert pat.guarded_steps == 30 - len(
        [i for i in range(30) if 0.3 <= i / 29.0 <= 0.7]
    )


# --- custom zone boundaries -------------------------------------------------


def test_custom_zone_boundaries_shift_classification():
    # boundaries (-2, 2): z=-3 -> danger; z=1 -> middle; z=3 -> safe.
    assert sp.zone_from_z(-3.0, (-2.0, 2.0)) == "danger"
    assert sp.zone_from_z(1.0, (-2.0, 2.0)) == "middle"
    assert sp.zone_from_z(3.0, (-2.0, 2.0)) == "safe"
    # Same z values under default boundaries classify differently.
    assert sp.zone_from_z(1.0) == "safe"  # default high=0.0
    assert sp.zone_from_z(-3.0) == "middle"  # default low=-4.0


def test_generate_records_zone_boundaries():
    sched = _t_now_schedule(30)
    pat = sp.generate_skip_pattern(
        sched, aggressiveness=0.5, skip_seed=3, zone_boundaries=(-2.0, 2.0),
        probability_model="sigmoid_band_v0.1",
    )
    assert pat.zone_boundaries == (-2.0, 2.0)


def test_custom_zone_boundaries_change_streak_trimming():
    # A run of 3 skips at z=[0.5, 1.0, 1.5] is entirely "safe" under default
    # boundaries (high=0.0) -> allowed 3, untrimmed. Under boundaries with
    # high=2.0 those become "middle" -> allowed 2, so one is trimmed.
    # (Unchanged by the 2026-08-03 per-step rule: single-zone runs behave
    # identically, only the count of survivors is asserted.)
    skip = [True, True, True]
    z = [0.5, 1.0, 1.5]
    p = [0.30, 0.10, 0.20]
    a = list(skip)
    sp.apply_max_streak_constraint(a, z, list(p), zone_boundaries=(-4.0, 0.0))
    assert a == [True, True, True]  # all safe under default
    b = list(skip)
    sp.apply_max_streak_constraint(b, z, list(p), zone_boundaries=(-4.0, 2.0))
    assert sum(b) == 2  # middle zone allows only 2


# --- derive_skip_seed -------------------------------------------------------


def test_derive_skip_seed_pinned():
    # Pinned literal: catches accidental use of builtin hash() (salted).
    assert sp.derive_skip_seed(12345, 0) == 3695650839502921262
    # Bounded to [0, 2**63).
    assert 0 <= sp.derive_skip_seed(12345, 0) < 2 ** 63


def test_derive_skip_seed_offset_differs():
    a = sp.derive_skip_seed(12345, 0)
    b = sp.derive_skip_seed(12345, 1)
    assert a != b


# --- streak constraint (synthetic) ------------------------------------------


def test_streak_middle_run_trimmed_at_third_step():
    # 4-long True run entirely in the middle zone (-4 <= z < 0), allowed 2.
    # Expectation changed 2026-08-03: the per-step left-to-right rule breaks
    # the run at the first step whose counter has already reached the limit
    # (index 2), instead of at the argmin-p step -- which here happens to be
    # the same index, so the result is unchanged; p no longer influences it.
    skip = [True, True, True, True]
    z = [-1.0, -1.5, -2.0, -0.5]
    p = [0.30, 0.20, 0.10, 0.40]
    sp.apply_max_streak_constraint(skip, z, p)
    assert skip == [True, True, False, True]
    _assert_streaks_ok(skip, z)


def test_streak_ignores_p_and_z_ordering():
    # Expectation changed 2026-08-03: the old rule flipped the (p, z, index)
    # argmin (index 1, the lowest z), giving [T, F, T, T]. The per-step rule
    # has no tie-break at all -- position alone decides, so the middle-zone
    # 4-run always breaks at index 2 regardless of p/z ordering.
    skip = [True, True, True, True]
    z = [-1.0, -3.0, -2.0, -0.5]  # lowest z is index 1
    p = [0.25, 0.25, 0.25, 0.25]
    sp.apply_max_streak_constraint(skip, z, p)
    assert skip == [True, True, False, True]
    _assert_streaks_ok(skip, z)


def test_streak_cross_zone_danger_then_middle():
    # Expectation changed 2026-08-03: the old run-level rule took the run
    # minimum (min(1, 2) = 1) and flipped the danger step out, giving
    # [F, T, T]. Under the per-step rule the danger step at index 0 starts
    # the run with counter 0 < 1, so it is *kept*; the limit that bites is
    # the middle limit 2 at index 2.
    skip = [True, True, True]
    z = [-5.0, -3.0, -2.0]  # danger, middle, middle
    p = [0.10, 0.30, 0.20]
    sp.apply_max_streak_constraint(skip, z, p)
    assert skip == [True, True, False]
    _assert_streaks_ok(skip, z)


def test_streak_danger_step_after_a_skip_is_flipped():
    # Expectation changed 2026-08-03: the old rule flipped the danger step
    # because it was the argmin-p of the whole run. Under the per-step rule
    # index 1 is flipped for a different reason -- it is a danger step
    # (allowed 1) preceded by one skip, so its counter already equals its
    # limit. Index 0 and the following middle pair survive.
    skip = [True, True, True, True]
    z = [-3.0, -5.0, -3.5, -2.0]  # middle, danger, middle, middle
    p = [0.40, 0.05, 0.30, 0.20]
    sp.apply_max_streak_constraint(skip, z, p)
    assert skip == [True, False, True, True]
    _assert_streaks_ok(skip, z)


def test_streak_safe_run_of_five_trimmed_to_three():
    # 5-run entirely in safe zone (z >= 0): allowed 3.
    skip = [True, True, True, True, True]
    z = [0.5, 1.0, 1.5, 2.0, 2.5]
    p = [0.50, 0.10, 0.20, 0.15, 0.50]
    sp.apply_max_streak_constraint(skip, z, p)
    assert _max_run(skip) <= 3
    _assert_streaks_ok(skip, z)


def test_streak_safe_run_spanning_former_final_boundary():
    # 3-zone model: the former z=4 "final" boundary no longer splits a run.
    # A run straddling z=4 stays a single safe-zone run (allowed 3).
    skip = [True, True, True]
    z = [3.5, 4.0, 4.5]  # all safe under the 3-zone model
    p = [0.30, 0.10, 0.20]
    sp.apply_max_streak_constraint(skip, z, p)
    # Run length 3 == allowed 3, so nothing is trimmed.
    assert skip == [True, True, True]
    _assert_streaks_ok(skip, z)
    for zi in z:
        assert sp.zone_from_z(zi) == "safe"


def test_streak_all_true_no_longer_collapses():
    # Regression for the 2026-08-03 fix. Under the old run-level rule an
    # all-True proposal on the 30-step reference schedule formed ONE run,
    # whose minimum zone limit was danger's 1, so the whole schedule was
    # trimmed to 3 skips. The per-step rule applies each step's own limit.
    sched = _t_now_schedule(30)
    z = [sp.logsnr_proxy_from_t_now(t) for t in sched]
    skip = [True] * 30
    sp.apply_max_streak_constraint(skip, z, [1.0] * 30)
    _assert_streaks_ok(skip, z)

    # The expected count is derived independently here, not hard-coded: a
    # greedy left-to-right walk keeps a step iff the run so far is shorter
    # than that step's limit -- which is exactly the maximum number of skips
    # placeable subject to the per-step constraint when every step is
    # proposed (each forced-full step is the shortest possible break).
    expected = 0
    counter = 0
    for zi in z:
        allowed = sp.ZONE_MAX_STREAK[sp.zone_from_z(zi)]
        if counter >= allowed:
            counter = 0
        else:
            counter += 1
            expected += 1
    assert sum(skip) == expected
    # For this schedule (4 danger + 11 middle + 15 safe steps) that is 21,
    # versus 3 under the old implementation.
    assert sum(skip) == 21
    assert expected > 3


def test_streak_mean_skip_count_non_decreasing_in_flat_p():
    # Regression for the 2026-08-03 collapse: with a flat skip probability
    # swept 0.1 -> 1.0, the post-trim mean skip count (over many seeds) must
    # be non-decreasing in p. The old rule inverted at high p (runs merged
    # into one, so the danger limit governed everything) and the mean fell.
    import random as _random

    sched = _t_now_schedule(30)
    z = [sp.logsnr_proxy_from_t_now(t) for t in sched]
    means = []
    for i in range(1, 11):
        p = i / 10.0
        counts = []
        for seed in range(40):
            rng = _random.Random(seed)
            skip = [rng.random() < p for _ in range(30)]
            sp.apply_max_streak_constraint(skip, z, [p] * 30)
            _assert_streaks_ok(skip, z)
            counts.append(sum(skip))
        means.append(sum(counts) / len(counts))
    for a, b in zip(means, means[1:]):
        assert b >= a, "mean skip count fell as p rose: %r" % (means,)
    # The sweep must actually span a wide range (not a flat, trivially
    # non-decreasing line) and end at the all-True ceiling.
    assert means[0] < 5.0
    assert means[-1] == 21.0


def test_streak_zone_crossing_run_splits_per_zone():
    # A single proposed run crossing danger -> middle -> safe. Each zone's
    # own limit must hold within it: danger steps may never be the 2nd skip
    # of a run, middle steps never the 3rd, safe steps never the 4th.
    z = (
        [-6.0, -5.5, -5.0, -4.5]       # danger (limit 1)
        + [-3.5, -3.0, -2.5, -2.0, -1.0]  # middle (limit 2)
        + [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # safe (limit 3)
    )
    skip = [True] * len(z)
    sp.apply_max_streak_constraint(skip, z, [1.0] * len(z))
    _assert_streaks_ok(skip, z)
    # Danger steps alternate (limit 1): indices 0 and 2 survive, 1 and 3 are
    # flipped. The counter then enters the middle zone at 0.
    assert skip[:4] == [True, False, True, False]
    # No run anywhere exceeds the global maximum limit (safe's 3).
    assert _max_run(skip) <= 3
    # And no run containing a danger step is longer than 1.
    for k, s in enumerate(skip):
        if s and sp.zone_from_z(z[k]) == "danger":
            assert k == 0 or not skip[k - 1]


def test_streak_determinism_on_copies():
    skip = [True, True, True, True, True, True]
    z = [-1.0, -2.0, -3.0, -1.5, -0.5, -2.5]
    p = [0.4, 0.1, 0.2, 0.3, 0.5, 0.15]
    a1, z1, p1 = list(skip), list(z), list(p)
    a2, z2, p2 = list(skip), list(z), list(p)
    sp.apply_max_streak_constraint(a1, z1, p1)
    sp.apply_max_streak_constraint(a2, z2, p2)
    assert a1 == a2


def _max_run(skip):
    best = cur = 0
    for s in skip:
        cur = cur + 1 if s else 0
        best = max(best, cur)
    return best


def _assert_streaks_ok(
    skip, z, boundaries=sp.DEFAULT_ZONE_BOUNDARIES, zone_max_streak=None
):
    """Per-step streak invariant (2026-08-03 spec).

    The old helper asserted the *run-level* rule (a run may not exceed the
    minimum ZONE_MAX_STREAK over the zones it touches). Under the new
    per-step rule a run legitimately spans zones and may be longer than that
    minimum, so the invariant checked here is the one the implementation
    actually enforces: at every skipped step, the number of consecutive
    skips ending at that step must not exceed that step's own zone limit.
    """
    counter = 0
    for k, s in enumerate(skip):
        if not s:
            counter = 0
            continue
        counter += 1
        limits = sp.ZONE_MAX_STREAK if zone_max_streak is None else zone_max_streak
        allowed = limits[sp.zone_from_z(z[k], boundaries)]
        assert counter <= allowed, (
            "step %d (zone %s, allowed %d) ends a run of %d"
            % (k, sp.zone_from_z(z[k], boundaries), allowed, counter)
        )


# --- generated pattern respects streaks -------------------------------------


def test_generated_pattern_respects_streaks():
    sched = _t_now_schedule(30)
    for seed in range(20):
        pat = sp.generate_skip_pattern(
            sched, aggressiveness=1.0, skip_seed=seed, probability_model="sigmoid_band_v0.1",
        )
        _assert_streaks_ok(pat.skip, pat.z_by_step)


def test_generated_pattern_respects_custom_boundaries():
    sched = _t_now_schedule(30)
    boundaries = (-2.0, 2.0)
    for seed in range(20):
        pat = sp.generate_skip_pattern(
            sched, aggressiveness=1.0, skip_seed=seed, zone_boundaries=boundaries,
            probability_model="sigmoid_band_v0.1",
        )
        assert pat.zone_boundaries == boundaries
        _assert_streaks_ok(pat.skip, pat.z_by_step, boundaries)


# --- reproducibility --------------------------------------------------------


def test_reproducible_same_inputs():
    sched = _t_now_schedule(30)
    a = sp.generate_skip_pattern(
        sched, aggressiveness=0.5, skip_seed=999, probability_model="sigmoid_band_v0.1",
    )
    b = sp.generate_skip_pattern(
        sched, aggressiveness=0.5, skip_seed=999, probability_model="sigmoid_band_v0.1",
    )
    assert a.skip == b.skip
    assert a.skipped_steps == b.skipped_steps
    assert a.skip_count == b.skip_count


def test_different_seed_generally_differs():
    sched = _t_now_schedule(30)
    patterns = set()
    for seed in range(8):
        pat = sp.generate_skip_pattern(
            sched, aggressiveness=0.6, skip_seed=seed, probability_model="sigmoid_band_v0.1",
        )
        patterns.add(tuple(pat.skip))
    # Not all seeds collapse to the same pattern.
    assert len(patterns) > 1


def test_skipped_steps_are_one_based():
    sched = _t_now_schedule(30)
    pat = sp.generate_skip_pattern(
        sched, aggressiveness=1.0, skip_seed=3, probability_model="sigmoid_band_v0.1",
    )
    for step_no in pat.skipped_steps:
        assert 1 <= step_no <= 30
        assert pat.skip[step_no - 1] is True
    assert pat.skip_count == len(pat.skipped_steps)


# --- exact target -----------------------------------------------------------


def test_exact_target_reachable():
    sched = _t_now_schedule(30)
    # Find a target that is actually reachable at high aggressiveness.
    # Determine an achievable count first, then request it exactly.
    counts = set()
    for seed in range(30):
        pat = sp.generate_skip_pattern(
            sched, aggressiveness=1.0, skip_seed=seed, probability_model="sigmoid_band_v0.1",
        )
        counts.add(pat.skip_count)
    target = sorted(counts)[len(counts) // 2]  # a mid, reachable count
    pat = sp.generate_skip_pattern(
        sched, aggressiveness=1.0, skip_seed=100, exact_target=target,
        max_resample=200,
        probability_model="sigmoid_band_v0.1",
    )
    assert pat.skip_count == target


def test_exact_target_impossible_no_raise():
    sched = _t_now_schedule(30)
    pat = sp.generate_skip_pattern(
        sched, aggressiveness=0.5, skip_seed=1, exact_target=10 ** 6,
        probability_model="sigmoid_band_v0.1",
    )
    # Returns the closest attainable, never raises. With 30 steps and
    # the window, cannot possibly reach a million.
    assert pat.skip_count < 30
    assert isinstance(pat, sp.SkipPattern)


def test_exact_target_deterministic():
    sched = _t_now_schedule(30)
    a = sp.generate_skip_pattern(
        sched, aggressiveness=0.7, skip_seed=42, exact_target=12,
        probability_model="sigmoid_band_v0.1",
    )
    b = sp.generate_skip_pattern(
        sched, aggressiveness=0.7, skip_seed=42, exact_target=12,
        probability_model="sigmoid_band_v0.1",
    )
    assert a.skip == b.skip


# --- expected skips ---------------------------------------------------------


def test_expected_skips_equals_sum_p_eligible():
    sched = _t_now_schedule(30)
    pat = sp.generate_skip_pattern(
        sched, aggressiveness=0.5, skip_seed=5, probability_model="sigmoid_band_v0.1",
    )
    window_start, window_end = pat.skip_window
    n = pat.num_steps
    manual = sum(
        pat.p_by_step[idx]
        for idx in range(n)
        if window_start <= sp.progress_for_step(idx, n) <= window_end
    )
    assert pat.expected_skips_before_streak == pytest.approx(manual)
    # Excluded steps contribute zero, so sum over all p equals the same.
    assert pat.expected_skips_before_streak == pytest.approx(sum(pat.p_by_step))


def test_expected_skips_monotone_in_aggressiveness():
    sched = _t_now_schedule(30)
    prev = None
    for a in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        pat = sp.generate_skip_pattern(
            sched, aggressiveness=a, skip_seed=0, probability_model="sigmoid_band_v0.1",
        )
        val = pat.expected_skips_before_streak
        if prev is not None:
            # Deterministic and strictly non-decreasing in a for this
            # descending schedule.
            assert val > prev
        prev = val


# --- configurable zone_max_streak (2026-09-01) ------------------------------


def test_zone_max_streak_none_matches_explicit_constant():
    sched = _t_now_schedule(30)
    for seed in range(8):
        implicit = sp.generate_skip_pattern(
            sched, aggressiveness=0.8, skip_seed=seed,
        )
        explicit = sp.generate_skip_pattern(
            sched, aggressiveness=0.8, skip_seed=seed,
            zone_max_streak=sp.ZONE_MAX_STREAK,
        )
        assert implicit.skip == explicit.skip
        assert implicit.skip_count == explicit.skip_count


def _mean_skip_count(sched, zone_max_streak, seeds=256, a=0.8):
    total = 0
    for seed in range(seeds):
        pat = sp.generate_skip_pattern(
            sched, aggressiveness=a, skip_seed=seed,
            zone_max_streak=zone_max_streak,
        )
        total += pat.skip_count
    return total / seeds


def test_relaxed_zone_max_streak_increases_mean_skip_count():
    sched = _t_now_schedule(30)
    baseline = _mean_skip_count(sched, None)
    relaxed = _mean_skip_count(sched, {"danger": 3, "middle": 5, "safe": 8})
    assert relaxed > baseline


def test_zero_limit_forces_that_zone_full():
    sched = _t_now_schedule(30)
    limits = {"danger": 0, "middle": 2, "safe": 3}
    for seed in range(16):
        pat = sp.generate_skip_pattern(
            sched, aggressiveness=1.0, skip_seed=seed, zone_max_streak=limits,
        )
        danger_steps = [
            idx for idx in range(pat.num_steps)
            if sp.zone_from_z(pat.z_by_step[idx], pat.zone_boundaries) == "danger"
        ]
        assert danger_steps, "schedule must contain danger-zone steps"
        for idx in danger_steps:
            assert pat.skip[idx] is False
        _assert_streaks_ok(
            pat.skip, pat.z_by_step, pat.zone_boundaries, zone_max_streak=limits
        )


def test_all_zero_limits_disable_skipping_entirely():
    sched = _t_now_schedule(30)
    limits = {"danger": 0, "middle": 0, "safe": 0}
    for seed in range(8):
        pat = sp.generate_skip_pattern(
            sched, aggressiveness=1.0, skip_seed=seed, zone_max_streak=limits,
        )
        assert pat.skip_count == 0
        assert pat.skipped_steps == []


def test_apply_max_streak_constraint_accepts_custom_limits():
    # Six consecutive safe-zone steps (z well above the high boundary), all
    # drawn as skips: a limit of 4 must trim the 5th, then restart the run.
    z = [1.0] * 6
    p = [1.0] * 6
    skip = [True] * 6
    sp.apply_max_streak_constraint(
        skip, z, p, sp.DEFAULT_ZONE_BOUNDARIES, {"danger": 1, "middle": 2, "safe": 4}
    )
    assert skip == [True, True, True, True, False, True]

    # The same input under the module default (safe -> 3).
    skip2 = [True] * 6
    sp.apply_max_streak_constraint(skip2, z, p)
    assert skip2 == [True, True, True, False, True, True]


def test_default_probability_model_is_monotone_saturate():
    sched = _t_now_schedule(30)
    pat = sp.generate_skip_pattern(sched, aggressiveness=0.5, skip_seed=1)
    assert pat.probability_model == "monotone_saturate_v0.1"
