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
def test_zone_boundaries(z, zone):
    assert sp.zone_from_z(z) == zone


@pytest.mark.parametrize(
    "n,expected",
    [(10, 1), (20, 1), (30, 2), (50, 2)],
)
def test_guard_count_for(n, expected):
    assert sp.guard_count_for(n) == expected


def test_zone_max_streak_table():
    assert sp.ZONE_MAX_STREAK == {
        "danger": 1,
        "middle": 2,
        "safe": 3,
    }


# --- guards -----------------------------------------------------------------


def test_guards_forced_full_30_steps():
    sched = _t_now_schedule(30)
    pat = sp.generate_skip_pattern(sched, aggressiveness=1.0, skip_seed=7)
    assert pat.guard_count == 2
    for idx in (0, 1, 28, 29):
        assert pat.skip[idx] is False
        assert pat.p_by_step[idx] == 0.0


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


def test_streak_middle_run_trimmed_argmin_p():
    # 4-long True run entirely in the middle zone (-4 <= z < 0).
    # allowed = 2, so trim to <= 2. Lowest p is index 2.
    skip = [True, True, True, True]
    z = [-1.0, -1.5, -2.0, -0.5]
    p = [0.30, 0.20, 0.10, 0.40]
    sp.apply_max_streak_constraint(skip, z, p)
    # Index 2 (lowest p) flipped first. After that run is [0,1] and [3],
    # both <= 2, so only one flip.
    assert skip == [True, True, False, True]
    # Verify no run exceeds allowed.
    _assert_streaks_ok(skip, z)


def test_streak_tie_broken_by_lower_z_then_index():
    # Equal p across a 4-run in middle zone: tie -> lower z, then lower idx.
    skip = [True, True, True, True]
    z = [-1.0, -3.0, -2.0, -0.5]  # lowest z is index 1
    p = [0.25, 0.25, 0.25, 0.25]
    sp.apply_max_streak_constraint(skip, z, p)
    # First flip removes index 1 (lowest z). Now runs: [0] and [2,3] (len 2)
    # -> both OK.
    assert skip == [True, False, True, True]


def test_streak_cross_zone_danger_middle_allowed_1():
    # Run spans danger (z<-4) + middle -> allowed = min(1, 2) = 1 while the
    # danger step is present. The danger step (lowest p) is flipped first;
    # the remaining pure-middle run is then re-evaluated with allowed 2.
    skip = [True, True, True]
    z = [-5.0, -3.0, -2.0]  # danger, middle, middle
    p = [0.10, 0.30, 0.20]
    sp.apply_max_streak_constraint(skip, z, p)
    # Danger step (index 0) removed; the middle pair is a valid length-2 run.
    assert skip == [False, True, True]
    _assert_streaks_ok(skip, z)


def test_streak_cross_zone_danger_forces_break():
    # A danger step embedded in a longer run must be broken: while any run
    # contains the danger step, allowed is 1, so it gets flipped out.
    skip = [True, True, True, True]
    z = [-3.0, -5.0, -3.5, -2.0]  # middle, danger, middle, middle
    # danger step has the lowest p so it is flipped, splitting the run.
    p = [0.40, 0.05, 0.30, 0.20]
    sp.apply_max_streak_constraint(skip, z, p)
    assert skip[1] is False  # danger step removed
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


def _assert_streaks_ok(skip, z):
    n = len(skip)
    i = 0
    while i < n:
        if not skip[i]:
            i += 1
            continue
        j = i
        while j < n and skip[j]:
            j += 1
        allowed = min(sp.ZONE_MAX_STREAK[sp.zone_from_z(z[k])] for k in range(i, j))
        assert (j - i) <= allowed
        i = j


# --- generated pattern respects streaks -------------------------------------


def test_generated_pattern_respects_streaks():
    sched = _t_now_schedule(30)
    for seed in range(20):
        pat = sp.generate_skip_pattern(sched, aggressiveness=1.0, skip_seed=seed)
        _assert_streaks_ok(pat.skip, pat.z_by_step)


# --- reproducibility --------------------------------------------------------


def test_reproducible_same_inputs():
    sched = _t_now_schedule(30)
    a = sp.generate_skip_pattern(sched, aggressiveness=0.5, skip_seed=999)
    b = sp.generate_skip_pattern(sched, aggressiveness=0.5, skip_seed=999)
    assert a.skip == b.skip
    assert a.skipped_steps == b.skipped_steps
    assert a.skip_count == b.skip_count


def test_different_seed_generally_differs():
    sched = _t_now_schedule(30)
    patterns = set()
    for seed in range(8):
        pat = sp.generate_skip_pattern(sched, aggressiveness=0.6, skip_seed=seed)
        patterns.add(tuple(pat.skip))
    # Not all seeds collapse to the same pattern.
    assert len(patterns) > 1


def test_skipped_steps_are_one_based():
    sched = _t_now_schedule(30)
    pat = sp.generate_skip_pattern(sched, aggressiveness=1.0, skip_seed=3)
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
        pat = sp.generate_skip_pattern(sched, aggressiveness=1.0, skip_seed=seed)
        counts.add(pat.skip_count)
    target = sorted(counts)[len(counts) // 2]  # a mid, reachable count
    pat = sp.generate_skip_pattern(
        sched, aggressiveness=1.0, skip_seed=100, exact_target=target,
        max_resample=200,
    )
    assert pat.skip_count == target


def test_exact_target_impossible_no_raise():
    sched = _t_now_schedule(30)
    pat = sp.generate_skip_pattern(
        sched, aggressiveness=0.5, skip_seed=1, exact_target=10 ** 6
    )
    # Returns the closest attainable, never raises. With 30 steps and
    # guards, cannot possibly reach a million.
    assert pat.skip_count < 30
    assert isinstance(pat, sp.SkipPattern)


def test_exact_target_deterministic():
    sched = _t_now_schedule(30)
    a = sp.generate_skip_pattern(
        sched, aggressiveness=0.7, skip_seed=42, exact_target=12
    )
    b = sp.generate_skip_pattern(
        sched, aggressiveness=0.7, skip_seed=42, exact_target=12
    )
    assert a.skip == b.skip


# --- expected skips ---------------------------------------------------------


def test_expected_skips_equals_sum_p_nonguard():
    sched = _t_now_schedule(30)
    pat = sp.generate_skip_pattern(sched, aggressiveness=0.5, skip_seed=5)
    guard = pat.guard_count
    n = pat.num_steps
    manual = sum(
        pat.p_by_step[idx]
        for idx in range(n)
        if not (idx + 1 <= guard or idx + 1 > n - guard)
    )
    assert pat.expected_skips_before_streak == pytest.approx(manual)
    # Guard steps contribute zero, so sum over all p equals the same.
    assert pat.expected_skips_before_streak == pytest.approx(sum(pat.p_by_step))


def test_expected_skips_monotone_in_aggressiveness():
    sched = _t_now_schedule(30)
    prev = None
    for a in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        pat = sp.generate_skip_pattern(sched, aggressiveness=a, skip_seed=0)
        val = pat.expected_skips_before_streak
        if prev is not None:
            # Deterministic and strictly non-decreasing in a for this
            # descending schedule.
            assert val > prev
        prev = val
