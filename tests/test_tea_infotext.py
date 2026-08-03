"""Unit tests for the TeaCache mode's realized-skip infotext key.

``Tea skipped_steps`` mirrors the existing ``Manual skipped_steps`` /
``Hare skipped_steps`` keys: it reports the REALIZED skips recorded by the
patcher's shared skip-execution path (``STATE.hareskip_skipped_steps``,
0-based), converted to 1-based, deduped, sorted, and space-joined — not the
threshold/window configuration (those are separate ``Tea ...`` keys written
earlier by ``_apply_infotext_metadata``).

``hareskip/script.py`` imports gradio + ``modules.scripts`` (Forge-only,
unavailable in this test environment); ``tests/conftest.py`` provides
``import_hareskip_script()`` which installs minimal stand-ins so the module
can be imported and its infotext helpers exercised directly, mirroring how
``tests/test_arg_sync.py`` avoids importing gradio by parsing script.py as
text — here we go one step further and actually import+run the functions.
"""

from __future__ import annotations

from conftest import import_hareskip_script

script = import_hareskip_script()

from hareskip.constants import MODE_HARESKIP, MODE_MANUAL, MODE_TEACACHE
from hareskip.state import STATE


class _FakeProcessing:
    """Minimal stand-in for Forge's ``StableDiffusionProcessing``."""

    def __init__(self):
        self.extra_generation_params = {}


def _reset_state():
    STATE.reset_generation("test")
    STATE.hareskip_enabled = True
    STATE.hareskip_mode = MODE_TEACACHE
    STATE.hareskip_skipped_steps = []
    STATE.auto_teacache_active = False
    STATE.auto_teacache_row_index = None
    STATE.auto_teacache_row_count = 0


# --- Tea skipped_steps: normal cases ----------------------------------------


def test_tea_skipped_steps_written_1_based_sorted():
    _reset_state()
    # Patcher records 0-based step indices in the order they were skipped;
    # here out of order to also exercise the sort.
    STATE.hareskip_skipped_steps = [9, 11, 14]
    p = _FakeProcessing()

    script._apply_hare_pattern_infotext(p)

    assert p.extra_generation_params["Tea skipped_steps"] == "10 12 15"


def test_tea_skipped_steps_empty_when_no_skips():
    _reset_state()
    STATE.hareskip_skipped_steps = []
    p = _FakeProcessing()

    script._apply_hare_pattern_infotext(p)

    assert p.extra_generation_params["Tea skipped_steps"] == ""


def test_tea_skipped_steps_dedupes():
    _reset_state()
    # Defensive: even if the patcher ever recorded a duplicate, output stays
    # deduped (mirrors _apply_manual_skip_infotext's use of a set).
    STATE.hareskip_skipped_steps = [4, 4, 7]
    p = _FakeProcessing()

    script._apply_hare_pattern_infotext(p)

    assert p.extra_generation_params["Tea skipped_steps"] == "5 8"


def test_tea_skipped_steps_not_written_when_disabled():
    _reset_state()
    STATE.hareskip_enabled = False
    STATE.hareskip_skipped_steps = [9]
    p = _FakeProcessing()

    script._apply_hare_pattern_infotext(p)

    assert "Tea skipped_steps" not in p.extra_generation_params


def test_tea_skipped_steps_only_in_teacache_mode():
    # Manual mode must still only write "Manual skipped_steps", not Tea's key.
    _reset_state()
    STATE.hareskip_mode = MODE_MANUAL
    STATE.hareskip_skipped_steps = [9]
    p = _FakeProcessing()

    script._apply_hare_pattern_infotext(p)

    assert "Tea skipped_steps" not in p.extra_generation_params
    assert p.extra_generation_params["Manual skipped_steps"] == "10"


def test_hare_mode_does_not_write_tea_key():
    # HareSkip mode writes "Hare skipped_steps" from the pattern object, not
    # from STATE.hareskip_skipped_steps directly, and must not emit a Tea key.
    _reset_state()
    STATE.hareskip_mode = MODE_HARESKIP
    STATE.hareskip_pattern = None
    p = _FakeProcessing()

    script._apply_hare_pattern_infotext(p)

    assert "Tea skipped_steps" not in p.extra_generation_params
    assert "Manual skipped_steps" not in p.extra_generation_params


# --- Batch (n_iter>1) pass isolation -----------------------------------------
#
# Manual version's contract (docstring on RuntimeState.reset_generation):
# hareskip_skipped_steps IS cleared per generation pass (unlike
# manual_skip_parsed, which deliberately survives). Tea mode reuses the same
# STATE field, so the same per-pass isolation applies: simulating two passes
# in the same batch must not let step records leak across the postprocess_image
# boundary.


def test_tea_skipped_steps_isolated_across_passes():
    _reset_state()
    STATE.hareskip_skipped_steps = [9, 11]
    p1 = _FakeProcessing()
    script._apply_hare_pattern_infotext(p1)
    assert p1.extra_generation_params["Tea skipped_steps"] == "10 12"

    # Next sampling pass: reset_generation runs again (per pass, per
    # process_before_every_sampling -> _begin_generation), clearing the skip
    # record before the patcher accumulates this pass's skips.
    STATE.reset_generation("test")
    STATE.hareskip_enabled = True
    STATE.hareskip_mode = MODE_TEACACHE
    STATE.hareskip_skipped_steps = [19]
    p2 = _FakeProcessing()
    script._apply_hare_pattern_infotext(p2)

    assert p2.extra_generation_params["Tea skipped_steps"] == "20"
    # First image's infotext dict is untouched by the second pass.
    assert p1.extra_generation_params["Tea skipped_steps"] == "10 12"
