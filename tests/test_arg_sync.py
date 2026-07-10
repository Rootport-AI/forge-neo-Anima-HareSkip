"""Static UI-argument synchronisation tests (no gradio / Forge required).

Guards the 3-point sync rule documented in ``hareskip/constants.py``:

  1. ``Script.ui()`` return list (order + count),
  2. ``RuntimeState.apply_options`` positional signature (order + count),
  3. ``constants.UI_ARG_ORDER`` / ``EXPECTED_UI_ARG_COUNT``.

These tests parse ``script.py`` as text so they never import gradio.
"""

from __future__ import annotations

import inspect
import os
import re

import hareskip.constants as constants
import hareskip.state as state


_SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "hareskip",
    "script.py",
)


def test_expected_count_matches_order_length():
    assert constants.EXPECTED_UI_ARG_COUNT == len(constants.UI_ARG_ORDER)


def test_no_duplicate_arg_names():
    order = constants.UI_ARG_ORDER
    assert len(order) == len(set(order)), "UI_ARG_ORDER has duplicates"


def test_apply_options_signature_matches_order():
    sig = inspect.signature(state.RuntimeState.apply_options)
    params = [name for name in sig.parameters if name != "self"]
    assert params == constants.UI_ARG_ORDER, (
        "apply_options positional params must match UI_ARG_ORDER exactly.\n"
        f"apply_options: {params}\n"
        f"UI_ARG_ORDER: {constants.UI_ARG_ORDER}"
    )


def _extract_ui_return_names() -> list[str]:
    """Parse the ``return [ ... ]`` list at the end of ``ui()`` from source.

    Robust to whitespace and trailing comments: finds the ``ui()`` method,
    takes the final top-level ``return [`` block, and collects bare identifier
    entries. Bracket-depth aware so nested brackets would not confuse it (there
    are none, but this keeps the parse deterministic).
    """
    with open(_SCRIPT_PATH, "r", encoding="utf-8") as handle:
        source = handle.read()

    ui_start = source.index("def ui(")
    # Stop at the next method definition so we only look inside ui().
    next_def = re.search(r"\n    def ", source[ui_start + 1 :])
    ui_end = ui_start + 1 + next_def.start() if next_def else len(source)
    ui_body = source[ui_start:ui_end]

    ret_idx = ui_body.rindex("return [")
    bracket_start = ui_body.index("[", ret_idx)
    depth = 0
    end = None
    for i in range(bracket_start, len(ui_body)):
        ch = ui_body[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i
                break
    assert end is not None, "could not find end of ui() return list"

    inner = ui_body[bracket_start + 1 : end]
    names: list[str] = []
    for raw in inner.split(","):
        token = raw.split("#", 1)[0].strip()
        if token:
            names.append(token)
    return names


def test_ui_return_list_matches_order():
    names = _extract_ui_return_names()
    assert all(re.fullmatch(r"[A-Za-z_]\w*", n) for n in names), (
        f"unexpected non-identifier tokens in ui() return list: {names}"
    )
    assert names == constants.UI_ARG_ORDER, (
        "ui() return list must match UI_ARG_ORDER exactly.\n"
        f"ui() return: {names}\n"
        f"UI_ARG_ORDER: {constants.UI_ARG_ORDER}"
    )
