"""Shared pytest fixtures/stubs.

``hareskip/script.py`` imports ``gradio`` and ``modules.scripts`` (Forge/
gradio-only). The system test environment does not have gradio installed
(package installs are not permitted for these tests), so tests that need to
exercise ``hareskip.script`` directly (e.g. infotext-writing helpers) import
it lazily through ``import_hareskip_script()`` below, which installs minimal
stand-in modules into ``sys.modules`` first. Other test files that don't
touch ``hareskip.script`` are unaffected — nothing here runs at collection
time unless a test actually calls the helper.
"""

from __future__ import annotations

import sys
import types


def _install_gradio_stub() -> None:
    if "gradio" in sys.modules:
        return

    gr = types.ModuleType("gradio")

    class _Component:
        def __init__(self, *args, **kwargs):
            pass

        # hareskip/script.py's ui() nests most of its layout in
        # ``with gr.Accordion(...):`` / ``with gr.Group(...):`` blocks. The
        # real gradio components support the context-manager protocol; the
        # stub needs the same so ui() can actually run under test (used by
        # the ui() smoke test in test_recommended_settings.py).
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def change(self, *args, **kwargs):
            return None

        # gradio Sliders expose .release (fires once on mouse-up); the
        # estimate wiring uses it so dragging does not re-run the Monte
        # Carlo per pixel. Same no-op contract as .change for the stub.
        release = change

        # gradio Buttons expose .click instead of .change; same no-op
        # contract — the "Load recommended settings" button wiring in
        # hareskip/script.py just needs the call to not raise.
        click = change

    for name in (
        "Accordion",
        "Checkbox",
        "Radio",
        "Group",
        "Slider",
        "Number",
        "Markdown",
        "HTML",
        "Dropdown",
        "Textbox",
        "Button",
    ):
        setattr(gr, name, _Component)

    def _update(*args, **kwargs):
        # Real gradio's gr.update(**kwargs) returns the kwargs as a dict
        # (plus an internal "__type__" marker) rather than applying them
        # immediately. Mirroring that (instead of returning None) lets
        # gradio-independent helpers that call other update-returning
        # helpers (e.g. hareskip.script._resolve_recommendation calling
        # _hareskip_prediction_control_updates) unwrap the result and expose
        # raw values for assertions, without any test relying on a bare
        # None return.
        return dict(kwargs)

    gr.update = _update
    sys.modules["gradio"] = gr


def _install_modules_scripts_stub() -> None:
    if "modules.scripts" in sys.modules:
        return

    modules_pkg = sys.modules.get("modules") or types.ModuleType("modules")
    scripts_mod = types.ModuleType("modules.scripts")

    class Script:
        pass

    scripts_mod.Script = Script
    scripts_mod.AlwaysVisible = "always"
    modules_pkg.scripts = scripts_mod
    sys.modules["modules"] = modules_pkg
    sys.modules["modules.scripts"] = scripts_mod


def import_hareskip_script():
    """Import and return ``hareskip.script``, installing gradio/Forge stubs first.

    Safe to call repeatedly (subsequent calls just return the cached module).
    """
    _install_gradio_stub()
    _install_modules_scripts_stub()
    import hareskip.script as script  # noqa: PLC0415

    return script
