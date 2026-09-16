from __future__ import annotations

from . import __version__
from .constants import MODE_HARESKIP
from .diagnostics import log_cond_trace
from .forge_introspection import sampling_schedule_t_now
from .logging import exception, info
from .model_detect import detect_model
from .patcher import remove_all_patches
from .settings import on_ui_settings
from .state import STATE
from .timing import step_end, step_start

_registered = False

# Core (non-HareSkip) Forge components captured for the "Load recommended
# settings" button, keyed by elem_id. Populated by on_after_component below
# as the main txt2img/img2img UI is built; the recommended-settings button
# reads from this registry at ui() time rather than owning its own capture
# callback, mirroring ControlNet's on_after_component + elem_id filter
# approach (controlnet.py:611,946 in the Forge codebase).
CORE_COMPONENTS: dict[str, object] = {}

_CORE_COMPONENT_ELEM_IDS = (
    "txt2img_sampling",
    "txt2img_scheduler",
    "txt2img_steps",
    "img2img_sampling",
    "img2img_scheduler",
    "img2img_steps",
)


def on_after_component(component, **kwargs) -> None:
    try:
        elem_id = getattr(component, "elem_id", None) or kwargs.get("elem_id")
        if elem_id in _CORE_COMPONENT_ELEM_IDS:
            CORE_COMPONENTS[elem_id] = component
    except Exception:
        exception("on_after_component capture failed")


def _reset_for_reload() -> None:
    """Undo register_callbacks()'s registration state ahead of a Reload UI.

    Reload UI clears every previously registered script_callbacks callback,
    but the ``_registered`` guard above (and CORE_COMPONENTS) survive because
    this module stays cached in sys.modules — so without this reset,
    register_callbacks() would see ``_registered`` already True post-reload
    and skip re-registering everything, silently going dark. This mirrors
    ControlNet's on_before_reload(reset) (controlnet.py:612).
    """
    global _registered
    _registered = False
    CORE_COMPONENTS.clear()


def register_callbacks() -> None:
    global _registered
    if _registered:
        return
    try:
        from modules import script_callbacks

        script_callbacks.on_ui_settings(on_ui_settings)
        script_callbacks.on_model_loaded(on_model_loaded)
        script_callbacks.on_cfg_denoiser(on_cfg_denoiser)
        script_callbacks.on_cfg_after_cfg(on_cfg_after_cfg)
        script_callbacks.on_script_unloaded(on_script_unloaded)
        script_callbacks.on_after_component(
            on_after_component, name="hareskip-core-capture"
        )
        script_callbacks.on_before_reload(_reset_for_reload)
        _registered = True
        info(f"callbacks registered version={__version__}")
    except Exception:
        exception("failed to register callbacks")


def on_model_loaded(sd_model) -> None:
    try:
        STATE.refresh_settings()
        STATE.model_detection = detect_model(sd_model)
        detection = STATE.model_detection
        info(
            "model_loaded "
            f"supported={detection.supported} confidence={detection.confidence} "
            f"family={detection.family}"
        )
    except Exception as exc:
        STATE.set_error(f"model detection failed: {exc}")
        exception("model detection failed")


def on_cfg_denoiser(params) -> None:
    try:
        if not STATE.active():
            return
        step_start()
        log_cond_trace(params)
        _capture_hareskip_schedule(params)
    except Exception as exc:
        STATE.set_error(f"cfg denoiser callback failed: {exc}")
        exception("cfg denoiser callback failed")


def _capture_hareskip_schedule(params) -> None:
    """Best-effort, once-per-generation capture of the sigma->t_now schedule.

    Only relevant in HareSkip mode. Isolated in its own try/except so a
    schedule-probing failure can never break the denoiser callback; the
    patcher degrades to full compute (with a one-time warning) when the
    schedule stays unavailable.
    """
    if STATE.hareskip_mode != MODE_HARESKIP:
        return
    if STATE.hareskip_schedule_t_now is not None:
        return
    try:
        t_now = sampling_schedule_t_now(params)
        if t_now:
            STATE.hareskip_schedule_t_now = t_now
    except Exception:
        # Never propagate — leave the schedule unavailable for a graceful
        # full-compute fallback in the patcher.
        pass


def on_cfg_after_cfg(params) -> None:
    try:
        if not STATE.active():
            return
        step_end()
    except Exception as exc:
        STATE.set_error(f"cfg after-cfg callback failed: {exc}")
        exception("cfg after-cfg callback failed")


def on_script_unloaded() -> None:
    try:
        remove_all_patches()
        STATE.status = "disabled"
        info("script unloaded")
    except Exception:
        exception("script unload cleanup failed")
