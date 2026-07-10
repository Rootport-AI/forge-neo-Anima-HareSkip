from __future__ import annotations

from . import __version__
from .diagnostics import log_cond_trace
from .logging import exception, info
from .model_detect import detect_model
from .patcher import remove_all_patches
from .settings import on_ui_settings
from .state import STATE
from .timing import step_end, step_start

_registered = False


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
    except Exception as exc:
        STATE.set_error(f"cfg denoiser callback failed: {exc}")
        exception("cfg denoiser callback failed")


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
