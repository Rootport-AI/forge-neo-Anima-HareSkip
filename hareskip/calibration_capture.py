from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from . import __version__
from .logging import info, warning
from .model_detect import ModelDetection
from .state import STATE

SCHEMA_VERSION = 1
FILE_NAME = "calibration_pairs.jsonl"


def capture_active() -> bool:
    return STATE.calibration_capture_active()


def initialize_for_generation(p: Any = None) -> None:
    """Write the run header line. Requires the tensor-dump run dir to exist."""
    if not capture_active() or STATE.calibration_capture_header_written:
        return
    run_dir = STATE.tensor_dump_run_dir
    if not run_dir:
        _warn_once(
            "run_dir_unavailable",
            "calibration_capture_unavailable reason=run_dir_unavailable",
        )
        return
    try:
        path = Path(run_dir) / FILE_NAME
        header = _build_header(p)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(header, default=str, sort_keys=True))
            handle.write("\n")
        STATE.calibration_capture_path = str(path)
        STATE.calibration_capture_header_written = True
        info(f"calibration_capture_initialized path={path}")
    except Exception as exc:
        STATE.calibration_capture_errors += 1
        _warn_once(
            "initialize_failed",
            f"calibration_capture_initialize_failed reason={_short_error(exc)}",
        )


def capture_pair(
    slot_key: int,
    step_index: int,
    rel_l1: float | None,
    out_rel: float | None,
    estimate: float | None,
    t_now: float | None,
    t_prev: float | None,
) -> None:
    """Append one (x, y) pair line. Lines without both x and y are dropped."""
    if not capture_active():
        return
    if rel_l1 is None or out_rel is None:
        return
    path = STATE.calibration_capture_path
    if not path:
        _warn_once(
            "path_unavailable",
            "calibration_capture_pair_dropped reason=path_unavailable",
        )
        return
    record = {
        "type": "pair",
        "schema_version": SCHEMA_VERSION,
        "generation_index": STATE.generation_index,
        "step_index": int(step_index),
        "cond_or_uncond": int(slot_key),
        "rel_l1": float(rel_l1),
        "out_rel": float(out_rel),
        "estimate": None if estimate is None else float(estimate),
        "t_now": None if t_now is None else float(t_now),
        "t_prev": None if t_prev is None else float(t_prev),
    }
    try:
        with Path(path).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")
        STATE.calibration_capture_records += 1
    except Exception as exc:
        STATE.calibration_capture_errors += 1
        _warn_once(
            "write_failed",
            f"calibration_capture_write_failed reason={_short_error(exc)}",
        )


def log_capture_summary() -> None:
    if not STATE.capture_calibration_pairs:
        return
    info(
        "calibration_capture_summary="
        f"records={STATE.calibration_capture_records} "
        f"errors={STATE.calibration_capture_errors} "
        f"path={STATE.calibration_capture_path}"
    )


def _build_header(p: Any) -> dict[str, Any]:
    now = datetime.now().astimezone()
    detection = STATE.model_detection
    evidence = detection.evidence if isinstance(detection, ModelDetection) else {}
    sampling = _model_sampling_snapshot()
    return {
        "type": "run",
        "schema_version": SCHEMA_VERSION,
        "extension": "HareSkip",
        "extension_version": __version__,
        "created_at": now.isoformat(),
        "generation_index": STATE.generation_index,
        "model": evidence.get("checkpoint_name") or evidence.get("filename") or "unknown",
        "steps": _safe_int(getattr(p, "steps", None)),
        "width": _safe_int(getattr(p, "width", None)),
        "height": _safe_int(getattr(p, "height", None)),
        "batch_size": _safe_int(getattr(p, "batch_size", None)),
        "sampler": str(getattr(p, "sampler_name", "unknown")),
        "scheduler": str(getattr(p, "scheduler", "unknown")),
        "cfg_scale": _safe_float(getattr(p, "cfg_scale", None)),
        "seed": _first_seed(p),
        "shift": sampling.get("shift"),
        "shift_ui_distilled_cfg": _safe_float(getattr(p, "distilled_cfg_scale", None)),
        "model_sampling": sampling,
        "ujicache": {
            "threshold": STATE.ujicache_threshold,
            "formula": STATE.ujicache_formula,
            "modulated_source": STATE.ujicache_modulated_source,
            "coefficient_profile": STATE.ujicache_coefficient_profile,
            "coefficients": _coefficients_snapshot(),
            "start_percent": STATE.ujicache_start_percent,
            "end_percent": STATE.ujicache_end_percent,
            "max_skip_streak": STATE.ujicache_max_skip_streak,
            "force_full_interval": STATE.ujicache_force_full_interval,
            "dry_run": STATE.ujicache_dry_run,
            "auto_row_index": STATE.auto_ujicache_row_index,
            "auto_row_name": STATE.auto_ujicache_row_name,
        },
        "forced_full": True,
        "rel_l1_definition": (
            "mean(|m_t - m_prev|) / mean(|m_prev|) on modulated source, per cond_or_uncond slot"
        ),
        "out_rel_definition": (
            "mean(|r_t - r_prev|) / mean(|r_prev|) on residual, float32, per cond_or_uncond slot"
        ),
    }


def _model_sampling_snapshot() -> dict[str, Any]:
    try:
        from modules import shared

        from .forge_introspection import model_sampling_info

        return model_sampling_info(getattr(shared, "sd_model", None))
    except Exception:
        return {"available": False}


def _coefficients_snapshot() -> list[float]:
    try:
        from .patcher import _ujicache_coefficients

        return [float(value) for value in _ujicache_coefficients()]
    except Exception:
        return []


def _first_seed(p: Any) -> int | None:
    try:
        seeds = getattr(p, "all_seeds", None)
        if seeds:
            return int(seeds[0])
    except Exception:
        pass
    return _safe_int(getattr(p, "seed", None))


def _warn_once(key: str, message: str) -> None:
    if key in STATE.calibration_capture_warned_reasons:
        return
    STATE.calibration_capture_warned_reasons.add(key)
    warning(message)


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if hasattr(value, "detach"):
            value = value.detach().flatten()[0].item()
        return float(value)
    except Exception:
        return None


def _short_error(value: Any) -> str:
    text = str(value).replace("\n", " ").strip()
    if len(text) > 220:
        return text[:217] + "..."
    return text
