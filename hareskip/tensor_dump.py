from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .logging import info, warning
from .model_detect import ModelDetection
from .state import STATE

_stats: list[dict[str, Any]] = []


def any_tensor_dump_enabled() -> bool:
    return STATE.tensor_dump_active()


def initialize_run_if_needed(p: Any = None) -> None:
    global _stats

    if not any_tensor_dump_enabled() or STATE.tensor_dump_initialized:
        return
    run_dir = ensure_run_dir(p)
    if run_dir is None:
        _mark_unavailable("run_dir_unavailable")
        return
    try:
        (run_dir / "tensors").mkdir(parents=True, exist_ok=True)
        _stats = []
        STATE.tensor_dump_run_dir = str(run_dir)
        STATE.tensor_dump_initialized = True
        _write_meta(run_dir, p)
        info(f"tensor_dump_initialized run_dir={run_dir}")
    except Exception as exc:
        STATE.tensor_dump_errors += 1
        _mark_unavailable(f"initialize_failed:{_short_error(exc)}")


def ensure_run_dir(p: Any = None) -> Path | None:
    try:
        now = datetime.now().astimezone()
        base_dir = _infer_log_base_dir(p, now)
        run_id = f"run_{now:%Y%m%d_%H%M%S}_gen{STATE.generation_index:04d}"
        return _create_unique_run_dir(base_dir, run_id)
    except Exception as exc:
        STATE.tensor_dump_errors += 1
        warning(f"tensor_dump_run_dir_failed reason={_short_error(exc)}")
        return None


def dump_tensor(
    tensor_type: str,
    tensor: Any,
    *,
    logical_step_index: int | None = None,
    local_call_index: int | None = None,
    call_index: int | None = None,
    slot: int | None = None,
    decision: str | None = None,
    timestep_value: Any = None,
    extra: dict[str, Any] | None = None,
    **_: Any,
) -> None:
    if not any_tensor_dump_enabled():
        return
    try:
        import torch

        if not torch.is_tensor(tensor):
            return
        initialize_run_if_needed()
        if not STATE.tensor_dump_initialized or not STATE.tensor_dump_run_dir:
            return

        run_dir = Path(STATE.tensor_dump_run_dir)
        tensor_cpu = tensor.detach().cpu()
        safe_slot = "none" if slot is None else str(int(slot))
        safe_call = "none" if local_call_index is None else str(int(local_call_index))
        file_name = f"{tensor_type}_slot{safe_slot}_call{safe_call}.pt"
        tensor_path = run_dir / "tensors" / file_name
        torch.save(tensor_cpu, tensor_path)

        stats = _tensor_stats(tensor_cpu)
        record = {
            "schema_version": 1,
            "run_id": run_dir.name,
            "generation_index": STATE.generation_index,
            "tensor_type": tensor_type,
            "logical_step_index": logical_step_index,
            "local_call_index": local_call_index,
            "call_index": call_index,
            "slot": slot,
            "decision": decision,
            "timestep_value": _safe_float(timestep_value),
            "shape": "x".join(str(part) for part in tensor_cpu.shape),
            "dtype": str(tensor_cpu.dtype).replace("torch.", ""),
            "numel": int(tensor_cpu.numel()),
            **stats,
            "tensor_path": str(tensor_path),
            "extra": extra or {},
        }
        _stats.append(record)
        STATE.tensor_dump_records += 1
    except Exception as exc:
        STATE.tensor_dump_errors += 1
        warning(f"tensor_dump_failed type={tensor_type} reason={_short_error(exc)}")


def flush_stats() -> None:
    if not STATE.tensor_dump_initialized or not STATE.tensor_dump_run_dir or not _stats:
        return
    try:
        run_dir = Path(STATE.tensor_dump_run_dir)
        stats_path = run_dir / "stats.jsonl"
        with stats_path.open("w", encoding="utf-8") as handle:
            for record in _stats:
                handle.write(json.dumps(record, default=str, sort_keys=True))
                handle.write("\n")
        info(
            "tensor_dump_summary="
            f"records={STATE.tensor_dump_records} errors={STATE.tensor_dump_errors} "
            f"run_dir={run_dir}"
        )
    except Exception as exc:
        STATE.tensor_dump_errors += 1
        warning(f"tensor_dump_flush_failed reason={_short_error(exc)}")


def _create_unique_run_dir(base_dir: Path, run_id: str) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(1000):
        suffix = "" if attempt == 0 else f"_{attempt:03d}"
        run_dir = base_dir / f"{run_id}{suffix}"
        try:
            run_dir.mkdir(exist_ok=False)
            return run_dir
        except FileExistsError:
            continue
    raise RuntimeError(f"could not create unique run directory for {run_id}")


def _infer_log_base_dir(p: Any, now: datetime) -> Path:
    candidates: list[Any] = []
    for attr in ("outpath_samples", "outpath_grids", "outdir_samples"):
        value = getattr(p, attr, None) if p is not None else None
        if value:
            candidates.append(value)
    try:
        from modules import shared

        opts = shared.opts
        for key in ("outdir_txt2img_samples", "outdir_samples", "outdir_save"):
            value = getattr(opts, key, None)
            if value:
                candidates.append(value)
    except Exception:
        pass

    for value in candidates:
        try:
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = (Path.cwd() / path).resolve()
            return path / "hareskip_logs" / f"{now:%Y-%m-%d}"
        except Exception:
            continue
    return Path.cwd() / "outputs" / "hareskip_logs" / f"{now:%Y-%m-%d}"


def _write_meta(run_dir: Path, p: Any) -> None:
    now = datetime.now().astimezone()
    detection = STATE.model_detection
    evidence = detection.evidence if isinstance(detection, ModelDetection) else {}
    meta = {
        "schema_version": 1,
        "extension": "HareSkip",
        "model": evidence.get("checkpoint_name") or evidence.get("filename") or "unknown",
        "generation_index": STATE.generation_index,
        "created_at": now.isoformat(),
        "steps": _safe_int(getattr(p, "steps", None)),
        "width": _safe_int(getattr(p, "width", None)),
        "height": _safe_int(getattr(p, "height", None)),
        "batch_size": _safe_int(getattr(p, "batch_size", None)),
        "sampler": str(getattr(p, "sampler_name", "unknown")),
        "scheduler": str(getattr(p, "scheduler", "unknown")),
        "cfg_scale": _safe_float(getattr(p, "cfg_scale", None)),
        "dump_flags": {
            "resrefine_residual": STATE.dump_resrefine_residual,
            "capture_calibration_pairs": STATE.capture_calibration_pairs,
        },
    }
    (run_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _tensor_stats(tensor_cpu: Any) -> dict[str, float]:
    import torch

    if tensor_cpu.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "l2": 0.0}
    data = tensor_cpu.float()
    return {
        "mean": float(data.mean().item()),
        "std": float(data.std(unbiased=False).item()),
        "min": float(data.min().item()),
        "max": float(data.max().item()),
        "l2": float(torch.linalg.vector_norm(data).item()),
    }


def _mark_unavailable(reason: str) -> None:
    if not reason:
        reason = "unknown"
    STATE.tensor_dump_unavailable_reason = reason
    _warn_once(reason, f"tensor_dump_unavailable reason={reason}")


def _warn_once(key: str, message: str) -> None:
    if key in STATE.tensor_dump_warned_reasons:
        return
    STATE.tensor_dump_warned_reasons.add(key)
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

