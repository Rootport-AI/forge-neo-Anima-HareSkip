from __future__ import annotations

from typing import Any


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:
        return f"<unprintable {type(value).__name__}>"


def _shape(value: Any) -> str:
    shape = _safe_getattr(value, "shape")
    if shape is None:
        return ""
    try:
        return "x".join(str(part) for part in shape)
    except Exception:
        return _safe_str(shape)


def processing_info(p: Any) -> dict[str, Any]:
    width = _safe_getattr(p, "width")
    height = _safe_getattr(p, "height")
    return {
        "sampler": _safe_str(_safe_getattr(p, "sampler_name", "")),
        "scheduler": _safe_str(_safe_getattr(p, "scheduler", "")),
        "steps": _safe_getattr(p, "steps"),
        "cfg_scale": _safe_getattr(p, "cfg_scale"),
        "width": width,
        "height": height,
        "is_img2img": _safe_getattr(p, "is_img2img", False),
        "enable_hr": _safe_getattr(p, "enable_hr", False),
    }


def attention_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "attention_backend": "unknown",
        "sage_available": False,
        "flash_available": False,
        "xformers_available": False,
        "pytorch_available": False,
        "anima_attention_path": False,
    }
    try:
        from backend import attention

        fn = _safe_getattr(attention, "attention_function")
        info["attention_backend"] = _safe_getattr(fn, "__name__", _safe_str(fn))
        info["sage_available"] = callable(_safe_getattr(attention, "sageattn"))
        info["flash_available"] = callable(_safe_getattr(attention, "flash_attn_wrapper"))
        xformers_module = _safe_getattr(attention, "xformers")
        xformers_ops = _safe_getattr(xformers_module, "ops")
        info["xformers_available"] = callable(
            _safe_getattr(xformers_ops, "memory_efficient_attention")
        )
        info["pytorch_available"] = callable(_safe_getattr(attention, "attention_pytorch"))
    except Exception as exc:
        info["attention_error"] = _safe_str(exc)

    try:
        from backend.nn import anima

        info["anima_attention_path"] = all(
            hasattr(anima, attr) for attr in ("SelfCrossAttention", "Block")
        )
    except Exception as exc:
        info["anima_error"] = _safe_str(exc)

    return info


def lowbit_info(sd_model: Any) -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        from backend.args import dynamic_args

        ops = _safe_getattr(dynamic_args, "ops")
        info["forge_ops"] = _safe_str(ops)
        info["forge_ops_type"] = type(ops).__name__ if ops is not None else ""
    except Exception as exc:
        info["forge_ops_error"] = _safe_str(exc)

    for name in ("storage_dtype", "computation_dtype", "dtype"):
        value = _safe_getattr(sd_model, name)
        if value is not None:
            info[f"sd_model_{name}"] = _safe_str(value)

    try:
        forge_objects = _safe_getattr(sd_model, "forge_objects")
        unet = None
        if isinstance(forge_objects, dict):
            unet = forge_objects.get("unet")
        else:
            unet = _safe_getattr(forge_objects, "unet")
        model = _safe_getattr(unet, "model")
        diffusion_model = _safe_getattr(model, "diffusion_model")
        for name in ("storage_dtype", "computation_dtype", "dtype"):
            value = _safe_getattr(diffusion_model, name)
            if value is not None:
                info[f"diffusion_model_{name}"] = _safe_str(value)
    except Exception as exc:
        info["dtype_probe_error"] = _safe_str(exc)

    return info


def model_structure_info(sd_model: Any) -> dict[str, Any]:
    info: dict[str, Any] = {}
    diffusion_model = _diffusion_model(sd_model)
    if diffusion_model is None:
        return {"structure_error": "diffusion_model unavailable"}

    blocks = _safe_getattr(diffusion_model, "blocks")
    info["diffusion_model_class"] = type(diffusion_model).__name__
    info["num_blocks"] = len(blocks) if blocks is not None else None
    for name in ("patch_spatial", "patch_temporal", "in_channels", "out_channels"):
        value = _safe_getattr(diffusion_model, name)
        if value is not None:
            info[name] = value

    if blocks:
        first_block = blocks[0]
        self_attn = _safe_getattr(first_block, "self_attn")
        cross_attn = _safe_getattr(first_block, "cross_attn")
        info["block_class"] = type(first_block).__name__
        info["self_heads"] = _safe_getattr(self_attn, "n_heads")
        info["self_head_dim"] = _safe_getattr(self_attn, "head_dim")
        info["cross_heads"] = _safe_getattr(cross_attn, "n_heads")
        info["cross_head_dim"] = _safe_getattr(cross_attn, "head_dim")
    return info


def cond_info(params: Any) -> dict[str, Any]:
    denoiser = _safe_getattr(params, "denoiser")
    p = _safe_getattr(denoiser, "p")
    transformer_options = _safe_getattr(params, "transformer_options", {})
    if transformer_options is None:
        transformer_options = {}

    def opt(name: str) -> Any:
        if isinstance(transformer_options, dict):
            return transformer_options.get(name)
        return _safe_getattr(transformer_options, name)

    cfg_scale = _safe_getattr(params, "cond_scale")
    if cfg_scale is None:
        cfg_scale = _safe_getattr(params, "cfg_scale")
    if cfg_scale is None:
        cfg_scale = _safe_getattr(p, "cfg_scale")

    return {
        "text_uncond_is_none": _safe_getattr(params, "text_uncond") is None,
        "text_cond_type": type(_safe_getattr(params, "text_cond")).__name__,
        "text_uncond_type": type(_safe_getattr(params, "text_uncond")).__name__,
        "x_shape": _shape(_safe_getattr(params, "x")),
        "sigma_shape": _shape(_safe_getattr(params, "sigma")),
        "cfg_scale": cfg_scale,
        "sampling_step": _safe_getattr(params, "sampling_step"),
        "total_sampling_steps": _safe_getattr(params, "total_sampling_steps"),
        "denoiser_step": _safe_getattr(denoiser, "step"),
        "denoiser_total_steps": _safe_getattr(denoiser, "total_steps"),
        "cond_or_uncond": opt("cond_or_uncond"),
        "cond_indices": opt("cond_indices"),
        "uncond_indices": opt("uncond_indices"),
        "transformer_options_stage": "not_created_at_cfg_denoiser_callback",
    }


def _diffusion_model(sd_model: Any) -> Any:
    try:
        forge_objects = _safe_getattr(sd_model, "forge_objects")
        unet = None
        if isinstance(forge_objects, dict):
            unet = forge_objects.get("unet")
        else:
            unet = _safe_getattr(forge_objects, "unet")
        model = _safe_getattr(unet, "model")
        return _safe_getattr(model, "diffusion_model")
    except Exception:
        return None


def model_sampling_info(sd_model: Any) -> dict[str, Any]:
    """Best-effort snapshot of the model sampling object (shift, sigma range).

    The Shift label is informational; the per-pair `t_now` column in
    calibration_pairs.jsonl is the ground truth for the timestep grid.
    """
    try:
        forge_objects = _safe_getattr(sd_model, "forge_objects")
        if isinstance(forge_objects, dict):
            unet = forge_objects.get("unet")
        else:
            unet = _safe_getattr(forge_objects, "unet")
        model = _safe_getattr(unet, "model")
        model_sampling = _safe_getattr(model, "model_sampling")
        if model_sampling is None:
            model_sampling = _safe_getattr(model, "predictor")
        if model_sampling is None:
            return {"available": False}
        return {
            "available": True,
            "class": type(model_sampling).__name__,
            "shift": _safe_number(_safe_getattr(model_sampling, "shift")),
            "multiplier": _safe_number(_safe_getattr(model_sampling, "multiplier")),
            "sigma_min": _safe_number(_safe_getattr(model_sampling, "sigma_min")),
            "sigma_max": _safe_number(_safe_getattr(model_sampling, "sigma_max")),
        }
    except Exception:
        return {"available": False}


def _safe_number(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().flatten()[0].item()
        return float(value)
    except Exception:
        return None
