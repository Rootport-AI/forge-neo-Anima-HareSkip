from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelDetection:
    supported: bool
    confidence: str
    family: str
    evidence: dict[str, str] = field(default_factory=dict)
    reason: str = ""

    @property
    def key(self) -> str:
        return "|".join(
            [
                self.evidence.get("filename", ""),
                self.evidence.get("checkpoint_name", ""),
                self.evidence.get("diffusion_model_class", ""),
            ]
        )


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:
        return f"<unprintable {type(value).__name__}>"


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _nested_attr(obj: Any, path: str) -> Any:
    current = obj
    for part in path.split("."):
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(part)
        else:
            current = _safe_getattr(current, part)
    return current


def detect_model(sd_model: Any) -> ModelDetection:
    if sd_model is None:
        return ModelDetection(False, "none", "unknown", reason="sd_model is None")

    model_config = _safe_getattr(sd_model, "model_config")
    checkpoint_info = _safe_getattr(sd_model, "sd_checkpoint_info")
    forge_objects = _safe_getattr(sd_model, "forge_objects")
    diffusion_model = _nested_attr(forge_objects, "unet.model.diffusion_model")

    evidence = {
        "sd_model_class": type(sd_model).__name__,
        "model_config_class": type(model_config).__name__ if model_config else "",
        "huggingface_repo": _safe_str(
            _safe_getattr(model_config, "huggingface_repo", "")
        ),
        "filename": _safe_str(_safe_getattr(sd_model, "filename", "")),
        "checkpoint_name": _safe_str(_safe_getattr(checkpoint_info, "name", "")),
        "diffusion_model_class": type(diffusion_model).__name__
        if diffusion_model is not None
        else "",
        "diffusion_model_module": _safe_str(
            _safe_getattr(type(diffusion_model), "__module__", "")
        )
        if diffusion_model is not None
        else "",
    }

    haystack = " ".join(evidence.values()).lower()
    strong_anima = evidence["diffusion_model_class"] == "Anima"
    strong_module = "backend.nn.anima" in evidence["diffusion_model_module"].lower()
    has_anima = "anima" in haystack
    has_cosmos = "cosmos" in haystack or "predict2" in haystack

    if strong_anima or strong_module:
        family = "anima" if has_anima else "cosmos_predict2"
        return ModelDetection(
            True,
            "strong",
            family,
            evidence,
            "diffusion model is backend.nn.anima.Anima",
        )

    if has_anima or has_cosmos:
        family = "anima" if has_anima else "cosmos_predict2"
        return ModelDetection(
            True,
            "weak",
            family,
            evidence,
            "model name or metadata contains Anima/Cosmos/Predict2",
        )

    return ModelDetection(
        False,
        "none",
        "unknown",
        evidence,
        "no Anima/Cosmos/Predict2 evidence found",
    )
