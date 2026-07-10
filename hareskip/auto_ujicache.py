from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from typing import Any

from .state import (
    STATE,
    UJICACHE_COEFFICIENT_PROFILES,
    UJICACHE_FORMULA_LINEAR,
    UJICACHE_FORMULA_TEACACHE,
    UJICACHE_FORMULA_TAYLOR2,
    ujicache_modulated_source_for_profile,
    ujicache_window_for_profile,
)


class AutoUjiCsvError(ValueError):
    pass


@dataclass
class AutoUjiRow:
    index: int
    name: str
    threshold: float | None = None
    formula: str | None = None
    prediction_strength: float | None = None
    slope_ema_smoothing: float | None = None
    curve_ema_smoothing: float | None = None
    taylor2_curve_strength: float | None = None
    apply_prediction_from_skip: int | None = None
    use_prediction_after_progress: float | None = None
    max_skip_streak: int | None = None
    force_full_interval: int | None = None
    coefficient_profile: str | None = None
    start_percent: float | None = None
    end_percent: float | None = None


@dataclass
class AutoUjiParseResult:
    rows: list[AutoUjiRow]
    warnings: list[str] = field(default_factory=list)


_FORMULA_ALIASES = {
    "teacache": UJICACHE_FORMULA_TEACACHE,
    "teacache (residual only)": UJICACHE_FORMULA_TEACACHE,
    "residual": UJICACHE_FORMULA_TEACACHE,
    "residual_only": UJICACHE_FORMULA_TEACACHE,
    "linear": UJICACHE_FORMULA_LINEAR,
    "linear_extrapolation": UJICACHE_FORMULA_LINEAR,
    "linear extrapolation": UJICACHE_FORMULA_LINEAR,
    "taylor2": UJICACHE_FORMULA_TAYLOR2,
    "taylor2 curve": UJICACHE_FORMULA_TAYLOR2,
    "taylor": UJICACHE_FORMULA_TAYLOR2,
    "quadratic": UJICACHE_FORMULA_TAYLOR2,
}

_SUPPORTED_COLUMNS = {
    "name",
    "threshold",
    "formula",
    "prediction_strength",
    "slope_ema_smoothing",
    "curve_ema_smoothing",
    "taylor2_curve_strength",
    "apply_prediction_from_skip",
    "use_prediction_after_progress",
    "max_skip_streak",
    "force_full_interval",
    "coefficient_profile",
    "start_percent",
    "end_percent",
}


def parse_auto_ujicache_csv(text: str) -> AutoUjiParseResult:
    if not str(text or "").strip():
        raise AutoUjiCsvError("reason=empty_csv")

    reader = csv.reader(io.StringIO(str(text)))
    raw_rows = [
        (line_number, row)
        for line_number, row in enumerate(reader, start=1)
        if any(str(cell).strip() for cell in row)
    ]
    if not raw_rows:
        raise AutoUjiCsvError("reason=empty_csv")

    header_line, header_cells = raw_rows[0]
    header = [_normalize_column_name(cell) for cell in header_cells]
    while header and not header[-1]:
        header.pop()

    if not header:
        raise AutoUjiCsvError(f"line={header_line} reason=empty_header")

    warnings: list[str] = []
    for column in header:
        if column and column not in _SUPPORTED_COLUMNS:
            warnings.append(f"unknown_column={column} ignored=True")

    rows: list[AutoUjiRow] = []
    for line_number, cells in raw_rows[1:]:
        values = _row_values(header, cells)
        try:
            row = AutoUjiRow(
                index=len(rows) + 1,
                name=_parse_name(values.get("name"), len(rows) + 1),
                threshold=_parse_float(values.get("threshold"), line_number, "threshold"),
                formula=_parse_formula(values.get("formula"), line_number),
                prediction_strength=_parse_float(
                    values.get("prediction_strength"),
                    line_number,
                    "prediction_strength",
                ),
                slope_ema_smoothing=_parse_float(
                    values.get("slope_ema_smoothing"),
                    line_number,
                    "slope_ema_smoothing",
                ),
                curve_ema_smoothing=_parse_float(
                    values.get("curve_ema_smoothing"),
                    line_number,
                    "curve_ema_smoothing",
                ),
                taylor2_curve_strength=_parse_float(
                    values.get("taylor2_curve_strength"),
                    line_number,
                    "taylor2_curve_strength",
                ),
                apply_prediction_from_skip=_parse_int(
                    values.get("apply_prediction_from_skip"),
                    line_number,
                    "apply_prediction_from_skip",
                ),
                use_prediction_after_progress=_parse_float(
                    values.get("use_prediction_after_progress"),
                    line_number,
                    "use_prediction_after_progress",
                ),
                max_skip_streak=_parse_int(
                    values.get("max_skip_streak"),
                    line_number,
                    "max_skip_streak",
                ),
                force_full_interval=_parse_int(
                    values.get("force_full_interval"),
                    line_number,
                    "force_full_interval",
                ),
                coefficient_profile=_parse_profile(
                    values.get("coefficient_profile"),
                    line_number,
                ),
                start_percent=_parse_float(
                    values.get("start_percent"),
                    line_number,
                    "start_percent",
                ),
                end_percent=_parse_float(
                    values.get("end_percent"),
                    line_number,
                    "end_percent",
                ),
            )
        except AutoUjiCsvError:
            raise
        except Exception as exc:
            raise AutoUjiCsvError(f"line={line_number} reason={exc}") from exc
        rows.append(row)

        if len(cells) > len(header):
            extras = [str(cell).strip() for cell in cells[len(header) :]]
            if any(extras):
                warnings.append(
                    f"line={line_number} extra_columns={len(extras)} ignored=True"
                )

    if not rows:
        raise AutoUjiCsvError("reason=no_data_rows")

    return AutoUjiParseResult(rows=rows, warnings=warnings)


def apply_auto_ujicache_row_to_state(row: AutoUjiRow) -> None:
    if row.threshold is not None:
        STATE.ujicache_threshold = _clamp_float(row.threshold, 0.0, 1.0)
    if row.formula is not None:
        STATE.ujicache_formula = row.formula
    if row.prediction_strength is not None:
        STATE.ujicache_prediction_strength = _clamp_float(
            row.prediction_strength,
            0.0,
            1.0,
        )
    if row.slope_ema_smoothing is not None:
        STATE.ujicache_slope_ema_smoothing = _clamp_float(
            row.slope_ema_smoothing,
            0.0,
            0.99,
        )
    if row.curve_ema_smoothing is not None:
        STATE.ujicache_curve_ema_smoothing = _clamp_float(
            row.curve_ema_smoothing,
            0.0,
            0.99,
        )
    if row.taylor2_curve_strength is not None:
        STATE.ujicache_taylor2_curve_strength = _clamp_float(
            row.taylor2_curve_strength,
            0.0,
            1.0,
        )
    if row.apply_prediction_from_skip is not None:
        STATE.ujicache_apply_prediction_from_skip = _clamp_int(
            row.apply_prediction_from_skip,
            1,
            3,
        )
    if row.use_prediction_after_progress is not None:
        STATE.ujicache_use_prediction_after_progress = _clamp_float(
            row.use_prediction_after_progress,
            0.0,
            1.0,
        )
    if row.max_skip_streak is not None:
        STATE.ujicache_max_skip_streak = _clamp_int(row.max_skip_streak, 0, 64)
    if row.force_full_interval is not None:
        STATE.ujicache_force_full_interval = _clamp_int(
            row.force_full_interval,
            0,
            64,
        )
    if row.coefficient_profile is not None:
        STATE.ujicache_coefficient_profile = row.coefficient_profile
        STATE.ujicache_modulated_source = ujicache_modulated_source_for_profile(
            row.coefficient_profile
        )
    # Window resolution (start/end independent): an explicit row value wins;
    # otherwise the selected profile's fit window applies; otherwise the current
    # STATE value is left untouched.
    profile_window = (
        ujicache_window_for_profile(row.coefficient_profile)
        if row.coefficient_profile is not None
        else None
    )
    if row.start_percent is not None:
        STATE.ujicache_start_percent = _clamp_float(row.start_percent, 0.0, 1.0)
    elif profile_window is not None:
        STATE.ujicache_start_percent = profile_window[0]
    if row.end_percent is not None:
        STATE.ujicache_end_percent = _clamp_float(row.end_percent, 0.0, 1.0)
    elif profile_window is not None:
        STATE.ujicache_end_percent = profile_window[1]


def _row_values(header: list[str], cells: list[str]) -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    for index, column in enumerate(header):
        if not column or column not in _SUPPORTED_COLUMNS:
            continue
        value = cells[index] if index < len(cells) else ""
        value = str(value).strip()
        values[column] = value if value else None
    return values


def _normalize_column_name(value: Any) -> str:
    return str(value or "").strip().lower()


def _parse_name(value: str | None, index: int) -> str:
    if value is None:
        return f"row_{index}"
    return str(value).strip() or f"row_{index}"


def _parse_float(value: str | None, line: int, column: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception as exc:
        raise AutoUjiCsvError(
            f"line={line} column={column} reason=invalid_float value={value}"
        ) from exc


def _parse_int(value: str | None, line: int, column: str) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception as exc:
        raise AutoUjiCsvError(
            f"line={line} column={column} reason=invalid_int value={value}"
        ) from exc


def _parse_formula(value: str | None, line: int) -> str | None:
    if value is None:
        return None
    formula = _FORMULA_ALIASES.get(str(value).strip().lower())
    if formula is None:
        raise AutoUjiCsvError(
            f"line={line} column=formula reason=unknown_formula value={value}"
        )
    return formula


def _parse_profile(value: str | None, line: int) -> str | None:
    if value is None:
        return None
    profile = str(value).strip()
    if profile not in UJICACHE_COEFFICIENT_PROFILES:
        raise AutoUjiCsvError(
            f"line={line} column=coefficient_profile reason=unknown_profile value={value}"
        )
    return profile


def _clamp_float(value: Any, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except Exception:
        number = minimum
    return max(minimum, min(maximum, number))


def _clamp_int(value: Any, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except Exception:
        number = minimum
    return max(minimum, min(maximum, number))
