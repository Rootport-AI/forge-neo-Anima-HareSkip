from __future__ import annotations

from .state import STATE


def start_sampling(source: str = "unknown") -> None:
    STATE.reset_generation(source)


def step_start() -> None:
    STATE.mark_step_start()


def step_end() -> None:
    STATE.mark_step_end()


def timing_summary() -> dict[str, float | int | None]:
    return {
        "denoiser_calls": STATE.denoiser_calls,
        "total_sampling_time": STATE.total_sampling_time(),
        "avg_step_time": STATE.avg_step_time(),
        "min_step_time": min(STATE.step_durations) if STATE.step_durations else None,
        "max_step_time": max(STATE.step_durations) if STATE.step_durations else None,
    }
