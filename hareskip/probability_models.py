"""Swappable skip-probability models for HareSkip (pure stdlib math).

The HareSkip stochastic skip density maps a trajectory coordinate ``z``
(``logSNR`` proxy) to a per-step skip probability. The exact formula is
expected to change substantially in future research, so the probability
computation is isolated behind a small registry. Pattern generation,
zone/streak constraints and guards live elsewhere and never need to change
when a new formula is registered.

Protocol
========
A probability model is any plain object (module, class instance, or simple
namespace) exposing two callables::

    params_from_aggressiveness(a: float) -> dict
        Map the aggressiveness slider a in [0, 1] to a parameter dict.

    skip_probability(z: float, params: dict) -> float
        Map trajectory coordinate z and the params dict to a skip
        probability clamped to [0, 1].

To add a future formula, define an object with those two callables and call
``register("my_model_vX", MyModel())``. Nothing else in HareSkip needs to
change; ``generate_skip_pattern(..., probability_model="my_model_vX")`` will
pick it up via ``get_model``.
"""

import math

METHOD_NAME = "HareSkipStochasticDensity"
METHOD_VERSION = "0.1"


# --- numeric helpers --------------------------------------------------------


def _sigmoid(x):
    """Numerically stable logistic sigmoid.

    Guards ``math.exp`` against overflow for large-magnitude inputs by using
    the sign-stable formulation.
    """
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


# --- registry ---------------------------------------------------------------

PROBABILITY_MODELS = {}


def register(name, model):
    """Register a probability model under ``name``.

    ``model`` must expose ``params_from_aggressiveness`` and
    ``skip_probability`` callables (see module docstring).
    """
    PROBABILITY_MODELS[name] = model
    return model


def get_model(name):
    """Return the registered probability model for ``name``.

    Raises ``ValueError`` (listing the available names) if ``name`` is
    unknown.
    """
    try:
        return PROBABILITY_MODELS[name]
    except KeyError:
        available = ", ".join(sorted(PROBABILITY_MODELS)) or "(none)"
        raise ValueError(
            f"Unknown probability model {name!r}. Available: {available}"
        )


# --- built-in model: sigmoid_band_v0.1 --------------------------------------


class SigmoidBandV0_1:
    """Design-spec formula (``sigmoid_band_v0.1``).

    p_skip(z; a) = p_cap * sigmoid((z - z_enter)/tau_enter)
                         * sigmoid((z_exit - z)/tau_exit)

    with, for a clamped to [0, 1]:
        p_cap     = 0.40 + 0.40 * a
        z_enter   = -1.8 - 5.0 * a**1.35
        tau_enter = 0.55 + 0.35 * a
        z_exit    = 4.2 + 1.0 * a
        tau_exit  = 0.45
    """

    name = "sigmoid_band_v0.1"

    @staticmethod
    def params_from_aggressiveness(a):
        a = _clamp(a, 0.0, 1.0)
        return {
            "p_cap": 0.40 + 0.40 * a,
            "z_enter": -1.8 - 5.0 * (a ** 1.35),
            "tau_enter": 0.55 + 0.35 * a,
            "z_exit": 4.2 + 1.0 * a,
            "tau_exit": 0.45,
        }

    @staticmethod
    def skip_probability(z, params):
        p = (
            params["p_cap"]
            * _sigmoid((z - params["z_enter"]) / params["tau_enter"])
            * _sigmoid((params["z_exit"] - z) / params["tau_exit"])
        )
        return _clamp(p, 0.0, 1.0)


register(SigmoidBandV0_1.name, SigmoidBandV0_1())


# --- built-in model: monotone_saturate_v0.1 (default) -----------------------


class MonotoneSaturateV0_1:
    """Monotone saturating formula (``monotone_saturate_v0.1``) -- default.

    p_skip(z; a) = p_cap * sigmoid((z - z_enter)/tau_enter)

    with, for a clamped to [0, 1]:
        p_cap     = 0.5049 + 0.3975 * a + 0.0255 * a**2
        z_enter   = -0.0068 - 10.2115 * a**1.690
        tau_enter = 1.20

    No falling edge (2026-09-01 user decision)
    ==========================================
    Earlier models (``sigmoid_band_v0.*``) multiply in a second, descending
    sigmoid so that p decays again at high z. That term is deliberately
    absent here: the real harm of late-generation skipping is missing
    full-DiT refinement, and moderating that is the responsibility of the
    **skip window** (docs/SPEC-alpha.md Sec. 4.4), not of the probability
    formula. The formula is purified to a monotone saturating function of the
    absolute trajectory coordinate z, which keeps it free of any implicit
    dependence on a particular step count or schedule shape.

    Calibration canon
    =================
    ``experiment-HareSkip/v02-proposal/r2/derivation_r2.json`` ->
    ``final_params.mono_r2`` (stage-4 recalibration campaign, 2026-07..09;
    p_cap fit shared with ``sigmoid_band_v0.2``'s z_enter/tau_enter).
    On the Beta 30-step (Shift 3) reference schedule with the default skip
    window and zone streak limits, a = 0.2 / 0.55 / 0.9 yields roughly
    5 / 10 / 15 skipped steps.
    """

    name = "monotone_saturate_v0.1"

    @staticmethod
    def params_from_aggressiveness(a):
        a = _clamp(a, 0.0, 1.0)
        return {
            "p_cap": 0.5049 + 0.3975 * a + 0.0255 * a ** 2,
            "z_enter": -0.0068 - 10.2115 * (a ** 1.690),
            "tau_enter": 1.20,
        }

    @staticmethod
    def skip_probability(z, params):
        p = params["p_cap"] * _sigmoid(
            (z - params["z_enter"]) / params["tau_enter"]
        )
        return _clamp(p, 0.0, 1.0)


register(MonotoneSaturateV0_1.name, MonotoneSaturateV0_1())


# --- built-in model: sigmoid_band_v0.2 --------------------------------------


class SigmoidBandV0_2(SigmoidBandV0_1):
    """Recalibrated band formula (``sigmoid_band_v0.2``).

    Same shape as ``sigmoid_band_v0.1`` (rise * fall product of sigmoids);
    only the aggressiveness mapping is refitted, with, for a in [0, 1]:
        p_cap     = 0.5419 + 0.3427 * a + 0.0510 * a**2
        z_enter   = -0.0068 - 10.2115 * a**1.690
        tau_enter = 1.20
        z_exit    = 4.94 + 1.00 * a
        tau_exit  = 0.81

    Calibration canon:
    ``experiment-HareSkip/v02-proposal/r2/derivation_r2.json`` ->
    ``final_params.v02_r2``. Registered for A/B comparison and rollback; the
    default model is ``monotone_saturate_v0.1``. ``sigmoid_band_v0.1`` is
    left untouched so older images stay reproducible.
    """

    name = "sigmoid_band_v0.2"

    @staticmethod
    def params_from_aggressiveness(a):
        a = _clamp(a, 0.0, 1.0)
        return {
            "p_cap": 0.5419 + 0.3427 * a + 0.0510 * a ** 2,
            "z_enter": -0.0068 - 10.2115 * (a ** 1.690),
            "tau_enter": 1.20,
            "z_exit": 4.94 + 1.00 * a,
            "tau_exit": 0.81,
        }


register(SigmoidBandV0_2.name, SigmoidBandV0_2())


# --- default model ----------------------------------------------------------

#: Single source of truth for the default probability model name. Referenced
#: by skip_pattern.generate_skip_pattern, state.RuntimeState and the UI
#: estimate in script.py instead of hardcoding the name in each place.
DEFAULT_PROBABILITY_MODEL = "monotone_saturate_v0.1"

assert DEFAULT_PROBABILITY_MODEL in PROBABILITY_MODELS
