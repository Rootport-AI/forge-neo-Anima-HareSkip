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
