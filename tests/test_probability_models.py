"""Unit tests for hareskip.probability_models (pure stdlib + pytest)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hareskip import probability_models as pm

MODEL = pm.get_model("sigmoid_band_v0.1")


def test_method_identity():
    assert pm.METHOD_NAME == "HareSkipStochasticDensity"
    assert pm.METHOD_VERSION == "0.1"


@pytest.mark.parametrize(
    "a,p_cap,z_enter,tau_enter,z_exit,tau_exit",
    [
        (0.0, 0.40, -1.80, 0.55, 4.20, 0.45),
        (0.5, 0.60, -3.758, 0.725, 4.70, 0.45),
        (1.0, 0.80, -6.80, 0.90, 5.20, 0.45),
    ],
)
def test_params_table(a, p_cap, z_enter, tau_enter, z_exit, tau_exit):
    params = MODEL.params_from_aggressiveness(a)
    assert params["p_cap"] == pytest.approx(p_cap, abs=1e-9)
    assert params["z_enter"] == pytest.approx(z_enter, abs=0.01)
    assert params["tau_enter"] == pytest.approx(tau_enter, abs=1e-9)
    assert params["z_exit"] == pytest.approx(z_exit, abs=1e-9)
    assert params["tau_exit"] == pytest.approx(tau_exit, abs=1e-9)


def test_aggressiveness_clamped():
    lo = MODEL.params_from_aggressiveness(-1.0)
    hi = MODEL.params_from_aggressiveness(2.0)
    assert lo == MODEL.params_from_aggressiveness(0.0)
    assert hi == MODEL.params_from_aggressiveness(1.0)


@pytest.mark.parametrize("a", [0.0, 0.3, 0.5, 0.7, 1.0])
def test_probability_in_unit_interval(a):
    params = MODEL.params_from_aggressiveness(a)
    for i in range(-120, 121):
        z = i / 10.0  # sweep [-12, 12]
        p = MODEL.skip_probability(z, params)
        assert 0.0 <= p <= 1.0


@pytest.mark.parametrize("a", [0.0, 0.5, 1.0])
def test_rise_then_fall_shape(a):
    params = MODEL.params_from_aggressiveness(a)
    zs = [i / 10.0 for i in range(-120, 121)]
    ps = [MODEL.skip_probability(z, params) for z in zs]
    # Argmax is in the band interior, not at either sweep extreme.
    argmax = max(range(len(ps)), key=lambda k: ps[k])
    assert 0 < argmax < len(ps) - 1
    peak_z = zs[argmax]
    assert -8.0 < peak_z < 5.5
    # p is near zero at both far extremes (well below the peak).
    assert ps[0] < 0.01
    assert ps[-1] < 0.01
    # And clearly rises to a nontrivial peak.
    assert ps[argmax] > 0.1


def test_sigmoid_no_overflow_extremes():
    # Large-magnitude arguments must not raise OverflowError.
    assert pm._sigmoid(1000.0) == pytest.approx(1.0)
    assert pm._sigmoid(-1000.0) == pytest.approx(0.0)


def test_registry_get_and_unknown_raises():
    assert pm.get_model("sigmoid_band_v0.1") is MODEL
    with pytest.raises(ValueError) as exc:
        pm.get_model("does_not_exist")
    # Error lists available names.
    assert "sigmoid_band_v0.1" in str(exc.value)


def test_register_roundtrip():
    class Dummy:
        @staticmethod
        def params_from_aggressiveness(a):
            return {"a": a}

        @staticmethod
        def skip_probability(z, params):
            return 0.0

    model = Dummy()
    pm.register("dummy_test_model", model)
    try:
        assert pm.get_model("dummy_test_model") is model
    finally:
        pm.PROBABILITY_MODELS.pop("dummy_test_model", None)


# --- 2026-09-01: monotone_saturate_v0.1 (new default) + sigmoid_band_v0.2 ---

MONO = pm.get_model("monotone_saturate_v0.1")
BAND2 = pm.get_model("sigmoid_band_v0.2")

_A_GRID = [i / 10.0 for i in range(11)]
_Z_GRID = [-15.0 + 0.5 * i for i in range(61)]  # -15 .. 15 inclusive


def test_registry_contains_three_models():
    for name in (
        "sigmoid_band_v0.1",
        "sigmoid_band_v0.2",
        "monotone_saturate_v0.1",
    ):
        assert name in pm.PROBABILITY_MODELS
        assert pm.get_model(name) is pm.PROBABILITY_MODELS[name]


def test_default_probability_model_is_monotone_and_registered():
    assert pm.DEFAULT_PROBABILITY_MODEL == "monotone_saturate_v0.1"
    assert pm.get_model(pm.DEFAULT_PROBABILITY_MODEL) is MONO
    assert MONO.name == "monotone_saturate_v0.1"
    assert BAND2.name == "sigmoid_band_v0.2"


@pytest.mark.parametrize(
    "a,p_cap,z_enter",
    [
        (0.0, 0.5049, -0.0068),
        (0.5, 0.710025, -3.171619),
        (1.0, 0.9279, -10.2183),
    ],
)
def test_mono_params_table(a, p_cap, z_enter):
    params = MONO.params_from_aggressiveness(a)
    assert params["p_cap"] == pytest.approx(p_cap, abs=1e-4)
    assert params["z_enter"] == pytest.approx(z_enter, abs=1e-4)
    assert params["tau_enter"] == pytest.approx(1.20, abs=1e-4)
    # The monotone model deliberately has no falling edge.
    assert "z_exit" not in params
    assert "tau_exit" not in params


@pytest.mark.parametrize(
    "a,p_cap,z_enter,z_exit",
    [
        (0.0, 0.5419, -0.0068, 4.94),
        (0.5, 0.726, -3.171619, 5.44),
        (1.0, 0.9356, -10.2183, 5.94),
    ],
)
def test_band2_params_table(a, p_cap, z_enter, z_exit):
    params = BAND2.params_from_aggressiveness(a)
    assert params["p_cap"] == pytest.approx(p_cap, abs=1e-4)
    assert params["z_enter"] == pytest.approx(z_enter, abs=1e-4)
    assert params["tau_enter"] == pytest.approx(1.20, abs=1e-4)
    assert params["z_exit"] == pytest.approx(z_exit, abs=1e-4)
    assert params["tau_exit"] == pytest.approx(0.81, abs=1e-4)


def test_mono_non_decreasing_in_z():
    for a in _A_GRID:
        params = MONO.params_from_aggressiveness(a)
        prev = None
        for z in _Z_GRID:
            p = MONO.skip_probability(z, params)
            if prev is not None:
                assert p >= prev - 1e-12, f"a={a} z={z}"
            prev = p


def test_mono_non_decreasing_in_aggressiveness():
    for z in _Z_GRID:
        prev = None
        for a in _A_GRID:
            p = MONO.skip_probability(z, MONO.params_from_aggressiveness(a))
            if prev is not None:
                assert p >= prev - 1e-12, f"z={z} a={a}"
            prev = p


def test_mono_aggressiveness_clamped():
    assert MONO.params_from_aggressiveness(-1.0) == (
        MONO.params_from_aggressiveness(0.0)
    )
    assert MONO.params_from_aggressiveness(2.0) == (
        MONO.params_from_aggressiveness(1.0)
    )


def test_band2_aggressiveness_clamped():
    assert BAND2.params_from_aggressiveness(-1.0) == (
        BAND2.params_from_aggressiveness(0.0)
    )
    assert BAND2.params_from_aggressiveness(2.0) == (
        BAND2.params_from_aggressiveness(1.0)
    )


@pytest.mark.parametrize("model", [MONO, BAND2])
@pytest.mark.parametrize("a", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("z", [-1e6, 1e6])
def test_extreme_z_stays_in_unit_interval(model, a, z):
    p = model.skip_probability(z, model.params_from_aggressiveness(a))
    assert 0.0 <= p <= 1.0


@pytest.mark.parametrize("a", [0.0, 0.5, 1.0])
def test_mono_asymptotes(a):
    params = MONO.params_from_aggressiveness(a)
    assert MONO.skip_probability(1e6, params) == pytest.approx(
        params["p_cap"], abs=1e-9
    )
    assert MONO.skip_probability(-1e6, params) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("a", [0.0, 0.5, 1.0])
def test_band2_has_falling_edge(a):
    params = BAND2.params_from_aggressiveness(a)
    peak = BAND2.skip_probability(params["z_exit"] - 3.0, params)
    beyond = BAND2.skip_probability(params["z_exit"] + 5.0, params)
    assert beyond < peak
    assert beyond < 0.01


@pytest.mark.parametrize("a", [0.0, 0.5, 1.0])
def test_mono_probability_in_unit_interval_sweep(a):
    params = MONO.params_from_aggressiveness(a)
    for i in range(-150, 151):
        p = MONO.skip_probability(i / 10.0, params)
        assert 0.0 <= p <= 1.0
