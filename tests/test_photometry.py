import numpy as np
import pytest

from topoinv import MODELS, lambert, lommel_seeliger, lunar_lambert, mcewen_L


@pytest.mark.parametrize('model', sorted(MODELS))
def test_models_preserve_shape_and_stay_finite(model):
    mu0 = np.linspace(0.05, 1.0, 12).reshape(3, 4)
    out = MODELS[model](mu0, np.full_like(mu0, 0.7), np.radians(30.0))
    assert out.shape == mu0.shape
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize('model', sorted(MODELS))
def test_reflectance_scales_linearly_with_albedo(model):
    mu0, mu, g = 0.6, 0.8, np.radians(25.0)
    assert np.isclose(MODELS[model](mu0, mu, g, 0.3), 0.3*MODELS[model](mu0, mu, g, 1.0))


def test_lommel_seeliger_exceeds_lambert_at_grazing_incidence():
    mu0 = np.cos(np.radians(80.0))
    assert lommel_seeliger(mu0, 1.0, 0.0) > lambert(mu0, 1.0, 0.0)


def test_mcewen_L_is_bounded_and_monotonic():
    L = mcewen_L(np.radians(np.arange(0.0, 180.0, 5.0)))
    assert L[0] == 1.0
    assert np.all((L >= 0.0) & (L <= 1.0))
    assert np.all(np.diff(L) <= 1e-12)


def test_lunar_lambert_interpolates_its_endpoints():
    mu0, mu = 0.4, 0.8
    assert np.isclose(lunar_lambert(mu0, mu, 0.0, L=1.0), lommel_seeliger(mu0, mu, 0.0))
    assert np.isclose(lunar_lambert(mu0, mu, 0.0, L=0.0), lambert(mu0, mu, 0.0))


@pytest.mark.parametrize('model', sorted(MODELS))
def test_reflectance_decreases_as_incidence_grows(model):
    mu0 = np.cos(np.radians([10.0, 30.0, 50.0, 70.0, 85.0]))
    r = MODELS[model](mu0, np.ones_like(mu0), np.radians(30.0))
    assert np.all(np.diff(r) < 0)
