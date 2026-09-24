import numpy as np
import pytest
from scipy.optimize import approx_fprime

from helpers import scene
from topoinv import estimate_valid, make_objective, prepare_geometry
from topoinv import MODELS, render


def _objective(n, cell, model, alpha, azimuth=57.0, altitude=38.0, view_azimuth=0.0,
               view_altitude=90.0, seed=0):
    truth = np.random.default_rng(seed).normal(size=(n, n))*0.3
    image = render(truth, azimuth=azimuth, altitude=altitude,
                                      cellsize=cell, view_azimuth=view_azimuth,
                                      view_altitude=view_altitude, model=model)
    s, v, phase, fn = prepare_geometry(azimuth, altitude, None, view_azimuth,
                                       view_altitude, model, False)
    return make_objective(np.asarray(image, dtype=np.float64), estimate_valid(image),
                          s, v, phase, fn, cell, 1.0, alpha)


@pytest.mark.parametrize('model', sorted(MODELS))
@pytest.mark.parametrize('n,cell', [(9, 0.7), (13, 1.4)])
def test_analytic_gradient_matches_numerical(model, n, cell):
    obj = _objective(n, cell, model, alpha=1e-3)
    z = np.random.default_rng(n).normal(size=n*n)*0.2
    analytic = obj(z)[1]
    numerical = approx_fprime(z, lambda t: obj(t)[0], 1e-7)
    assert np.linalg.norm(analytic - numerical) / np.linalg.norm(numerical) < 1e-5


@pytest.mark.parametrize('alpha', [0.0, 1e-7, 1e-2])
def test_gradient_correct_across_regularisation_weights(alpha):
    obj = _objective(11, 0.9, 'lunar_lambert', alpha=alpha)
    z = np.random.default_rng(3).normal(size=121)*0.2
    numerical = approx_fprime(z, lambda t: obj(t)[0], 1e-7)
    assert np.linalg.norm(obj(z)[1] - numerical) / np.linalg.norm(numerical) < 1e-5


def test_gradient_correct_for_oblique_viewing_geometry():
    obj = _objective(11, 0.9, 'lunar_lambert', alpha=1e-4, view_azimuth=120.0, view_altitude=55.0)
    z = np.random.default_rng(5).normal(size=121)*0.2
    numerical = approx_fprime(z, lambda t: obj(t)[0], 1e-7)
    assert np.linalg.norm(obj(z)[1] - numerical) / np.linalg.norm(numerical) < 1e-5


def test_objective_is_invariant_to_constant_offset():
    obj = _objective(10, 1.0, 'lunar_lambert', alpha=1e-5)
    z = np.random.default_rng(7).normal(size=100)*0.2
    assert np.isclose(obj(z)[0], obj(z + 7.3)[0], rtol=1e-12)
