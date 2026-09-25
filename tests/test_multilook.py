import warnings

import numpy as np
import pytest
from scipy.optimize import approx_fprime

from helpers import corr, scene, slope_degrees
from topoinv import (estimate_valid, invert, invert_multilook, make_objective,
                     prepare_geometry, render, slope_correlation)

ALT, N = 40.0, 24


def _views(azimuths, amp=0.4, albedo=1.0, n=N):
    z, cell = scene(n, amp)
    z = z - z.mean()
    images = [render(z, azimuth=a, altitude=ALT, cellsize=cell, albedo=albedo)
              for a in azimuths]
    return images, z, cell


def _solve(images, azimuths, cell, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return invert_multilook(images, azimuths, altitudes=[ALT]*len(azimuths),
                                cellsizes=cell, **kwargs)


def _slope_rmse(a, b, cell):
    return float(np.sqrt(((slope_degrees(a, cell) - slope_degrees(b, cell))**2).mean()))


def test_composite_gradient_matches_numerical():
    azimuths = [45.0, 135.0, 225.0]
    images, _, cell = _views(azimuths, n=11)
    objectives = []
    for image, azimuth in zip(images, azimuths):
        s, v, phase, fn = prepare_geometry(azimuth, ALT)
        objectives.append(make_objective(image.astype(np.float64), estimate_valid(image),
                                         s, v, phase, fn, cell, 1.0, 1e-3))
    z = np.random.default_rng(0).normal(size=121)*0.2
    analytic = sum(o(z)[1] for o in objectives)
    numerical = approx_fprime(z, lambda t: sum(o(t)[0] for o in objectives), 1e-7)
    assert np.linalg.norm(analytic - numerical) / np.linalg.norm(numerical) < 1e-5


def test_nadir_viewing_defaults_do_not_need_per_view_lists():
    images, truth, cell = _views([45.0, 135.0])
    dem = _solve(images, [45.0, 135.0], cell, maxiter=6000)
    assert slope_correlation(dem, truth, cell) > 0.99


def test_explicit_per_view_viewing_geometry_is_accepted():
    azimuths = [45.0, 135.0]
    images, truth, cell = _views(azimuths)
    dem = _solve(images, azimuths, cell, view_azimuths=[0.0, 0.0],
                 view_altitudes=[90.0, 90.0], maxiter=6000)
    assert slope_correlation(dem, truth, cell) > 0.99


def test_per_view_cellsizes_are_accepted():
    azimuths = [45.0, 135.0]
    images, truth, cell = _views(azimuths)
    dem = _solve(images, azimuths, [cell, cell], maxiter=6000)
    assert slope_correlation(dem, truth, cell) > 0.99


def test_zenith_may_replace_altitude():
    azimuths = [45.0, 135.0]
    images, truth, cell = _views(azimuths)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        dem = invert_multilook(images, azimuths, zeniths=[90.0 - ALT]*2,
                               cellsizes=cell, maxiter=6000)
    assert slope_correlation(dem, truth, cell) > 0.99


@pytest.mark.slow
def test_two_looks_beat_one_look():
    azimuths = [45.0, 135.0]
    images, truth, cell = _views(azimuths)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        single = invert(images[0], azimuth=45.0, altitude=ALT, cellsize=cell, maxiter=8000)
    multi = _solve(images, azimuths, cell, maxiter=8000)
    assert _slope_rmse(multi, truth, cell) < _slope_rmse(single, truth, cell)


@pytest.mark.slow
def test_four_looks_recover_terrain_under_unknown_albedo():
    n = N
    yy, xx = np.indices((n, n))
    patches = np.where(xx < n//2, 0.10, 0.18)
    azimuths = [45.0, 135.0, 225.0, 315.0]
    images, truth, cell = _views(azimuths, albedo=patches)
    dem = _solve(images, azimuths, cell, albedo=float(patches.mean()), maxiter=8000)
    assert slope_correlation(dem, truth, cell) > 0.9
    assert _slope_rmse(dem, truth, cell) < 5.0


@pytest.mark.slow
def test_single_look_fails_under_unknown_albedo():
    n = N
    yy, xx = np.indices((n, n))
    patches = np.where(xx < n//2, 0.10, 0.18)
    images, truth, cell = _views([45.0], albedo=patches)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        dem = invert(images[0], azimuth=45.0, altitude=ALT, cellsize=cell,
                     albedo=float(patches.mean()), maxiter=8000)
    assert slope_correlation(dem, truth, cell) < 0.5


def test_mismatched_angle_list_length_is_rejected():
    images, _, cell = _views([45.0, 135.0])
    with pytest.raises(ValueError, match='azimuths has 1 entries'):
        invert_multilook(images, [45.0], altitudes=[ALT], cellsizes=cell, maxiter=5)


def test_mismatched_image_shapes_are_rejected():
    images, _, cell = _views([45.0, 135.0])
    with pytest.raises(ValueError, match='share a shape'):
        invert_multilook([images[0], np.zeros((10, 10))], [45.0, 135.0],
                         altitudes=[ALT]*2, cellsizes=cell, maxiter=5)


def test_empty_image_list_is_rejected():
    with pytest.raises(ValueError, match='at least one image'):
        invert_multilook([], [], altitudes=[], cellsizes=1.0, maxiter=5)


def test_one_look_matches_single_image_invert():
    images, truth, cell = _views([45.0])
    single = _solve(images, [45.0], cell, maxiter=4000)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        direct = invert(images[0], azimuth=45.0, altitude=ALT, cellsize=cell, maxiter=4000)
    assert corr(single, direct) > 0.999


def test_multilook_can_return_the_optimiser_result():
    images, _, cell = _views([45.0, 135.0])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        dem, res = invert_multilook(images, [45.0, 135.0], altitudes=[ALT]*2,
                                    cellsizes=cell, maxiter=40, return_result=True)
    assert dem.shape == images[0].shape
    assert res.nit <= 40 and np.isfinite(res.fun)


def test_integer_imagery_is_refused_in_multilook():
    images, _, cell = _views([45.0, 135.0])
    dn = [(np.clip(i, 0, 1)*255).astype(np.uint8) for i in images]
    with pytest.raises(ValueError, match='digital number'):
        invert_multilook(dn, [45.0, 135.0], altitudes=[ALT]*2, cellsizes=cell, maxiter=5)
