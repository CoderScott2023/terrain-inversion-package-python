import warnings

import numpy as np
import pytest

from helpers import scene
from topoinv import estimate_valid, invert, render, saturated, to_reflectance

AZ, ALT = 45.0, 40.0


def _image(n=24, amp=0.4):
    z, cell = scene(n, amp)
    return render(z, azimuth=AZ, altitude=ALT, cellsize=cell), z - z.mean(), cell


def test_integer_imagery_is_refused_without_a_scale():
    image, _, cell = _image()
    dn = (np.clip(image, 0, 1)*255).astype(np.uint8)
    with pytest.raises(ValueError, match='digital number'):
        invert(dn, azimuth=AZ, altitude=ALT, cellsize=cell)


def test_scaled_integer_imagery_recovers_sane_relief():
    image, truth, cell = _image()
    dn = (np.clip(image, 0, 1)*255).astype(np.uint8)
    dem = invert(dn, azimuth=AZ, altitude=ALT, cellsize=cell, scale=1/255, maxiter=800,
                 warn_unconverged=False)
    assert np.abs(dem).max() < 20*np.abs(truth).max()


def test_unscaled_digital_numbers_would_have_been_garbage():
    image, truth, cell = _image()
    dn = (np.clip(image, 0, 1)*255).astype(np.uint8).astype(np.float64)
    dem = invert(dn, azimuth=AZ, altitude=ALT, cellsize=cell, maxiter=400,
                 warn_unconverged=False)
    assert np.abs(dem).max() > 50*np.abs(truth).max()


def test_implausible_peak_warns():
    image, _, _ = _image()
    with pytest.warns(RuntimeWarning, match='plausible reflectance'):
        to_reflectance(image.astype(np.float64)*50.0)


def test_plausible_reflectance_does_not_warn():
    image, _, _ = _image()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        to_reflectance(image.astype(np.float64))
    assert not [x for x in caught if issubclass(x.category, RuntimeWarning)]


def test_invert_can_return_the_optimiser_result():
    image, _, cell = _image()
    dem, res = invert(image, azimuth=AZ, altitude=ALT, cellsize=cell, maxiter=50,
                      return_result=True, warn_unconverged=False)
    assert dem.shape == image.shape
    assert res.nit <= 50 and np.isfinite(res.fun)


def test_hitting_the_iteration_limit_warns():
    image, _, cell = _image()
    with pytest.warns(RuntimeWarning, match='did not converge'):
        invert(image, azimuth=AZ, altitude=ALT, cellsize=cell, maxiter=3)


def test_converged_solve_does_not_warn():
    image, _, cell = _image()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        invert(image, azimuth=AZ, altitude=ALT, cellsize=cell, maxiter=30000)
    assert not [x for x in w if 'did not converge' in str(x.message)]


def test_pileup_saturation_is_detected_on_float_imagery():
    image, _, _ = _image()
    clipped = image.astype(np.float64).copy()
    clipped[clipped > 0.8] = 0.8
    flagged = saturated(clipped)
    assert flagged.sum() > 0
    assert not estimate_valid(clipped)[flagged].any()


def test_clean_float_imagery_has_no_false_saturation():
    image, _, _ = _image()
    assert saturated(image.astype(np.float64)).sum() == 0


def test_pileup_saturation_is_detected_on_digital_numbers():
    image, _, _ = _image(amp=1.2)
    dn = np.clip(image*400, 0, 255).astype(np.uint8)
    assert saturated(dn).sum() > 0
