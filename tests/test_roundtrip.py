import numpy as np
import pytest

from helpers import block_mean, block_mean as _bm, corr, rmse, scene, slope_degrees
from topoinv import invert
from topoinv import render

AZ, ALT, N, K = 45.0, 40.0, 32, 4


def _mismatched_pair(amp=0.4, model='lunar_lambert'):
    hi, cell_hi = scene(N*K, amp)
    _, cell_lo = scene(N, amp)
    image_hi = render(hi, azimuth=AZ, altitude=ALT,
                                         cellsize=cell_hi, model=model)
    truth = block_mean(hi, K)
    return block_mean(image_hi, K), truth - truth.mean(), cell_lo


@pytest.mark.slow
def test_resolution_mismatched_roundtrip_recovers_slopes():
    image, truth, cell = _mismatched_pair()
    dem = invert(image, azimuth=AZ, altitude=ALT, cellsize=cell, maxiter=3000)
    assert corr(slope_degrees(dem, cell), slope_degrees(truth, cell)) > 0.85
    assert rmse(slope_degrees(dem, cell), slope_degrees(truth, cell)) < 6.0


@pytest.mark.slow
def test_resolution_mismatched_roundtrip_recovers_heights():
    image, truth, cell = _mismatched_pair()
    dem = invert(image, azimuth=AZ, altitude=ALT, cellsize=cell, maxiter=3000)
    assert corr(dem, truth) > 0.75
    assert rmse(dem, truth) < truth.std()


@pytest.mark.slow
def test_matched_model_beats_mismatched_model():
    image, truth, cell = _mismatched_pair(model='lunar_lambert')
    ref = slope_degrees(truth, cell)
    scores = {}
    for model in ('lunar_lambert', 'lambert'):
        dem = invert(image, azimuth=AZ, altitude=ALT, cellsize=cell,
                                 model=model, maxiter=3000)
        scores[model] = corr(slope_degrees(dem, cell), ref)
    assert scores['lunar_lambert'] > scores['lambert']


@pytest.mark.slow
def test_eight_bit_quantisation_is_tolerable():
    image, truth, cell = _mismatched_pair()
    quantised = np.round(np.clip(image, 0, 1)*255)/255.0
    dem = invert(quantised, azimuth=AZ, altitude=ALT, cellsize=cell, maxiter=3000)
    assert corr(slope_degrees(dem, cell), slope_degrees(truth, cell)) > 0.85


@pytest.mark.slow
def test_recovered_dem_is_finite_and_zero_mean():
    image, truth, cell = _mismatched_pair()
    dem = invert(image, azimuth=AZ, altitude=ALT, cellsize=cell, maxiter=500)
    assert np.all(np.isfinite(dem))
    assert abs(float(dem.mean())) < 1e-9
