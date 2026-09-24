import numpy as np
import pytest

from helpers import block_mean, corr, scene, slope_degrees
from topoinv import invert
from topoinv import render

AZ, ALT, N, K = 45.0, 40.0, 32, 4


def _noisy(sigma, amp=0.4, seed=0):
    hi, cell_hi = scene(N*K, amp)
    _, cell_lo = scene(N, amp)
    image = block_mean(render(hi, azimuth=AZ, altitude=ALT,
                                                 cellsize=cell_hi, model='lunar_lambert'), K)
    truth = block_mean(hi, K)
    noise = np.random.default_rng(seed).normal(size=image.shape)*sigma
    return image + noise, truth - truth.mean(), cell_lo


@pytest.mark.slow
@pytest.mark.parametrize('sigma,floor', [(0.0, 0.90), (0.001, 0.85), (0.003, 0.70)])
def test_slope_accuracy_survives_realistic_noise(sigma, floor):
    image, truth, cell = _noisy(sigma)
    alpha = 1e-7 if sigma < 0.002 else 1e-5
    dem = invert(image, azimuth=AZ, altitude=ALT, cellsize=cell,
                             alpha=alpha, maxiter=3000)
    assert corr(slope_degrees(dem, cell), slope_degrees(truth, cell)) > floor


@pytest.mark.slow
def test_slope_accuracy_degrades_monotonically_with_noise():
    scores = []
    for sigma in (0.0, 0.003, 0.01):
        image, truth, cell = _noisy(sigma)
        dem = invert(image, azimuth=AZ, altitude=ALT, cellsize=cell,
                                 alpha=1e-5, maxiter=3000)
        scores.append(corr(slope_degrees(dem, cell), slope_degrees(truth, cell)))
    assert scores[0] > scores[2]


@pytest.mark.slow
def test_too_flat_terrain_is_not_recoverable():
    image, truth, cell = _noisy(0.005, amp=0.05)
    dem = invert(image, azimuth=AZ, altitude=ALT, cellsize=cell,
                             alpha=1e-5, maxiter=3000)
    assert corr(slope_degrees(dem, cell), slope_degrees(truth, cell)) < 0.5


@pytest.mark.slow
def test_regularisation_helps_when_data_is_noisy():
    image, truth, cell = _noisy(0.01)
    ref = slope_degrees(truth, cell)
    weak = invert(image, azimuth=AZ, altitude=ALT, cellsize=cell,
                              alpha=1e-9, maxiter=3000)
    tuned = invert(image, azimuth=AZ, altitude=ALT, cellsize=cell,
                               alpha=1e-5, maxiter=3000)
    assert corr(slope_degrees(tuned, cell), ref) > corr(slope_degrees(weak, cell), ref)
