import numpy as np
import pytest

from helpers import plane
from topoinv import LIT, MODELS, SELF_SHADOWED, direction_vector, render, surface_normals


def test_flat_ground_has_upward_normals():
    assert np.allclose(surface_normals(np.zeros((5, 5)), 1.0), np.array([0.0, 0.0, 1.0]))


def test_normals_are_unit_length():
    rng = np.random.default_rng(0)
    n = surface_normals(rng.normal(size=(20, 20))*5.0, 2.5)
    assert np.allclose(np.linalg.norm(n, axis=-1), 1.0)


def test_larger_cellsize_gives_gentler_slope():
    dem = np.tile(np.arange(9.0), (9, 1))
    assert surface_normals(dem, 10.0)[4, 4, 2] > surface_normals(dem, 1.0)[4, 4, 2]


def test_direction_vector_axes():
    assert np.allclose(direction_vector(0.0, 0.0), [0.0, 1.0, 0.0])
    assert np.allclose(direction_vector(np.pi/2, 0.0), [1.0, 0.0, 0.0])
    assert np.allclose(direction_vector(0.0, np.pi/2), [0.0, 0.0, 1.0])


@pytest.mark.parametrize('azimuth,expected', [(0.0, 'N'), (90.0, 'E'), (180.0, 'S'), (270.0, 'W')])
def test_compass_azimuth_lights_the_facing_slope(azimuth, expected):
    got = {f: render(plane(f), azimuth=azimuth, altitude=45,
                                        cellsize=1.0, model='lambert')[4, 4] for f in 'NESW'}
    assert max(got, key=got.get) == expected
    assert got[{'N': 'S', 'E': 'W', 'S': 'N', 'W': 'E'}[expected]] == 0.0


@pytest.mark.parametrize('altitude', [10.0, 30.0, 75.0])
def test_flat_lambertian_equals_sin_altitude(altitude):
    r = render(np.zeros((5, 5)), azimuth=45, altitude=altitude,
                                  cellsize=1.0, model='lambert')
    assert np.allclose(r, np.sin(np.radians(altitude)))


@pytest.mark.parametrize('model', sorted(MODELS))
def test_overhead_sun_and_nadir_view_return_albedo(model):
    r = render(np.zeros((5, 5)), azimuth=0, altitude=90,
                                  cellsize=1.0, albedo=0.14, model=model)
    assert np.allclose(r, 0.14)


def test_mask_marks_exactly_the_dark_pixels():
    dem, cell = np.outer(np.arange(12.0), np.arange(12.0))*0.1, 1.0
    r, m = render(dem, azimuth=45, altitude=20, cellsize=cell, return_mask=True)
    assert np.all((r == 0) == (m != LIT))
    assert set(np.unique(m)) <= {LIT, SELF_SHADOWED}


def test_borders_are_not_silently_zero():
    dem, _ = np.outer(np.arange(20.0), np.arange(20.0))*0.05, 1.0
    r = render(dem, azimuth=45, altitude=40, cellsize=1.0)
    assert r[0].max() > 0 and r[-1].max() > 0 and r[:, 0].max() > 0 and r[:, -1].max() > 0


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match='phong'):
        render(np.zeros((4, 4)), azimuth=45, altitude=30, model='phong')


def test_missing_sun_angle_is_rejected():
    with pytest.raises(ValueError):
        render(np.zeros((4, 4)), azimuth=45)
