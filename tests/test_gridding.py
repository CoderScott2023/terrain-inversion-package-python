import numpy as np
import pytest

from topoinv.sfm.reconstruct import AGGREGATORS, points_to_dem


def _surface(n=40000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(100.0, 140.0, n)
    y = rng.uniform(200.0, 230.0, n)
    z = 5.0 + 0.5*np.sin(x/4)*np.cos(y/3)
    return np.stack([x, y, z], 1)


def test_grid_shape_and_transform_follow_bounds_and_cellsize():
    dem, transform = points_to_dem(_surface(), 1.0)
    assert dem.shape == (30, 40)
    origin_x = transform.c if hasattr(transform, 'c') else transform[2]
    origin_y = transform.f if hasattr(transform, 'f') else transform[5]
    assert np.isclose(origin_x, 100.0) and np.isclose(origin_y, 230.0)


def test_median_grid_matches_the_analytic_surface():
    dem, _ = points_to_dem(_surface(), 1.0, aggregate='median')
    rows, cols = np.indices(dem.shape)
    cx, cy = 100.0 + cols + 0.5, 230.0 - rows - 0.5
    truth = 5.0 + 0.5*np.sin(cx/4)*np.cos(cy/3)
    finite = np.isfinite(dem)
    assert np.sqrt(((dem[finite] - truth[finite])**2).mean()) < 0.02


@pytest.mark.parametrize('aggregate', AGGREGATORS)
def test_every_aggregator_produces_a_full_grid(aggregate):
    dem, _ = points_to_dem(_surface(), 1.0, aggregate=aggregate)
    assert dem.shape == (30, 40)
    assert np.isfinite(dem).all()


def test_aggregators_order_as_expected():
    points = _surface()
    lo, _ = points_to_dem(points, 1.0, aggregate='min')
    mid, _ = points_to_dem(points, 1.0, aggregate='median')
    hi, _ = points_to_dem(points, 1.0, aggregate='max')
    assert np.all(lo <= mid) and np.all(mid <= hi)


def test_count_aggregator_totals_the_input():
    points = _surface(n=5000)
    counts, _ = points_to_dem(points, 1.0, aggregate='count')
    assert np.nansum(counts) == len(points)


def test_grid_is_north_up():
    points = np.array([[1.0, 19.0, 100.0], [1.0, 1.0, 0.0]])
    dem, _ = points_to_dem(points, 1.0, bounds=(0.0, 0.0, 20.0, 20.0))
    assert np.nanmax(dem[:2]) == 100.0
    assert np.nanmax(dem[-2:]) == 0.0


def test_min_points_thins_sparse_cells():
    rng = np.random.default_rng(1)
    points = np.stack([rng.uniform(0, 20, 400), rng.uniform(0, 20, 400),
                       np.full(400, 7.0)], 1)
    coverage = [np.isfinite(points_to_dem(points, 1.0, min_points=m)[0]).mean()
                for m in (1, 3, 10)]
    assert coverage[0] > coverage[1] > coverage[2]


def test_fill_holes_respects_a_distance_limit():
    rng = np.random.default_rng(1)
    points = np.stack([rng.uniform(0, 20, 180), rng.uniform(0, 20, 180),
                       np.full(180, 7.0)], 1)
    base = np.isfinite(points_to_dem(points, 1.0, min_points=3)[0]).mean()
    near = np.isfinite(points_to_dem(points, 1.0, min_points=3, fill_holes=1.0)[0]).mean()
    far = np.isfinite(points_to_dem(points, 1.0, min_points=3, fill_holes=5.0)[0]).mean()
    full = np.isfinite(points_to_dem(points, 1.0, min_points=3, fill_holes=True)[0]).mean()
    assert base < near < far < full
    assert full == 1.0


def test_explicit_bounds_crop_the_grid():
    dem, transform = points_to_dem(_surface(), 2.0, bounds=(110.0, 210.0, 120.0, 220.0))
    assert dem.shape == (5, 5)
    origin_x = transform.c if hasattr(transform, 'c') else transform[2]
    assert np.isclose(origin_x, 110.0)


def test_non_finite_points_are_dropped_not_propagated():
    points = _surface(n=2000)
    dirty = np.vstack([points, np.full((20, 3), np.nan)])
    clean, _ = points_to_dem(points, 2.0)
    mixed, _ = points_to_dem(dirty, 2.0, bounds=(100.0, 200.0, 140.0, 230.0))
    assert np.isfinite(mixed).sum() > 0


@pytest.mark.parametrize('bad,message', [
    ('shape', 'shape'),
    ('aggregate', 'aggregate must be one of'),
    ('cellsize', 'cellsize must be positive'),
    ('bounds', 'xmax > xmin'),
    ('outside', 'no points fall inside'),
    ('empty', 'no finite points'),
])
def test_invalid_inputs_are_rejected(bad, message):
    points = _surface(n=100)
    calls = {
        'shape': lambda: points_to_dem(points[:, :2], 1.0),
        'aggregate': lambda: points_to_dem(points, 1.0, aggregate='kriging'),
        'cellsize': lambda: points_to_dem(points, 0.0),
        'bounds': lambda: points_to_dem(points, 1.0, bounds=(10.0, 10.0, 0.0, 0.0)),
        'outside': lambda: points_to_dem(points, 1.0, bounds=(1e5, 1e5, 1e5 + 10, 1e5 + 10)),
        'empty': lambda: points_to_dem(np.full((5, 3), np.nan), 1.0),
    }
    with pytest.raises(ValueError, match=message):
        calls[bad]()
