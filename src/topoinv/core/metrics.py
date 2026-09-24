import numpy as np

from ..geometry.grid import horn_gradients


def _pair(a, b, valid):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if valid is None:
        valid = np.isfinite(a) & np.isfinite(b)
    return a, b, valid


def dem_error(dem1, dem2, valid=None):
    dem1, dem2, valid = _pair(dem1, dem2, valid)
    diff = np.where(valid, dem1 - dem2, 0.0)
    n_valid = max(int(valid.sum()), 1)
    return np.sqrt(np.sum(diff**2) / n_valid), np.sum(np.abs(diff)) / n_valid


def slope_degrees(dem, cellsize=1.0):
    p, q = horn_gradients(dem, cellsize)
    return np.degrees(np.arctan(np.hypot(p, q)))


def slope_error(dem1, dem2, cellsize=1.0, valid=None):
    return dem_error(slope_degrees(dem1, cellsize), slope_degrees(dem2, cellsize), valid)


def correlation(a, b, valid=None):
    a, b, valid = _pair(a, b, valid)
    return float(np.corrcoef(a[valid].ravel(), b[valid].ravel())[0, 1])


def slope_correlation(dem1, dem2, cellsize=1.0, valid=None):
    return correlation(slope_degrees(dem1, cellsize), slope_degrees(dem2, cellsize), valid)


def summary(estimate, truth, cellsize=1.0, valid=None):
    h_rmse, h_mae = dem_error(estimate, truth, valid)
    s_rmse, s_mae = slope_error(estimate, truth, cellsize, valid)
    return {
        'slope_rmse_deg': s_rmse,
        'slope_mae_deg': s_mae,
        'slope_corr': slope_correlation(estimate, truth, cellsize, valid),
        'height_rmse': h_rmse,
        'height_mae': h_mae,
        'height_corr': correlation(estimate, truth, valid),
    }
