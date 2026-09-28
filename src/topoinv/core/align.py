import warnings
from typing import NamedTuple

import numpy as np


class Alignment(NamedTuple):
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    residuals: np.ndarray
    inliers: np.ndarray
    rmse: float

    def apply(self, points):
        points = np.asarray(points, dtype=np.float64)
        return self.scale * (self.rotation @ points.T).T + self.translation

    def inverse(self):
        scale = 1.0 / self.scale
        rotation = self.rotation.T
        return Alignment(scale, rotation, -scale * (rotation @ self.translation),
                         self.residuals, self.inliers, self.rmse)

    @property
    def matrix(self):
        M = np.eye(4)
        M[:3, :3] = self.scale * self.rotation
        M[:3, 3] = self.translation
        return M


def _as_points(array, name):
    points = np.asarray(array, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(name + " must have shape (N, 3), got " + str(points.shape))
    if not np.all(np.isfinite(points)):
        raise ValueError(name + " contains non-finite values")
    return points


def _fit_scale_translation(source, target, rotation, with_scale):
    rotated = (rotation @ source.T).T
    rc, tc = rotated.mean(0), target.mean(0)
    dr, dt = rotated - rc, target - tc
    if with_scale:
        denominator = float((dr * dr).sum())
        if denominator <= 1e-15:
            raise ValueError("source points are coincident; scale is unobservable")
        scale = float((dr * dt).sum() / denominator)
    else:
        scale = 1.0
    return scale, rotation, tc - scale * rc


def collinearity(points):
    centred = points - points.mean(0)
    singular = np.linalg.svd(centred, compute_uv=False)
    if singular[0] <= 1e-15:
        return 0.0
    return float(singular[1] / singular[0])


def _fit_similarity(source, target, with_scale):
    sc, tc = source.mean(0), target.mean(0)
    ds, dt = source - sc, target - tc
    variance = float((ds * ds).sum() / len(source))
    if variance <= 1e-15:
        raise ValueError("source points are coincident; rotation is unobservable")

    covariance = (dt.T @ ds) / len(source)
    U, S, Vt = np.linalg.svd(covariance)
    correction = np.ones(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        correction[-1] = -1.0
    rotation = U @ np.diag(correction) @ Vt
    scale = float((S * correction).sum() / variance) if with_scale else 1.0
    return scale, rotation, tc - scale * (rotation @ sc)


def _residuals(source, target, scale, rotation, translation):
    predicted = scale * (rotation @ source.T).T + translation
    return np.linalg.norm(predicted - target, axis=1)


def similarity_fit(source, target, rotation=None, with_scale=True, robust=False,
                   threshold=None, iterations=200, min_inliers=None, seed=0):
    source = _as_points(source, "source")
    target = _as_points(target, "target")
    if len(source) != len(target):
        raise ValueError("source and target must have the same number of points, got "
                         + str(len(source)) + " and " + str(len(target)))

    if rotation is None:
        minimum = 3
        straightness = collinearity(source)
        if straightness < 1e-3:
            warnings.warn(
                "source control points are nearly collinear (spread ratio "
                + format(straightness, '.2e') + "); rotation about that axis is "
                "unconstrained. Supply rotation= or use control points that are "
                "not in a straight line.", RuntimeWarning, stacklevel=2)

        def fit(s, t):
            return _fit_similarity(s, t, with_scale)
    else:
        rotation = np.asarray(rotation, dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError("rotation must be 3x3, got " + str(rotation.shape))
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6):
            raise ValueError("rotation is not orthonormal")
        minimum = 2 if with_scale else 1
        def fit(s, t):
            return _fit_scale_translation(s, t, rotation, with_scale)

    if len(source) < minimum:
        raise ValueError("need at least " + str(minimum) + " point pairs, got "
                         + str(len(source)))

    if not robust:
        scale, R, t = fit(source, target)
        residuals = _residuals(source, target, scale, R, t)
        inliers = np.ones(len(source), dtype=bool)
        return Alignment(scale, R, t, residuals, inliers,
                         float(np.sqrt((residuals**2).mean())))

    if threshold is None:
        coarse = fit(source, target)
        spread = _residuals(source, target, *coarse)
        threshold = max(3.0 * float(np.median(spread)), 1e-9)
    if min_inliers is None:
        min_inliers = minimum

    rng = np.random.default_rng(seed)
    best_inliers = None
    best_count = -1
    for _ in range(iterations):
        sample = rng.choice(len(source), minimum, replace=False)
        try:
            candidate = fit(source[sample], target[sample])
        except (ValueError, np.linalg.LinAlgError):
            continue
        inliers = _residuals(source, target, *candidate) < threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count, best_inliers = count, inliers

    if best_inliers is None or best_count < min_inliers:
        raise ValueError("RANSAC found no consensus at threshold " + str(threshold)
                         + "; best support was " + str(max(best_count, 0)) + " of "
                         + str(len(source)))

    scale, R, t = fit(source[best_inliers], target[best_inliers])
    residuals = _residuals(source, target, scale, R, t)
    return Alignment(scale, R, t, residuals, best_inliers,
                     float(np.sqrt((residuals[best_inliers]**2).mean())))


def coregister(points, source_control, target_control, rotation=None, with_scale=True,
               robust=False, return_alignment=False, **kwargs):
    alignment = similarity_fit(source_control, target_control, rotation=rotation,
                               with_scale=with_scale, robust=robust, **kwargs)
    moved = alignment.apply(points)
    return (moved, alignment) if return_alignment else moved
