import numpy as np
import pytest

from topoinv import Alignment, coregister, similarity_fit


def _transform(seed=0, n=40, scale=3.7, spread=5.0):
    rng = np.random.default_rng(seed)
    axis = rng.normal(0, 0.4, 3)
    angle = np.linalg.norm(axis)
    k = axis / angle
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + np.sin(angle)*Kx + (1 - np.cos(angle))*(Kx @ Kx)
    t = rng.normal(0, 50, 3)
    source = rng.normal(0, spread, (n, 3))
    return source, scale*(R @ source.T).T + t, scale, R, t


def test_known_rotation_recovers_scale_and_translation():
    source, target, scale, R, t = _transform()
    fit = similarity_fit(source, target, rotation=R)
    assert np.isclose(fit.scale, scale)
    assert np.allclose(fit.translation, t)
    assert fit.rmse < 1e-10


def test_unknown_rotation_recovers_full_similarity():
    source, target, scale, R, t = _transform()
    fit = similarity_fit(source, target)
    assert np.isclose(fit.scale, scale)
    assert np.allclose(fit.rotation, R, atol=1e-10)
    assert np.allclose(fit.translation, t, atol=1e-9)


def test_rigid_fit_holds_scale_at_one():
    source, _, _, R, t = _transform(scale=1.0)
    target = (R @ source.T).T + t
    fit = similarity_fit(source, target, with_scale=False)
    assert fit.scale == 1.0
    assert np.allclose(fit.rotation, R, atol=1e-10)
    assert fit.rmse < 1e-10


def test_rotation_is_proper_not_a_reflection():
    source, target, _, _, _ = _transform()
    assert np.isclose(np.linalg.det(similarity_fit(source, target).rotation), 1.0)


@pytest.mark.parametrize('n,rotation_known,ok', [(1, True, False), (2, True, True),
                                                 (2, False, False), (3, False, True)])
def test_minimum_point_counts(n, rotation_known, ok):
    source, target, _, R, _ = _transform()
    kwargs = {'rotation': R} if rotation_known else {}
    if ok:
        similarity_fit(source[:n], target[:n], **kwargs)
    else:
        with pytest.raises(ValueError, match='at least'):
            similarity_fit(source[:n], target[:n], **kwargs)


def test_robust_fit_rejects_gross_outliers():
    source, target, scale, R, t = _transform(n=60)
    rng = np.random.default_rng(3)
    bad = rng.choice(60, 5, replace=False)
    corrupted = target.copy()
    corrupted[bad] += 50.0

    plain = similarity_fit(source, corrupted, rotation=R)
    robust = similarity_fit(source, corrupted, rotation=R, robust=True)

    assert abs(plain.scale - scale) > 0.05*scale
    assert np.isclose(robust.scale, scale, rtol=1e-6)
    assert sorted(np.where(~robust.inliers)[0]) == sorted(bad)


def test_robust_fit_matches_plain_fit_on_clean_data():
    source, target, scale, R, t = _transform()
    robust = similarity_fit(source, target, rotation=R, robust=True)
    assert np.isclose(robust.scale, scale)
    assert robust.inliers.all()


def test_residuals_expose_a_wrong_rotation_that_scale_hides():
    source, target, scale, R, t = _transform()
    yaw = np.radians(0.5)
    spin = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    fit = similarity_fit(source, target, rotation=spin @ R)
    assert abs(fit.scale - scale) < 0.001*scale
    assert fit.rmse > 0.01


def test_apply_and_inverse_round_trip():
    source, target, scale, R, t = _transform()
    fit = similarity_fit(source, target, rotation=R)
    cloud = np.random.default_rng(9).normal(0, 6, (200, 3))
    assert np.allclose(fit.inverse().apply(fit.apply(cloud)), cloud, atol=1e-9)


def test_matrix_matches_apply():
    source, target, _, R, _ = _transform()
    fit = similarity_fit(source, target, rotation=R)
    cloud = np.random.default_rng(4).normal(0, 6, (50, 3))
    homogeneous = np.hstack([cloud, np.ones((50, 1))])
    assert np.allclose((fit.matrix @ homogeneous.T).T[:, :3], fit.apply(cloud))


def test_coregister_moves_a_cloud_onto_the_target_frame():
    source, target, scale, R, t = _transform()
    cloud = np.random.default_rng(5).normal(0, 6, (100, 3))
    moved, fit = coregister(cloud, source, target, rotation=R, return_alignment=True)
    assert np.allclose(moved, scale*(R @ cloud.T).T + t)
    assert isinstance(fit, Alignment)


def test_residuals_are_per_point():
    source, target, _, R, _ = _transform(n=17)
    assert similarity_fit(source, target, rotation=R).residuals.shape == (17,)


@pytest.mark.parametrize('bad,message', [
    ('count', 'same number of points'),
    ('shape', 'shape'),
    ('nan', 'non-finite'),
    ('rotation', 'orthonormal'),
    ('coincident', 'coincident'),
])
def test_invalid_inputs_are_rejected(bad, message):
    source, target, _, R, _ = _transform()
    calls = {
        'count': lambda: similarity_fit(source[:10], target),
        'shape': lambda: similarity_fit(source[:, :2], target[:, :2], rotation=R),
        'nan': lambda: similarity_fit(np.full((5, 3), np.nan), target[:5]),
        'rotation': lambda: similarity_fit(source, target, rotation=np.ones((3, 3))),
        'coincident': lambda: similarity_fit(np.zeros((5, 3)), target[:5], rotation=R),
    }
    with pytest.raises(ValueError, match=message):
        calls[bad]()


def test_ransac_reports_when_no_consensus_exists():
    rng = np.random.default_rng(7)
    source = rng.normal(0, 5, (30, 3))
    target = rng.normal(0, 5, (30, 3))
    with pytest.raises(ValueError, match='consensus'):
        similarity_fit(source, target, rotation=np.eye(3), robust=True,
                       threshold=1e-9, min_inliers=25)


def _control(seed=0, n=25, scale=2.6):
    rng = np.random.default_rng(seed)
    axis = rng.normal(0, 0.3, 3)
    angle = np.linalg.norm(axis)
    k = axis/angle
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + np.sin(angle)*Kx + (1-np.cos(angle))*(Kx @ Kx)
    t = rng.normal(0, 100, 3)
    source = rng.normal(0, 5, (n, 3))
    return source, scale*(R @ source.T).T + t, scale, R, t


def test_georeference_places_a_cloud_in_world_coordinates():
    from topoinv.sfm.reconstruct import georeference
    source, world, scale, R, t = _control()
    cloud = np.random.default_rng(2).normal(0, 5, (300, 3))
    moved, fit = georeference(cloud, R, source, world, return_alignment=True)
    assert np.allclose(moved, scale*(R @ cloud.T).T + t)
    assert np.isclose(fit.scale, scale)
    assert fit.rmse < 1e-10


def test_georeference_robust_mode_rejects_bad_control_points():
    from topoinv.sfm.reconstruct import georeference
    source, world, scale, R, t = _control()
    rng = np.random.default_rng(12)
    corrupted = world.copy()
    corrupted[[3, 11]] += rng.normal(0, 40.0, (2, 3))
    cloud = np.zeros((1, 3))
    plain = georeference(cloud, R, source, corrupted, return_alignment=True)[1]
    robust = georeference(cloud, R, source, corrupted, robust=True, return_alignment=True)[1]

    assert plain.rmse > 100*robust.rmse
    assert np.linalg.norm(plain.translation - t) > 1.0
    assert np.isclose(robust.scale, scale, rtol=1e-6)
    assert np.allclose(robust.translation, t, atol=1e-8)
    assert sorted(np.where(~robust.inliers)[0]) == [3, 11]


def test_georeference_requires_control_points():
    from topoinv.sfm.reconstruct import georeference
    source, world, _, R, _ = _control()
    with pytest.raises(ValueError, match='control_points'):
        georeference(np.zeros((4, 3)), R, None, world)


def test_georeference_rejects_bad_point_shape():
    from topoinv.sfm.reconstruct import georeference
    source, world, _, R, _ = _control()
    with pytest.raises(ValueError, match='shape'):
        georeference(np.zeros((4, 2)), R, source, world)


def test_camera_centres_inverts_the_pose_convention():
    from topoinv.sfm.reconstruct import camera_centres
    rng = np.random.default_rng(6)
    axis = rng.normal(0, 0.3, 3)
    angle = np.linalg.norm(axis)
    k = axis/angle
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + np.sin(angle)*Kx + (1-np.cos(angle))*(Kx @ Kx)
    centre = np.array([5.0, 1.0, 2.0])
    poses = {0: (np.eye(3), np.zeros(3)), 2: (R, -R @ centre)}
    centres, keys = camera_centres(poses)
    assert keys == [0, 2]
    assert np.allclose(centres[0], 0.0)
    assert np.allclose(centres[1], centre)


def test_camera_centres_can_select_a_subset():
    from topoinv.sfm.reconstruct import camera_centres
    poses = {i: (np.eye(3), np.full(3, float(i))) for i in range(5)}
    centres, keys = camera_centres(poses, indices=[1, 3])
    assert keys == [1, 3]
    assert centres.shape == (2, 3)


def test_collinear_control_points_warn_when_rotation_is_unknown():
    line = np.stack([np.linspace(0, 10, 12), np.zeros(12), np.zeros(12)], 1)
    with pytest.warns(RuntimeWarning, match='collinear'):
        similarity_fit(line, 2*line + np.array([1.0, 2.0, 3.0]))


def test_collinear_control_points_are_fine_when_rotation_is_known():
    import warnings as w
    line = np.stack([np.linspace(0, 10, 12), np.zeros(12), np.zeros(12)], 1)
    with w.catch_warnings(record=True) as caught:
        w.simplefilter('always')
        fit = similarity_fit(line, 2*line + np.array([1.0, 2.0, 3.0]), rotation=np.eye(3))
    assert not [c for c in caught if 'collinear' in str(c.message)]
    assert np.isclose(fit.scale, 2.0)
