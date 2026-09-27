import numpy as np
import cv2


def camera_matrix(focal_length, principal_point=None, image_shape=None, pixel_size=1.0):
    fx = fy = float(focal_length) / float(pixel_size)
    if principal_point is None:
        if image_shape is None:
            raise ValueError("principal_point or image_shape must be provided")
        rows, cols = image_shape[:2]
        principal_point = (cols / 2.0, rows / 2.0)
    cx, cy = principal_point
    return np.array([[fx, 0.0, cx],
                     [0.0, fy, cy],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def calibrate_camera(object_points, image_points, image_shape):
    object_points = [np.asarray(p, dtype=np.float32) for p in object_points]
    image_points = [np.asarray(p, dtype=np.float32) for p in image_points]

    if len(object_points) != len(image_points):
        raise ValueError("object_points and image_points must have the same number of views")
    if len(object_points) == 0:
        raise ValueError("At least one calibration view is required")

    rows, cols = image_shape[:2]
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=object_points, imagePoints=image_points, imageSize=(cols, rows),
        cameraMatrix=None, distCoeffs=None, flags=cv2.CALIB_ZERO_TANGENT_DIST)
    return K, dist, rvecs, tvecs, rms


def as_uint8(image):
    image = np.asarray(image)
    if image.ndim == 3:
        image = cv2.cvtColor(image.astype(np.uint8) if image.dtype == np.uint8
                             else image.astype(np.float32), cv2.COLOR_BGR2GRAY)
    if image.dtype == np.uint8:
        return image
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape[:2], np.uint8)
    lo, hi = float(finite.min()), float(finite.max())
    span = hi - lo if hi > lo else 1.0
    return np.clip((np.nan_to_num(image) - lo) / span * 255.0, 0, 255).astype(np.uint8)


def view_pairs(n_images, match_pairs='sequential'):
    if match_pairs == 'sequential':
        return [(i, i + 1) for i in range(n_images - 1)]
    if match_pairs == 'exhaustive':
        return [(i, j) for i in range(n_images) for j in range(i + 1, n_images)]
    raise ValueError("match_pairs must be 'sequential' or 'exhaustive'")


def detect_and_match_features(images, detector='sift', ratio_test=0.75,
                              match_pairs='sequential', K=None, distortion=None):
    if detector == 'sift':
        engine = cv2.SIFT_create()
        norm = cv2.NORM_L2
    elif detector == 'orb':
        engine = cv2.ORB_create(nfeatures=5000)
        norm = cv2.NORM_HAMMING
    else:
        raise ValueError("Unsupported detector: " + repr(detector) + "; use 'sift' or 'orb'")

    keypoints, descriptors, points = [], [], []
    for image in images:
        kp, desc = engine.detectAndCompute(as_uint8(image), None)
        keypoints.append(kp)
        descriptors.append(desc)
        xy = np.array([k.pt for k in kp], dtype=np.float64).reshape(-1, 2)
        if distortion is not None and len(xy):
            xy = cv2.undistortPoints(xy.reshape(-1, 1, 2), K, distortion, P=K).reshape(-1, 2)
        points.append(xy)

    pairs = view_pairs(len(images), match_pairs)
    matcher = cv2.BFMatcher(norm, crossCheck=False)
    matches = []
    for i, j in pairs:
        di, dj = descriptors[i], descriptors[j]
        if di is None or dj is None or len(di) < 2 or len(dj) < 2:
            matches.append([])
            continue
        good = []
        for candidates in matcher.knnMatch(di, dj, k=2):
            if len(candidates) < 2:
                continue
            m, n = candidates
            if m.distance < ratio_test * n.distance:
                good.append(m)
        matches.append(good)
    return keypoints, points, pairs, matches


def build_tracks(pairs, matches, min_views=2):
    parent = {}

    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for (i, j), pair_matches in zip(pairs, matches):
        for m in pair_matches:
            union((i, m.queryIdx), (j, m.trainIdx))

    groups = {}
    for node in list(parent):
        groups.setdefault(find(node), []).append(node)

    tracks = []
    for members in groups.values():
        observations = {}
        conflicted = False
        for image_index, keypoint_index in members:
            if image_index in observations:
                conflicted = True
                break
            observations[image_index] = keypoint_index
        if not conflicted and len(observations) >= min_views:
            tracks.append(observations)
    return tracks


def triangulate_track(observations, projections):
    rows = []
    for image_index, (u, v) in observations.items():
        P = projections[image_index]
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.asarray(rows))
    X = Vt[-1]
    if abs(X[3]) < 1e-12:
        return None
    return X[:3] / X[3]


def reprojection_error(point, observations, projections):
    errors = []
    for image_index, (u, v) in observations.items():
        P = projections[image_index]
        x = P @ np.append(point, 1.0)
        if x[2] <= 1e-9:
            return np.inf
        errors.append(np.hypot(x[0]/x[2] - u, x[1]/x[2] - v))
    return float(np.mean(errors)) if errors else np.inf


def in_front_of(point, pose):
    R, t = pose
    return float((R @ point + t)[2]) > 0.0


def reconstruct(images, K, distortion=None, poses=None, detector='sift',
                ratio_test=0.75, min_matches=30, ransac_threshold=1.0,
                ransac_confidence=0.999, max_reprojection_error=4.0,
                min_track_views=2, match_pairs='sequential',
                dense=False, return_result=False):
    if len(images) < 2:
        raise ValueError("reconstruct needs at least two images")
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError("K must be a 3x3 camera matrix, got shape " + str(K.shape))
    if dense:
        raise NotImplementedError("dense reconstruction is not implemented")

    keypoints, points, pairs, matches = detect_and_match_features(
        images, detector=detector, ratio_test=ratio_test,
        match_pairs=match_pairs, K=K, distortion=distortion)

    tracks = build_tracks(pairs, matches, min_views=min_track_views)
    if not tracks:
        raise ValueError("no feature tracks survived matching; try a lower ratio_test "
                         "or a different detector")

    def observed(track):
        return {i: points[i][k] for i, k in track.items()}

    solved = {}
    if poses is not None:
        for i, pose in enumerate(poses):
            if pose is not None:
                solved[i] = (np.asarray(pose[0], np.float64),
                             np.asarray(pose[1], np.float64).reshape(3))
        if len(solved) < 2:
            raise ValueError("at least two known poses are required when poses is given")
    else:
        seed = None
        for (i, j), pair_matches in zip(pairs, matches):
            if len(pair_matches) >= min_matches:
                seed = (i, j, pair_matches)
                break
        if seed is None:
            best = max((len(m) for m in matches), default=0)
            raise ValueError("no image pair reached min_matches=" + str(min_matches) +
                             "; best pair had " + str(best))
        i, j, pair_matches = seed
        p1 = np.float64([points[i][m.queryIdx] for m in pair_matches])
        p2 = np.float64([points[j][m.trainIdx] for m in pair_matches])
        E, mask = cv2.findEssentialMat(p1, p2, K, method=cv2.RANSAC,
                                       prob=ransac_confidence, threshold=ransac_threshold)
        if E is None or E.shape != (3, 3):
            raise ValueError("essential matrix estimation failed; the views may be "
                             "degenerate (pure rotation or a planar scene)")
        _, R, t, _ = cv2.recoverPose(E, p1, p2, K, mask=mask.copy())
        solved[i] = (np.eye(3), np.zeros(3))
        solved[j] = (np.asarray(R, np.float64), np.asarray(t, np.float64).reshape(3))

    def projections():
        return {i: K @ np.hstack((R, t.reshape(3, 1))) for i, (R, t) in solved.items()}

    def triangulate():
        P = projections()
        found = {}
        for index, track in enumerate(tracks):
            visible = {i: xy for i, xy in observed(track).items() if i in solved}
            if len(visible) < 2:
                continue
            X = triangulate_track(visible, P)
            if X is None or not np.all(np.isfinite(X)):
                continue
            if not all(in_front_of(X, solved[i]) for i in visible):
                continue
            if reprojection_error(X, visible, P) > max_reprojection_error:
                continue
            found[index] = X
        return found

    cloud = triangulate()

    for image_index in range(len(images)):
        if image_index in solved:
            continue
        object_points, image_points = [], []
        for index, X in cloud.items():
            if image_index in tracks[index]:
                object_points.append(X)
                image_points.append(points[image_index][tracks[index][image_index]])
        if len(object_points) < 6:
            continue
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            np.float64(object_points).reshape(-1, 1, 3),
            np.float64(image_points).reshape(-1, 1, 2),
            K, None, reprojectionError=max(ransac_threshold, 2.0),
            confidence=ransac_confidence, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or inliers is None or len(inliers) < 6:
            continue
        solved[image_index] = (cv2.Rodrigues(rvec)[0], np.asarray(tvec, np.float64).reshape(3))
        cloud = triangulate()

    if not cloud:
        raise ValueError("no points could be triangulated; check K, the baseline, "
                         "and max_reprojection_error")

    track_ids = sorted(cloud)
    cloud_points = np.array([cloud[i] for i in track_ids], dtype=np.float64)

    if not return_result:
        return cloud_points

    P = projections()
    errors = np.array([
        reprojection_error(cloud[i],
                           {v: xy for v, xy in observed(tracks[i]).items() if v in solved},
                           P)
        for i in track_ids])
    result = {
        'poses': {i: solved[i] for i in sorted(solved)},
        'keypoints': keypoints,
        'points': points,
        'pairs': pairs,
        'matches': matches,
        'tracks': tracks,
        'track_ids': track_ids,
        'reprojection_error': errors,
        'registered': sorted(solved),
    }
    return cloud_points, result


def georeference(points, R, t, camera_positions=None, control_points=None,
                 similarity=None, crs=None):

    points = np.asarray(points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")

    if camera_positions is None and control_points is None:
        raise ValueError(
            "Provide camera_positions or control_points to georeference."
        )

    if camera_positions is not None:
        camera_positions = np.asarray(camera_positions, dtype=np.float64)

        if camera_positions.ndim != 2 or camera_positions.shape[1] != 3:
            raise ValueError(
                "camera_positions must have shape (N, 3)"
            )

    if control_points is not None:
        control_points = np.asarray(control_points, dtype=np.float64)

        if control_points.ndim != 2 or control_points.shape[1] != 3:
            raise ValueError(
                "control_points must have shape (N, 3)"
            )

    raise NotImplementedError


def points_to_dem(points, cellsize, bounds=None, aggregate='median',
                  min_points=1, fill_holes=False, crs=None):
    raise NotImplementedError


def sfm_to_dem(images, K, cellsize, distortion=None, poses=None,
               camera_positions=None, control_points=None, crs=None,
               bounds=None, aggregate='median', fill_holes=False,
               return_result=False, **reconstruct_kwargs):
    points, result = reconstruct(images, K, distortion=distortion, poses=poses,
                                 return_result=True, **reconstruct_kwargs)
    points = georeference(points, camera_positions=camera_positions,
                          control_points=control_points, crs=crs)
    dem = points_to_dem(points, cellsize, bounds=bounds, aggregate=aggregate,
                        fill_holes=fill_holes, crs=crs)
    return (dem, points, result) if return_result else dem
