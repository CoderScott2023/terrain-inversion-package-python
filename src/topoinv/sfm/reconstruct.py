import numpy as np
import cv2


import numpy as np
import cv2


def detect_and_match_features(images, detector='sift', ratio_test=0.75):
    if detector == 'sift':
        feature_detector = cv2.SIFT_create()
    elif detector == 'orb':
        feature_detector = cv2.ORB_create()
    else:
        raise ValueError(f"Unsupported detector: {detector}")

    keypoints_list = []
    descriptors_list = []

    for image in images:
        keypoints, descriptors = feature_detector.detectAndCompute(image, None)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)

    bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches_list = []

    for i in range(len(images) - 1):
        matches = bf_matcher.knnMatch(descriptors_list[i], descriptors_list[i + 1], k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_test * n.distance:
                good_matches.append(m)
        matches_list.append(good_matches)

    return keypoints_list, matches_list

def calibrate_camera(object_points, image_points, image_shape):
    object_points = [
        np.asarray(points, dtype=np.float64)
        for points in object_points
    ]

    image_points = [
        np.asarray(points, dtype=np.float64)
        for points in image_points
    ]

    if len(object_points) != len(image_points):
        raise ValueError(
            "object_points and image_points must have the same "
            "number of views"
        )

    if len(object_points) == 0:
        raise ValueError("At least one calibration view is required")

    rows, cols = image_shape[:2]

    rms, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=object_points,
        imagePoints=image_points,
        imageSize=(cols, rows),
        cameraMatrix=None,
        distCoeffs=None,
        flags=cv2.CALIB_ZERO_TANGENT_DIST
    )

    return K, dist_coeffs, rvecs, tvecs, rms


def reconstruct(images, K, intrinsics, distortion=None, poses=None,
                detector='sift', ratio_test=0.75, min_matches=30,
                ransac_threshold=1.0, ransac_confidence=0.999,
                dense=False, return_result=False):

    keypoints, matches = detect_and_match_features(
        images, detector=detector, ratio_test=ratio_test
    )

    def build_tracks(keypoints, matches):
        tracks = []
        for i, match in enumerate(matches):
            for m in match:
                track = [None] * len(images)
                track[i] = keypoints[i][m.queryIdx].pt
                track[i + 1] = keypoints[i + 1][m.trainIdx].pt
                tracks.append(track)
        return tracks

    tracks = build_tracks(keypoints, matches)

    def find_essential_matrix(points1, points2, K):
        E, mask = cv2.findEssentialMat(
            points1, points2, K, method=cv2.RANSAC,
            prob=ransac_confidence, threshold=ransac_threshold
        )
        return E, mask

    point_cloud = []

    for i in range(len(images) - 1):

        points1 = []
        points2 = []

        # Collect ALL correspondences between image i and image i+1
        for track in tracks:
            if track[i] is not None and track[i + 1] is not None:
                points1.append(track[i])
                points2.append(track[i + 1])

        if len(points1) < min_matches:
            continue

        points1 = np.asarray(points1, dtype=np.float32)
        points2 = np.asarray(points2, dtype=np.float32)

        E, mask = find_essential_matrix(points1, points2, K)

        _, R, t, pose_mask = cv2.recoverPose(
            E,
            points1,
            points2,
            K,
            mask=mask
        )

        R0 = np.eye(3)
        t0 = np.zeros((3, 1))

        R1 = R
        t1 = t

        P0 = K @ np.hstack((R0, t0))
        P1 = K @ np.hstack((R1, t1))

        points_4d = cv2.triangulatePoints(
            P0,
            P1,
            points1.T,
            points2.T
        )

        points_3d = points_4d[:3] / points_4d[3]

        point_cloud.append(points_3d.T)

    point_cloud = np.vstack(point_cloud)
    

    return point_cloud if not return_result else (point_cloud, keypoints, matches)


def georeference(points, camera_positions=None, control_points=None,
                 similarity=None, crs=None):
    raise NotImplementedError


def points_to_dem(points, cellsize, bounds=None, aggregate='median',
                  min_points=1, fill_holes=False, crs=None):
    raise NotImplementedError


def sfm_to_dem(images, intrinsics, cellsize, distortion=None, poses=None,
               camera_positions=None, control_points=None, crs=None,
               bounds=None, aggregate='median', fill_holes=False,
               return_result=False, **reconstruct_kwargs):
    points, solved_poses = reconstruct(
        images, intrinsics, distortion=distortion, poses=poses,
        return_result=True, **reconstruct_kwargs)

    points = georeference(points, camera_positions=camera_positions,
                          control_points=control_points, crs=crs)

    dem = points_to_dem(points, cellsize, bounds=bounds, aggregate=aggregate,
                        fill_holes=fill_holes, crs=crs)

    return (dem, points, solved_poses) if return_result else dem
