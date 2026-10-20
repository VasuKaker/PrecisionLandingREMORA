import cv2
import numpy as np
import pytest

from drone_landing_service.lightweight_landing_detector import (
    LANDMARK_DETECTED_BORDER_SCALE,
    LANDMARK_FOOTPRINT_MM,
    LightweightLandingDetector,
    _board_geometry,
)


EXPECTED_DETECTED_SPANS_M = {
    0: 0.00768,
    1: 0.0192,
    2: 0.048,
    3: 0.120,
}


def test_recursive_landmark_geometry_uses_detected_border_and_200mm_outline():
    standard, landmarks, outline = _board_geometry()

    assert standard == {}
    assert LANDMARK_FOOTPRINT_MM == {0: 12.8, 1: 32.0, 2: 80.0, 3: 200.0}
    assert LANDMARK_DETECTED_BORDER_SCALE == pytest.approx(0.6)
    assert set(landmarks) == {0, 1, 2, 3}
    np.testing.assert_allclose(
        outline,
        [[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]],
    )

    for marker_id, corners in landmarks.items():
        np.testing.assert_allclose(corners.mean(axis=0), [0.0, 0.0], atol=1e-9)
        horizontal = np.linalg.norm(corners[1] - corners[0])
        vertical = np.linalg.norm(corners[2] - corners[1])
        assert horizontal == pytest.approx(EXPECTED_DETECTED_SPANS_M[marker_id])
        assert vertical == pytest.approx(EXPECTED_DETECTED_SPANS_M[marker_id])


def _synthetic_detector():
    detector = LightweightLandingDetector.__new__(LightweightLandingDetector)
    detector.camera_matrix = np.array(
        [[718.4, 0.0, 640.0], [0.0, 718.4, 360.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    detector.distortion = np.zeros((4, 1), dtype=np.float64)
    detector.max_reprojection_error_px = 0.1
    detector.board, detector.landmark_board, detector.board_outline = _board_geometry()
    return detector


@pytest.mark.parametrize("marker_ids", [(0,), (1,), (2,), (3,), (0, 1, 2, 3)])
def test_synthetic_pose_from_each_recursive_tag_and_all_tags(marker_ids):
    detector = _synthetic_detector()
    rvec = np.array([[0.12], [-0.08], [0.03]], dtype=np.float64)
    tvec = np.array([[0.04], [-0.02], [0.75]], dtype=np.float64)
    landmarks = {}
    for marker_id in marker_ids:
        object_points = np.column_stack(
            (detector.landmark_board[marker_id], np.zeros(4))
        )
        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            detector.camera_matrix,
            detector.distortion,
        )
        landmarks[marker_id] = projected.reshape(-1, 2).astype(np.float32)

    detection = detector._pose_from_observations({}, landmarks, "synthetic")

    assert detection is not None
    assert detection.range_m == pytest.approx(0.75, abs=2e-4)
    assert detection.forward_m == pytest.approx(0.02, abs=2e-4)
    assert detection.left_m == pytest.approx(-0.04, abs=2e-4)
    expected_outline, _ = cv2.projectPoints(
        np.column_stack((detector.board_outline, np.zeros(4))),
        rvec,
        tvec,
        detector.camera_matrix,
        detector.distortion,
    )
    np.testing.assert_allclose(
        detection.board_corners_px,
        expected_outline.reshape(-1, 2),
        atol=0.1,
    )
