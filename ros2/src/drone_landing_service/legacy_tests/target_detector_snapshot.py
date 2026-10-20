import cv2
import numpy as np

from drone_landing_service.target_detector import TargetDetector
from drone_landing_service.target_detector_deprecated import TargetResult


def make_detector():
    detector = TargetDetector.__new__(TargetDetector)
    detector._multiplex_geometric_tolerance = 0.35
    detector._multiplex_min_model_tags = 6
    detector._multiplex_methods = {}
    detector._grid_tag_ids = set(range(1, 10))
    detector._cam_w = 1280
    detector._cam_h = 720
    detector._frame_area = float(1280 * 720)
    detector._tag_board_corners = {}
    for marker_id in range(1, 10):
        row, col = divmod(marker_id - 1, 3)
        x0 = col * 2.0
        y0 = row * 2.0
        detector._tag_board_corners[marker_id] = np.array(
            [[x0, y0], [x0 + 1.0, y0], [x0 + 1.0, y0 + 1.0], [x0, y0 + 1.0]],
            dtype=np.float32,
        )
    return detector


def make_result(marker_id, corners, method):
    center = np.mean(corners, axis=0)
    return TargetResult(
        cx=int(round(float(center[0]))),
        cy=int(round(float(center[1]))),
        yaw=0.0,
        area_frac=float(abs(cv2.contourArea(corners)) / (1280 * 720)),
        corners=corners.astype(np.float32),
        source=f"tag_{marker_id}_{method}",
        target_id=marker_id,
        tag_ids=(marker_id,),
    )


def projected_results(detector, marker_ids, homography, method):
    return {
        marker_id: make_result(
            marker_id,
            cv2.perspectiveTransform(
                detector._tag_board_corners[marker_id].reshape(-1, 1, 2),
                homography,
            ).reshape(-1, 2),
            method,
        )
        for marker_id in marker_ids
    }


def test_merge_recovers_complementary_tags_and_rejects_bad_geometry():
    detector = make_detector()
    source = np.array(
        [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0]],
        dtype=np.float32,
    )
    destination = np.array(
        [[410.0, 130.0], [850.0, 145.0], [825.0, 580.0], [395.0, 555.0]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(source, destination)

    primary = projected_results(detector, range(1, 6), homography, "flat17_clahe")
    secondary = projected_results(detector, range(6, 10), homography, "adaptive81")
    secondary[3] = make_result(
        3,
        np.array(
            [[50.0, 50.0], [90.0, 50.0], [90.0, 90.0], [50.0, 90.0]],
            dtype=np.float32,
        ),
        "adaptive81",
    )

    merged = detector._merge_multiplex_passes(
        [
            ("flat17_clahe", primary),
            ("adaptive81", secondary),
        ]
    )

    assert sorted(merged) == list(range(1, 10))
    assert merged[3].source.endswith("flat17_clahe")

    bad_secondary = dict(secondary)
    bad_secondary[9] = make_result(
        9,
        np.array(
            [[20.0, 600.0], [60.0, 600.0], [60.0, 640.0], [20.0, 640.0]],
            dtype=np.float32,
        ),
        "adaptive81",
    )
    merged = detector._merge_multiplex_passes(
        [
            ("flat17_clahe", primary),
            ("adaptive81", bad_secondary),
        ]
    )
    assert 9 not in merged


def test_black_border_contrast_validation_rejects_plain_quad():
    detector = make_detector()
    detector._black_border_inches = 1.108
    detector._black_square_size_inches = 16.3
    detector._inner_padding_inches = 0.7386
    corners = np.array(
        [[0.0, 0.0], [239.0, 0.0], [239.0, 239.0], [0.0, 239.0]],
        dtype=np.float32,
    )

    target = np.full((240, 240), 225, dtype=np.uint8)
    target[:18, :] = 20
    target[-18:, :] = 20
    target[:, :18] = 20
    target[:, -18:] = 20
    target[100:140, :] = np.minimum(target[100:140, :] + 25, 255)

    assert detector._black_border_contrast_score(target, corners) is not None
    assert (
        detector._black_border_contrast_score(
            np.full((240, 240), 128, dtype=np.uint8),
            corners,
        )
        is None
    )


def test_board_model_completes_an_in_frame_missing_tag():
    detector = make_detector()
    source = np.array(
        [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0]],
        dtype=np.float32,
    )
    destination = np.array(
        [[410.0, 130.0], [850.0, 145.0], [825.0, 580.0], [395.0, 555.0]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(source, destination)
    direct = projected_results(detector, range(1, 9), homography, "flat17_clahe")

    completed = detector._complete_board_model_tags(direct)

    assert sorted(completed) == list(range(1, 10))
    assert completed[9].source == "tag_9_board_model"
    assert detector._multiplex_methods["board_model"] == (9,)
