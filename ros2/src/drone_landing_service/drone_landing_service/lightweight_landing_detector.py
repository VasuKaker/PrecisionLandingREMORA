#!/usr/bin/env python3
"""Standalone low-latency AprilTag-grid pose detector for offline trials.

This intentionally has no ROS, square fallback, image enhancement ladder, worker
threads, or legacy base class. It performs one grayscale AprilTag pass, tracks a
region of interest, fits the board homography, and solves the planar PnP pose.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np

from drone_landing_service.landmark_tag_detector import LandmarkTagDetector


STANDARD_LAYOUT_IN = {}
LANDMARK_FOOTPRINT_MM = {
    0: 12.8,
    1: 32.0,
    2: 80.0,
    3: 200.0,
}
# Landmark reports the coding-boundary corners, which span 60% of each
# recursive tag's full printed footprint.
LANDMARK_DETECTED_BORDER_SCALE = 0.6
TARGET_IDS = frozenset(STANDARD_LAYOUT_IN)


@dataclass(frozen=True)
class LandingDetection:
    ids: tuple[int, ...]
    center_px: tuple[float, float]
    forward_m: float
    left_m: float
    range_m: float
    yaw_rad: float
    reprojection_error_px: float
    board_corners_px: np.ndarray
    tag_corners_px: dict[int, np.ndarray]
    search: str


def _detector_parameters(quad_decimate: float):
    params = (
        cv2.aruco.DetectorParameters_create()
        if hasattr(cv2.aruco, "DetectorParameters_create")
        else cv2.aruco.DetectorParameters()
    )
    settings = {
        "minMarkerPerimeterRate": 0.01,
        "maxMarkerPerimeterRate": 4.0,
        "minDistanceToBorder": 0,
        "errorCorrectionRate": 1.0,
        "adaptiveThreshWinSizeMin": 3,
        "adaptiveThreshWinSizeMax": 31,
        "adaptiveThreshWinSizeStep": 4,
        "adaptiveThreshConstant": 7,
        "aprilTagQuadDecimate": quad_decimate,
        "aprilTagQuadSigma": 0.0,
        "aprilTagMinWhiteBlackDiff": 2,
        "aprilTagDeglitch": 1,
        "cornerRefinementMethod": getattr(cv2.aruco, "CORNER_REFINE_SUBPIX", 1),
    }
    for name, value in settings.items():
        if hasattr(params, name):
            setattr(params, name, value)
    return params


def _metric_layout(
    layout, corner_order: tuple[int, int, int, int], border_scale: float = 1.0
) -> dict[int, np.ndarray]:
    board = {}
    for marker_id, (top_left, top_right, bottom_left, bottom_right) in layout.items():
        corners = np.asarray(
            [top_left, top_right, bottom_right, bottom_left], dtype=np.float32
        )
        center = corners.mean(axis=0)
        corners = center + border_scale * (corners - center)
        corners = corners[list(corner_order)]
        corners[:, 1] *= -1.0  # Printed-map +y is up; image/PnP +y is down.
        board[marker_id] = corners * 0.0254
    return board


def _centered_landmark_board() -> dict[int, np.ndarray]:
    board = {}
    for marker_id, footprint_mm in LANDMARK_FOOTPRINT_MM.items():
        detected_half_m = footprint_mm * LANDMARK_DETECTED_BORDER_SCALE / 2000.0
        board[marker_id] = np.array(
            [
                [-detected_half_m, -detected_half_m],
                [detected_half_m, -detected_half_m],
                [detected_half_m, detected_half_m],
                [-detected_half_m, detected_half_m],
            ],
            dtype=np.float32,
        )
    return board


def _board_geometry() -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], np.ndarray]:
    half = 0.200 / 2.0
    outline = np.array(
        [[-half, -half], [half, -half], [half, half], [-half, half]],
        dtype=np.float32,
    )
    return (
        _metric_layout(STANDARD_LAYOUT_IN, (2, 3, 0, 1)),
        _centered_landmark_board(),
        outline,
    )


class LightweightLandingDetector:
    """One-pass detector with reduced-resolution search and full-frame outputs."""

    def __init__(
        self,
        width: int,
        height: int,
        focal_px: float = 737.12,
        max_search_dimension: int = 960,
        roi_padding_px: int = 96,
        roi_padding_tag_widths: float = 4.0,
        max_reprojection_error_px: float = 12.0,
        use_roi: bool = True,
        quad_decimate: float = 1.0,
        white_threshold_on_miss: bool = True,
        temporary_white_pose_retry: bool = True,
    ) -> None:
        self.width, self.height = width, height
        self.max_search_dimension = max_search_dimension
        self.roi_padding_px = roi_padding_px
        self.roi_padding_tag_widths = roi_padding_tag_widths
        self.max_reprojection_error_px = max_reprojection_error_px
        self.use_roi = use_roi
        self.white_threshold_on_miss = white_threshold_on_miss
        # Temporary close-range recovery: retry crisp thresholded corners only
        # when decoded grayscale tags fail the pose-quality gate.
        self.temporary_white_pose_retry = temporary_white_pose_retry
        self.roi: tuple[int, int, int, int] | None = None
        self.last_tags: dict[int, np.ndarray] = {}
        self.last_landmark_ids: tuple[int, ...] = ()
        self.last_search = ""
        self.camera_matrix = np.array(
            [[focal_px, 0.0, width / 2.0], [0.0, focal_px, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.distortion = np.zeros((4, 1), dtype=np.float64)
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
        self.parameters = _detector_parameters(quad_decimate)
        detector_type = getattr(cv2.aruco, "ArucoDetector", None)
        self.detector = detector_type(self.dictionary, self.parameters) if detector_type else None
        self.landmark_detector = LandmarkTagDetector(quad_decimate)
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.parallel_detectors = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="landing-family"
        )
        self.board, self.landmark_board, self.board_outline = _board_geometry()

    def _detect_markers(self, gray: np.ndarray):
        if self.detector is not None:
            return self.detector.detectMarkers(gray)
        return cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)

    def _find_tags(
        self, frame: np.ndarray, force_white: bool = False
    ) -> tuple[dict[int, np.ndarray], str]:
        if self.use_roi and self.roi is not None:
            x0, y0, x1, y1 = self.roi
            crop, search = frame[y0:y1, x0:x1], "roi"
        else:
            x0 = y0 = 0
            crop, search = frame, "full"
        gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        scale = min(1.0, self.max_search_dimension / max(gray.shape))
        work = gray if scale == 1.0 else cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
        if force_white:
            white = (work >= 195).astype(np.uint8) * 255
            corners, ids, _ = self._detect_markers(white)
            search += "+white195"
        else:
            corners, ids, _ = self._detect_markers(work)
            if ids is None and self.white_threshold_on_miss:
                white = (work >= 195).astype(np.uint8) * 255
                corners, ids, _ = self._detect_markers(white)
                search += "+white195"
        tags: dict[int, np.ndarray] = {}
        if ids is not None:
            offset = np.array([x0, y0], dtype=np.float32)
            for index, raw_id in enumerate(ids.flatten()):
                marker_id = int(raw_id)
                if marker_id not in TARGET_IDS:
                    continue
                points = corners[index][0].astype(np.float32) / scale + offset
                previous = tags.get(marker_id)
                if previous is None or abs(cv2.contourArea(points)) > abs(cv2.contourArea(previous)):
                    tags[marker_id] = points
        return tags, f"{search}_{max(work.shape)}px"

    def _next_roi(self, tags: dict[int, np.ndarray]) -> tuple[int, int, int, int]:
        points = np.concatenate(list(tags.values()))
        tag_width = max(
            max(np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[2] - p[1]))
            for p in tags.values()
        )
        padding = max(self.roi_padding_px, int(round(tag_width * self.roi_padding_tag_widths)))
        x0 = max(0, int(np.floor(points[:, 0].min())) - padding)
        y0 = max(0, int(np.floor(points[:, 1].min())) - padding)
        x1 = min(self.width, int(np.ceil(points[:, 0].max())) + padding + 1)
        y1 = min(self.height, int(np.ceil(points[:, 1].max())) + padding + 1)
        return x0, y0, x1, y1

    def _find_landmarks(
        self, frame: np.ndarray
    ) -> tuple[dict[int, np.ndarray], str]:
        if not self.landmark_detector.available:
            self.last_landmark_ids = ()
            return {}, "landmark-unavailable"
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale = min(1.0, self.max_search_dimension / max(gray.shape))
        work = gray if scale == 1.0 else cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
        detections = self.landmark_detector.detect(work)
        landmark_method = "raw"
        if not detections:
            detections = self.landmark_detector.detect(
                (work >= 195).astype(np.uint8) * 255
            )
            landmark_method = "white195"
        if not detections:
            detections = self.landmark_detector.detect(self.clahe.apply(work))
            landmark_method = "clahe"
        self.last_landmark_ids = tuple(sorted(item.marker_id for item in detections))
        landmarks = {}
        for item in detections:
            if item.marker_id not in self.landmark_board:
                continue
            # Native wrapper order is LB,RB,RT,LT; normalize to TL,TR,BR,BL.
            points = item.corners[[3, 2, 1, 0]].astype(np.float32) / scale
            previous = landmarks.get(item.marker_id)
            if previous is None or abs(cv2.contourArea(points)) > abs(cv2.contourArea(previous)):
                landmarks[item.marker_id] = points
        return landmarks, f"landmark:{landmark_method}_{max(work.shape)}px"

    def _solve_pose(self, board: np.ndarray, image: np.ndarray):
        objects = np.column_stack((board.astype(np.float64), np.zeros(len(board))))
        pixels = image.astype(np.float64)
        candidates: list[tuple[float, np.ndarray]] = []
        try:
            ok, rvecs, tvecs, errors = cv2.solvePnPGeneric(
                objects, pixels, self.camera_matrix, self.distortion, flags=cv2.SOLVEPNP_IPPE
            )
            if ok:
                for rvec, tvec, error in zip(rvecs, tvecs, errors):
                    tvec = np.asarray(tvec).reshape(3)
                    if np.all(np.isfinite(tvec)) and tvec[2] > 0.0:
                        candidates.append((float(np.asarray(error).reshape(-1)[0]), tvec))
        except cv2.error:
            pass
        if not candidates:
            try:
                ok, rvec, tvec = cv2.solvePnP(
                    objects, pixels, self.camera_matrix, self.distortion,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if ok:
                    projected, _ = cv2.projectPoints(
                        objects, rvec, tvec, self.camera_matrix, self.distortion
                    )
                    error = float(np.mean(np.linalg.norm(projected.reshape(-1, 2) - pixels, axis=1)))
                    candidates.append((error, np.asarray(tvec).reshape(3)))
            except cv2.error:
                pass
        return min(candidates, key=lambda item: item[0]) if candidates else None

    def _pose_from_observations(
        self,
        tags: dict[int, np.ndarray],
        landmarks: dict[int, np.ndarray],
        search: str,
    ) -> LandingDetection | None:
        parts = [
            (self.board[tag_id], tags[tag_id]) for tag_id in sorted(tags)
        ] + [
            (self.landmark_board[tag_id], landmarks[tag_id])
            for tag_id in sorted(landmarks)
        ]
        if not parts:
            return None
        board = np.concatenate([part[0] for part in parts])
        image = np.concatenate([part[1] for part in parts])
        homography, mask = cv2.findHomography(
            board, image, 0 if len(parts) == 1 else cv2.RANSAC, 3.0
        )
        if homography is None:
            return None
        pose_board, pose_image = board, image
        if mask is not None and len(parts) > 1:
            keep = mask.reshape(-1).astype(bool)
            if np.count_nonzero(keep) >= 4:
                pose_board, pose_image = board[keep], image[keep]
        pose = self._solve_pose(pose_board, pose_image)
        if pose is None or pose[0] > self.max_reprojection_error_px:
            return None
        error, tvec = pose
        query = np.array([[[0.0, 0.0]], [[0.10, 0.0]]], dtype=np.float32)
        projected = cv2.perspectiveTransform(query, homography).reshape(-1, 2)
        outline = cv2.perspectiveTransform(
            self.board_outline.reshape(-1, 1, 2), homography
        ).reshape(-1, 2)
        if not np.all(np.isfinite(outline)):
            return None
        yaw = math.atan2(*(projected[1] - projected[0])[::-1])
        return LandingDetection(
            ids=tuple(sorted(tags)),
            center_px=(float(projected[0, 0]), float(projected[0, 1])),
            forward_m=float(-tvec[1]),
            left_m=float(-tvec[0]),
            range_m=float(tvec[2]),
            yaw_rad=float(yaw),
            reprojection_error_px=float(error),
            board_corners_px=outline,
            tag_corners_px=tags,
            search=search,
        )

    def detect(self, frame: np.ndarray) -> LandingDetection | None:
        standard_future = self.parallel_detectors.submit(self._find_tags, frame)
        landmark_future = self.parallel_detectors.submit(
            self._find_landmarks, frame
        )
        tags, search = standard_future.result()
        landmarks, landmark_search = landmark_future.result()
        combined_search = f"{search}|{landmark_search}"
        self.last_tags, self.last_search = tags, combined_search
        if not tags and not landmarks:
            self.roi = None
            return None
        result = self._pose_from_observations(tags, landmarks, combined_search)
        roi_tags = tags
        if result is None and self.temporary_white_pose_retry and "+white195" not in search:
            retry_tags, retry_search = self._find_tags(frame, force_white=True)
            retry_result = (
                self._pose_from_observations(
                    retry_tags, landmarks, f"{retry_search}|{landmark_search}"
                )
                if retry_tags or landmarks else None
            )
            if retry_result is not None:
                result, roi_tags = retry_result, retry_tags
                self.last_tags = retry_tags
                self.last_search = retry_result.search
        self.roi = self._next_roi(roi_tags) if self.use_roi and roi_tags else None
        return result

    def close(self) -> None:
        self.parallel_detectors.shutdown(wait=True, cancel_futures=True)
