#!/usr/bin/env python3
"""Experimental black-square detector for the landing target.

This node intentionally runs beside target_detector.py. It does not publish to
the production landing topics. Its job is to make the black outer square/frame
easy to test, tune, and compare against AprilTag pose.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from ament_index_python.packages import get_package_share_directory
except ModuleNotFoundError:
    get_package_share_directory = None

try:
    import rclpy
    from cv_bridge import CvBridge
    from geometry_msgs.msg import PointStamped
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image
    from std_msgs.msg import Bool, Float32, Float32MultiArray, String
except ModuleNotFoundError:
    rclpy = None
    CvBridge = None
    PointStamped = None
    QoSProfile = None
    ReliabilityPolicy = None
    Image = None
    Bool = Float32 = Float32MultiArray = String = None

    class Node:  # type: ignore[no-redef]
        pass


@dataclass
class SquareDetection:
    cx: float
    cy: float
    yaw: float
    area_frac: float
    corners: np.ndarray
    metric_forward_m: float
    metric_left_m: float
    range_m: float
    score: float
    method: str
    outer_board_corners: np.ndarray


def order_corners(corners: np.ndarray) -> np.ndarray:
    pts = corners.reshape(-1, 2).astype(np.float32)
    sums = pts.sum(axis=1)
    diffs = pts[:, 0] - pts[:, 1]
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmax(diffs)]
    ordered[3] = pts[np.argmin(diffs)]
    return ordered


def corner_angles_deg(quad: np.ndarray) -> list[float]:
    pts = quad.reshape(4, 2).astype(np.float32)
    angles = []
    for index in range(4):
        prev_pt = pts[(index - 1) % 4]
        pt = pts[index]
        next_pt = pts[(index + 1) % 4]
        v1 = prev_pt - pt
        v2 = next_pt - pt
        norm = float(np.linalg.norm(v1) * np.linalg.norm(v2))
        if norm <= 1e-6:
            return []
        cos_theta = float(np.dot(v1, v2) / norm)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        angles.append(float(np.degrees(np.arccos(cos_theta))))
    return angles


class BlackSquareDetector:
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        focal_px: float = 867.0,
        black_square_size_inches: float = 16.25,
        outer_board_size_inches: float = 18.0,
        target_size_inches: Optional[float] = None,
        outer_white_inches: Optional[float] = None,
        gray_threshold: int = 142,
        max_black_saturation: int = 130,
        min_area_frac: float = 0.012,
        max_area_frac: float = 0.98,
        min_angle_deg: float = 70.0,
        max_angle_deg: float = 110.0,
        enable_template_match: bool = True,
    ):
        self.width = int(width)
        self.height = int(height)
        self.frame_area = float(max(1, self.width * self.height))
        self.focal_px = float(focal_px)
        self.gray_threshold = int(gray_threshold)
        self.max_black_saturation = int(max_black_saturation)
        self.min_area_frac = float(min_area_frac)
        self.max_area_frac = float(max_area_frac)
        self.min_angle_deg = float(min_angle_deg)
        self.max_angle_deg = float(max_angle_deg)
        self._template_match_min_score = 0.39

        if target_size_inches is not None and outer_white_inches is not None:
            black_square_size_inches = float(target_size_inches) - 2.0 * float(outer_white_inches)
            outer_board_size_inches = float(target_size_inches)

        self.black_square_size_m = float(black_square_size_inches) * 0.0254
        self.outer_board_size_m = float(outer_board_size_inches) * 0.0254
        self.camera_matrix = np.array(
            [
                [self.focal_px, 0.0, self.width / 2.0],
                [0.0, self.focal_px, self.height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        half = self.black_square_size_m / 2.0
        self.black_square_board_corners = np.array(
            [
                [-half, -half],
                [half, -half],
                [half, half],
                [-half, half],
            ],
            dtype=np.float32,
        )
        outer_half = self.outer_board_size_m / 2.0
        self.outer_board_corners_m = np.array(
            [
                [-outer_half, -outer_half],
                [outer_half, -outer_half],
                [outer_half, outer_half],
                [-outer_half, outer_half],
            ],
            dtype=np.float32,
        )
        self._template_bank = self._build_template_bank() if enable_template_match else []

    def detect(self, image: np.ndarray) -> Optional[SquareDetection]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        best = self._detect_by_template(image)
        masks = list(self._black_masks(gray, hsv))
        for method, mask in masks:
            candidate = self._detect_from_mask(mask, method, gray)
            if candidate is not None and self._is_better_candidate(candidate, best):
                best = candidate
            cluster_candidate = self._detect_from_closed_cluster(mask, f"{method}_cluster", gray)
            if cluster_candidate is not None and (
                self._is_better_candidate(cluster_candidate, best)
            ):
                best = cluster_candidate
        ring = self._detect_by_black_ring(gray)
        if ring is not None and self._is_better_candidate(ring, best):
            best = ring
        return best

    @staticmethod
    def _is_better_candidate(
        candidate: SquareDetection,
        current: Optional[SquareDetection],
    ) -> bool:
        if current is None:
            return True
        candidate_is_tiny_template = (
            candidate.method == "template_board" and candidate.area_frac < 0.04
        )
        current_is_tiny_template = (
            current.method == "template_board" and current.area_frac < 0.04
        )
        # Template matching can lock onto a single AprilTag-like subpattern at
        # the wrong scale. If another validated square candidate is much larger,
        # prefer the larger physical board over the sharper but tiny template hit.
        if candidate.area_frac > 2.2 * current.area_frac and candidate.score >= 3.5:
            return True
        if current.area_frac > 2.2 * candidate.area_frac and current.score >= 3.5:
            return False
        if current_is_tiny_template and candidate.area_frac > max(0.10, 4.0 * current.area_frac):
            return True
        if candidate_is_tiny_template and current.area_frac > max(0.10, 4.0 * candidate.area_frac):
            return False
        return candidate.score > current.score

    def _build_template_bank(self):
        template = self._load_template_image()
        if template is None:
            return []
        bank = []
        for size_px in range(16, 91, 2):
            for angle_deg in range(-35, 36, 5):
                rotated, corners = self._rotated_template(template, size_px, angle_deg)
                if rotated is None:
                    continue
                height, width = rotated.shape[:2]
                if height > 120 or width > 160:
                    continue
                bank.append(
                    {
                        "size_px": size_px,
                        "angle_deg": angle_deg,
                        "image": self._template_preprocess(rotated),
                        "corners": corners.astype(np.float32),
                    }
                )
        return bank

    @staticmethod
    def _load_template_image() -> Optional[np.ndarray]:
        candidates = [
            Path.cwd() / "apriltag_grid_target" / "landing_apriltag_grid.png",
        ]
        if get_package_share_directory is not None:
            try:
                candidates.append(
                    Path(get_package_share_directory("drone_landing_service"))
                    / "assets"
                    / "landing_apriltag_grid.png"
                )
            except Exception:
                pass
        here = Path(__file__).resolve()
        candidates.append(here.parent.parent / "assets" / "landing_apriltag_grid.png")
        candidates.extend(
            parent / "apriltag_grid_target" / "landing_apriltag_grid.png"
            for parent in here.parents
        )
        candidates.extend(
            parent / "assets" / "landing_apriltag_grid.png"
            for parent in here.parents
        )
        for path in candidates:
            if path.exists():
                image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if image is not None and image.size > 0:
                    return image
        return None

    @staticmethod
    def _rotated_template(
        template: np.ndarray,
        size_px: int,
        angle_deg: float,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        resized = cv2.resize(
            template,
            (size_px, size_px),
            interpolation=cv2.INTER_AREA if size_px < template.shape[0] else cv2.INTER_CUBIC,
        )
        center = (size_px / 2.0, size_px / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        cos_theta = abs(float(matrix[0, 0]))
        sin_theta = abs(float(matrix[0, 1]))
        width = max(1, int(size_px * cos_theta + size_px * sin_theta))
        height = max(1, int(size_px * sin_theta + size_px * cos_theta))
        matrix[0, 2] += width / 2.0 - center[0]
        matrix[1, 2] += height / 2.0 - center[1]
        rotated = cv2.warpAffine(
            resized,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderValue=255,
        )
        corners = np.array(
            [[0, 0], [size_px - 1, 0], [size_px - 1, size_px - 1], [0, size_px - 1]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        corners = cv2.transform(corners, matrix).reshape(-1, 2)
        return rotated, corners

    @staticmethod
    def _template_preprocess(gray: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=3, sigmaY=3)
        return cv2.addWeighted(gray, 1.6, blurred, -0.6, 128)

    def _detect_by_template(self, image: np.ndarray) -> Optional[SquareDetection]:
        if not self._template_bank:
            return None
        small_width = 160
        small_height = 120
        small = cv2.resize(image, (small_width, small_height), interpolation=cv2.INTER_AREA)
        source = self._template_preprocess(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))

        best = None
        for item in self._template_bank:
            template = item["image"]
            template_h, template_w = template.shape[:2]
            if template_h > source.shape[0] or template_w > source.shape[1]:
                continue
            response = cv2.matchTemplate(source, template, cv2.TM_CCOEFF_NORMED)
            _, max_value, _, max_location = cv2.minMaxLoc(response)
            if not np.isfinite(max_value):
                continue
            if best is None or max_value > best[0]:
                best = (float(max_value), max_location, item)
        if best is None or best[0] < self._template_match_min_score:
            return None

        match_score, top_left, item = best
        scale_x = float(self.width) / small_width
        scale_y = float(self.height) / small_height
        outer_quad = item["corners"] + np.array(top_left, dtype=np.float32)
        outer_quad[:, 0] *= scale_x
        outer_quad[:, 1] *= scale_y
        outer_quad = order_corners(outer_quad)

        center = np.mean(outer_quad, axis=0)
        black_ratio = self.black_square_size_m / max(self.outer_board_size_m, 1e-6)
        black_quad = center + (outer_quad - center) * black_ratio
        black_quad = order_corners(black_quad)

        layout_score = self._outer_board_layout_score(image, outer_quad)
        if layout_score is None:
            return None
        angles = corner_angles_deg(black_quad)
        if len(angles) != 4:
            return None
        area_frac = float(abs(cv2.contourArea(black_quad)) / self.frame_area)
        detection = self._detection_from_quad(
            black_quad,
            area_frac,
            pattern_score=layout_score + match_score,
            aspect=1.0,
            angles=angles,
            method="template_board",
        )
        if detection is not None:
            detection.score += 2.0 * match_score
        return detection

    def _outer_board_layout_score(
        self,
        image: np.ndarray,
        outer_quad: np.ndarray,
    ) -> Optional[float]:
        warp_size = 240
        dst = np.array(
            [[0, 0], [warp_size - 1, 0], [warp_size - 1, warp_size - 1], [0, warp_size - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(outer_quad.astype(np.float32), dst)
        warped = cv2.warpPerspective(image, matrix, (warp_size, warp_size))
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        if float(np.std(gray)) < 24.0:
            return None
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        threshold, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = max(75.0, min(170.0, float(threshold)))
        black = blur < threshold
        white = (gray > 120) & (hsv[:, :, 1] < 145)

        margin = max(4, int(round((self.outer_board_size_m - self.black_square_size_m) / 2.0 / self.outer_board_size_m * warp_size)))
        ring_outer = max(6, margin)
        estimated_border = max(8, int(round(0.055 * warp_size)))
        ring_inner = min(warp_size // 2 - 6, ring_outer + estimated_border + 5)

        frame_bands = [
            black[ring_outer:ring_inner, ring_outer : warp_size - ring_outer],
            black[warp_size - ring_inner : warp_size - ring_outer, ring_outer : warp_size - ring_outer],
            black[ring_outer : warp_size - ring_outer, ring_outer:ring_inner],
            black[ring_outer : warp_size - ring_outer, warp_size - ring_inner : warp_size - ring_outer],
        ]
        edge_values = [float(np.mean(region)) for region in frame_bands if region.size > 0]
        if len(edge_values) != 4:
            return None
        edge_min = min(edge_values)
        edge_mean = float(np.mean(edge_values))
        if edge_min < 0.09 or edge_mean < 0.28:
            return None

        white_margin = max(3, ring_outer - 2)
        margin_bands = [
            white[:white_margin, :],
            white[-white_margin:, :],
            white[:, :white_margin],
            white[:, -white_margin:],
        ]
        margin_values = [float(np.mean(region)) for region in margin_bands if region.size > 0]
        if len(margin_values) != 4:
            return None
        white_margin_mean = float(np.mean(margin_values))
        if white_margin_mean < 0.14:
            return None

        inner_start = min(warp_size // 2 - 8, ring_inner + 8)
        inner_end = max(warp_size // 2 + 8, warp_size - ring_inner - 8)
        if inner_end <= inner_start:
            return None
        inner = black[inner_start:inner_end, inner_start:inner_end]
        inner_black = float(np.mean(inner))
        if not 0.05 <= inner_black <= 0.76:
            return None

        cell_scores = []
        for row in range(3):
            for col in range(3):
                y0 = int(inner_start + (inner_end - inner_start) * (row + 0.10) / 3.0)
                y1 = int(inner_start + (inner_end - inner_start) * (row + 0.90) / 3.0)
                x0 = int(inner_start + (inner_end - inner_start) * (col + 0.10) / 3.0)
                x1 = int(inner_start + (inner_end - inner_start) * (col + 0.90) / 3.0)
                cell = black[y0:y1, x0:x1]
                if cell.size > 0:
                    cell_scores.append(float(np.mean(cell)))
        cells_present = sum(score > 0.08 for score in cell_scores)
        if cells_present < 5:
            return None

        corridors = []
        for fraction in (1.0 / 3.0, 2.0 / 3.0):
            x = int(inner_start + (inner_end - inner_start) * fraction)
            y = int(inner_start + (inner_end - inner_start) * fraction)
            corridors.append(float(np.mean(black[inner_start:inner_end, max(inner_start, x - 5) : min(inner_end, x + 5)])))
            corridors.append(float(np.mean(black[max(inner_start, y - 5) : min(inner_end, y + 5), inner_start:inner_end])))
        corridor_black = float(np.mean(corridors)) if corridors else 1.0
        if corridor_black > 0.65:
            return None

        return (
            1.5 * edge_mean
            + 0.8 * white_margin_mean
            + 0.12 * min(cells_present, 9)
            + 0.5 * (1.0 - min(corridor_black, 1.0))
            + 0.4 * (1.0 - min(abs(inner_black - 0.34), 0.34) / 0.34)
        )

    def _detect_by_black_ring(self, gray: np.ndarray) -> Optional[SquareDetection]:
        binary_white = (gray >= self.gray_threshold).astype(np.uint8) * 255
        raw_black_mask = cv2.bitwise_not(binary_white)
        black_mask = self._clean_mask(raw_black_mask, close=9, open_=3)
        contours, hierarchy = cv2.findContours(black_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            return None
        hierarchy = hierarchy[0]
        best = None
        for contour_index, contour in enumerate(contours):
            candidate = self._candidate_from_ring_contour(
                contour,
                contour_index,
                hierarchy,
                raw_black_mask,
                gray,
            )
            if candidate is not None and (best is None or candidate.score > best.score):
                best = candidate
        return best

    @staticmethod
    def _contour_depth(hierarchy: np.ndarray, contour_index: int) -> int:
        depth = 0
        parent = int(hierarchy[contour_index][3])
        while parent >= 0:
            depth += 1
            parent = int(hierarchy[parent][3])
        return depth

    def _candidate_from_ring_contour(
        self,
        contour: np.ndarray,
        contour_index: int,
        hierarchy: np.ndarray,
        score_mask: np.ndarray,
        gray: np.ndarray,
    ) -> Optional[SquareDetection]:
        contour_area = float(abs(cv2.contourArea(contour)))
        area_frac = contour_area / self.frame_area
        if area_frac < self.min_area_frac or area_frac > self.max_area_frac:
            return None
        rect = cv2.minAreaRect(contour)
        quad = order_corners(cv2.boxPoints(rect))
        x, y, box_w, box_h = cv2.boundingRect(quad.astype(np.int32))
        if box_w < 35 or box_h < 35:
            return None
        aspect = box_w / max(1.0, float(box_h))
        if not 0.45 <= aspect <= 2.2:
            return None
        angles = corner_angles_deg(quad)
        if len(angles) != 4:
            return None
        if any(angle < self.min_angle_deg or angle > self.max_angle_deg for angle in angles):
            return None
        if self._quad_gray_std(gray, quad) < 24.0:
            return None
        pattern_score = self._black_frame_pattern_score(score_mask, quad)
        detection_method = "ring_rect_pattern"
        if pattern_score is None:
            pattern_score = self._black_frame_grid_score(score_mask, quad)
            detection_method = "ring_rect_grid"
        if pattern_score is None:
            return None

        detection = self._detection_from_quad(quad, area_frac, pattern_score, aspect, angles, detection_method)
        return detection

    def _black_masks(self, gray: np.ndarray, hsv: np.ndarray):
        dark = (gray < self.gray_threshold).astype(np.uint8) * 255
        yield "gray_dark", self._clean_mask(dark, close=5, open_=3)

        low_sat_dark = (
            ((gray < self.gray_threshold) & (hsv[:, :, 1] < self.max_black_saturation))
            .astype(np.uint8)
            * 255
        )
        yield "gray_dark_lowsat", self._clean_mask(low_sat_dark, close=5, open_=3)

        # Local contrast helps when carpet/shadow shifts the absolute black level.
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=17, sigmaY=17)
        flat = cv2.divide(gray, blurred, scale=128)
        flat_dark = (flat < 118).astype(np.uint8) * 255
        yield "flat_dark", self._clean_mask(flat_dark, close=5, open_=3)

    @staticmethod
    def _clean_mask(mask: np.ndarray, close: int, open_: int) -> np.ndarray:
        cleaned = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (close, close)),
            iterations=1,
        )
        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (open_, open_)),
            iterations=1,
        )
        return cleaned

    def _detect_from_mask(
        self,
        black_mask: np.ndarray,
        method: str,
        gray: Optional[np.ndarray] = None,
    ) -> Optional[SquareDetection]:
        contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for contour in contours:
            candidate = self._candidate_from_contour(contour, black_mask, method, gray)
            if candidate is not None and (best is None or candidate.score > best.score):
                best = candidate
        return best

    def _detect_from_closed_cluster(
        self,
        black_mask: np.ndarray,
        method: str,
        gray: Optional[np.ndarray] = None,
    ) -> Optional[SquareDetection]:
        kernel = max(15, int(round(min(self.width, self.height) * 0.055)))
        if kernel % 2 == 0:
            kernel += 1
        clustered = cv2.morphologyEx(
            black_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel)),
            iterations=1,
        )
        clustered = cv2.morphologyEx(
            clustered,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )
        contours, _ = cv2.findContours(clustered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for contour in contours:
            candidate = self._candidate_from_contour(contour, black_mask, method, gray)
            if candidate is not None:
                candidate.score += 0.35
                if best is None or candidate.score > best.score:
                    best = candidate
        return best

    def _candidate_from_contour(
        self,
        contour: np.ndarray,
        black_mask: np.ndarray,
        method: str,
        gray: Optional[np.ndarray] = None,
    ) -> Optional[SquareDetection]:
        contour_area = float(abs(cv2.contourArea(contour)))
        area_frac = contour_area / self.frame_area
        if area_frac < self.min_area_frac or area_frac > self.max_area_frac:
            return None

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0.0:
            return None

        quad = None
        epsilon_used = 0.0
        for epsilon_frac in np.linspace(0.01, 0.10, 19):
            approx = cv2.approxPolyDP(contour, float(epsilon_frac) * perimeter, True)
            if len(approx) == 4:
                quad = order_corners(approx)
                epsilon_used = float(epsilon_frac)
                break
        if quad is None:
            rect = cv2.minAreaRect(contour)
            quad = order_corners(cv2.boxPoints(rect))
            epsilon_used = -1.0

        x, y, box_w, box_h = cv2.boundingRect(quad.astype(np.int32))
        if box_w < 35 or box_h < 35:
            return None
        aspect = box_w / max(1.0, float(box_h))
        if not 0.45 <= aspect <= 2.2:
            return None

        angles = corner_angles_deg(quad)
        if len(angles) != 4:
            return None
        if any(angle < self.min_angle_deg or angle > self.max_angle_deg for angle in angles):
            return None

        rect_area = float(abs(cv2.contourArea(quad)))
        if rect_area <= 1.0:
            return None
        fill = contour_area / rect_area
        if fill < 0.18 or fill > 1.15:
            return None
        if gray is not None and self._quad_gray_std(gray, quad) < 24.0:
            return None

        pattern_score = self._black_frame_pattern_score(black_mask, quad)
        detection_method = f"{method}_pattern"
        if pattern_score is None:
            pattern_score = self._black_frame_grid_score(black_mask, quad)
            detection_method = f"{method}_grid"
        if pattern_score is None:
            return None

        return self._detection_from_quad(quad, area_frac, pattern_score, aspect, angles, detection_method)

    @staticmethod
    def _quad_gray_std(gray: np.ndarray, quad: np.ndarray, warp_size: int = 120) -> float:
        dst = np.array(
            [[0, 0], [warp_size - 1, 0], [warp_size - 1, warp_size - 1], [0, warp_size - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
        warped = cv2.warpPerspective(gray, matrix, (warp_size, warp_size))
        return float(np.std(warped))

    def _detection_from_quad(
        self,
        quad: np.ndarray,
        area_frac: float,
        pattern_score: float,
        aspect: float,
        angles: list[float],
        method: str,
    ) -> Optional[SquareDetection]:

        homography, _ = cv2.findHomography(self.black_square_board_corners, quad, 0)
        if homography is None:
            return None
        metric_pose = self._metric_from_object_image_points(
            self.black_square_board_corners,
            quad,
        )
        metric_forward_m, metric_left_m, homography_range = self._metric_from_homography(homography)
        range_m = self._range_from_square_corners(quad) or homography_range
        if metric_pose is not None:
            metric_forward_m, metric_left_m, range_m = metric_pose
        center = self._project(homography, np.array([[0.0, 0.0]], dtype=np.float32))[0]
        outer_board_corners = self._project(homography, self.outer_board_corners_m)
        x_axis = self._project(homography, np.array([[0.0, 0.0], [0.10, 0.0]], dtype=np.float32))
        yaw = math.atan2(x_axis[1][1] - x_axis[0][1], x_axis[1][0] - x_axis[0][0])

        score = (
            2.0 * pattern_score
            + 1.2 * min(area_frac * 12.0, 1.0)
            + 0.6 * (1.0 - min(abs(1.0 - aspect), 1.0))
            + 0.4 * (1.0 - min(float(np.std(angles)) / 25.0, 1.0))
        )
        return SquareDetection(
            cx=float(center[0]),
            cy=float(center[1]),
            yaw=float(yaw),
            area_frac=float(area_frac),
            corners=quad.astype(np.float32),
            metric_forward_m=float(metric_forward_m),
            metric_left_m=float(metric_left_m),
            range_m=float(range_m),
            score=float(score),
            method=method,
            outer_board_corners=outer_board_corners.astype(np.float32),
        )

    def _black_frame_pattern_score(self, black_mask: np.ndarray, quad: np.ndarray) -> Optional[float]:
        warp_size = 240
        dst = np.array(
            [[0, 0], [warp_size - 1, 0], [warp_size - 1, warp_size - 1], [0, warp_size - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
        warped = cv2.warpPerspective(black_mask, matrix, (warp_size, warp_size))
        black = warped > 128

        outer = 18
        inner_margin = 44
        border_bands = (
            black[:outer, :],
            black[-outer:, :],
            black[:, :outer],
            black[:, -outer:],
        )
        border_min = min(float(np.mean(band)) for band in border_bands)
        if border_min < 0.62:
            return None
        if float(np.mean(black)) > 0.72:
            return None

        moat = (
            black[outer:inner_margin, inner_margin:-inner_margin],
            black[-inner_margin:-outer, inner_margin:-inner_margin],
            black[inner_margin:-inner_margin, outer:inner_margin],
            black[inner_margin:-inner_margin, -inner_margin:-outer],
        )
        moat_max = max(float(np.mean(region)) for region in moat)
        if moat_max > 0.58:
            return None

        interior = black[inner_margin:-inner_margin, inner_margin:-inner_margin]
        interior_black = float(np.mean(interior))
        # Tags live inside, so the interior should be mixed, not all carpet/black or all white.
        if not 0.02 <= interior_black <= 0.75:
            return None
        return border_min + (1.0 - moat_max) + 0.25 * interior_black

    def _black_frame_grid_score(self, black_mask: np.ndarray, quad: np.ndarray) -> Optional[float]:
        warp_size = 240
        dst = np.array(
            [[0, 0], [warp_size - 1, 0], [warp_size - 1, warp_size - 1], [0, warp_size - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
        warped = cv2.warpPerspective(black_mask, matrix, (warp_size, warp_size))
        black = warped > 128

        # The candidate is the black hollow square. Its outer edge should have
        # black pixels, but perspective/partial shadows can break full bands.
        border = 22
        edge_means = [
            float(np.mean(black[:border, :])),
            float(np.mean(black[-border:, :])),
            float(np.mean(black[:, :border])),
            float(np.mean(black[:, -border:])),
        ]
        border_ratio = float(np.mean(edge_means))
        if min(edge_means) < 0.55:
            return None
        if float(np.mean(black)) > 0.72:
            return None
        inner = black[border : warp_size - border, border : warp_size - border].astype(np.uint8) * 255
        contours, _ = cv2.findContours(inner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tag_like = 0
        for contour in contours:
            area = float(abs(cv2.contourArea(contour)))
            if area < 80.0 or area > 5000.0:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            aspect = w / max(1.0, float(h))
            if 0.35 <= aspect <= 2.8:
                tag_like += 1
        if tag_like < 4:
            return None
        interior_ratio = float(np.mean(inner > 128))
        if not 0.03 <= interior_ratio <= 0.75:
            return None
        return float(
            border_ratio
            + 0.12 * min(tag_like, 9)
            + 0.25 * interior_ratio
        )

    def _range_from_square_corners(self, corners: np.ndarray) -> float:
        pts = corners.reshape(4, 2).astype(np.float32)
        sides = [float(np.linalg.norm(pts[(index + 1) % 4] - pts[index])) for index in range(4)]
        valid = [side for side in sides if side > 1.0]
        if not valid:
            return 0.0
        side_px = float(np.median(valid))
        return float(self.focal_px * self.black_square_size_m / side_px)

    def _metric_from_object_image_points(
        self,
        board_points_m: np.ndarray,
        image_points_px: np.ndarray,
    ) -> Optional[tuple[float, float, float]]:
        board_arr = np.asarray(board_points_m, dtype=np.float64)
        image_arr = np.asarray(image_points_px, dtype=np.float64)
        if board_arr.ndim != 2 or image_arr.ndim != 2 or len(board_arr) < 4:
            return None
        object_points = np.column_stack(
            (board_arr, np.zeros((len(board_arr),), dtype=np.float64))
        )
        candidates = []

        def add_candidate(rvec: np.ndarray, tvec: np.ndarray, error=None) -> None:
            tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
            if not np.all(np.isfinite(tvec)) or tvec[2] <= 0.0:
                return
            if error is None:
                projected, _ = cv2.projectPoints(
                    object_points,
                    rvec,
                    tvec.reshape(3, 1),
                    self.camera_matrix,
                    self.dist_coeffs,
                )
                error = float(
                    np.mean(
                        np.linalg.norm(projected.reshape(-1, 2) - image_arr, axis=1)
                    )
                )
            candidates.append((float(error), tvec))

        if hasattr(cv2, "solvePnPGeneric"):
            try:
                ok, rvecs, tvecs, errors = cv2.solvePnPGeneric(
                    object_points,
                    image_arr,
                    self.camera_matrix,
                    self.dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE,
                )
                if ok:
                    if errors is None:
                        errors = [None] * len(rvecs)
                    for rvec, tvec, error in zip(rvecs, tvecs, errors):
                        if error is not None:
                            error = float(np.asarray(error).reshape(-1)[0])
                        add_candidate(rvec, tvec, error)
            except cv2.error:
                pass

        try:
            ok, rvec, tvec = cv2.solvePnP(
                object_points,
                image_arr,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if ok:
                add_candidate(rvec, tvec)
        except cv2.error:
            pass

        if not candidates:
            return None
        _, tvec = min(candidates, key=lambda item: item[0])
        return float(-tvec[1]), float(-tvec[0]), float(tvec[2])

    @staticmethod
    def _project(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
        projected = cv2.perspectiveTransform(points.reshape(-1, 1, 2), homography)
        return projected.reshape(-1, 2).astype(np.float32)

    def _metric_from_homography(self, homography: np.ndarray) -> tuple[float, float, float]:
        try:
            inverse = np.linalg.inv(homography)
        except np.linalg.LinAlgError:
            return 0.0, 0.0, 0.0
        image_center = np.array([[self.width / 2.0, self.height / 2.0]], dtype=np.float32)
        board_at_camera = self._project(inverse, image_center)[0]
        board_error = -board_at_camera.astype(np.float64)
        scale_points = self._project(
            homography,
            np.array([[0.0, 0.0], [0.10, 0.0], [0.0, 0.10]], dtype=np.float32),
        )
        px_per_m_x = float(np.linalg.norm(scale_points[1] - scale_points[0]) / 0.10)
        px_per_m_y = float(np.linalg.norm(scale_points[2] - scale_points[0]) / 0.10)
        px_per_m = 0.5 * (px_per_m_x + px_per_m_y)
        range_m = self.focal_px / px_per_m if px_per_m > 1e-6 else 0.0
        return float(board_error[1]), float(board_error[0]), float(range_m)


class SquareDetectorExperimental(Node):
    def __init__(self):
        super().__init__("square_detector_experimental")

        self.declare_parameter("video_feed_topic", "/landing/image_raw")
        self.declare_parameter("camera_width", 640)
        self.declare_parameter("camera_height", 480)
        self.declare_parameter("camera_focal_px", 867.0)
        self.declare_parameter("process_rate_hz", 8.0)
        self.declare_parameter("gray_threshold", 142)
        self.declare_parameter("max_black_saturation", 130)
        self.declare_parameter("black_square_size_inches", 16.25)
        self.declare_parameter("outer_board_size_inches", 18.0)
        self.declare_parameter("publish_annotated_image", True)

        feed_topic = str(self.get_parameter("video_feed_topic").value)
        self._cam_w = int(self.get_parameter("camera_width").value)
        self._cam_h = int(self.get_parameter("camera_height").value)
        self._process_rate_hz = float(self.get_parameter("process_rate_hz").value)
        self._publish_annotated_image = bool(self.get_parameter("publish_annotated_image").value)
        self._detector = BlackSquareDetector(
            width=self._cam_w,
            height=self._cam_h,
            focal_px=float(self.get_parameter("camera_focal_px").value),
            black_square_size_inches=float(self.get_parameter("black_square_size_inches").value),
            outer_board_size_inches=float(self.get_parameter("outer_board_size_inches").value),
            gray_threshold=int(self.get_parameter("gray_threshold").value),
            max_black_saturation=int(self.get_parameter("max_black_saturation").value),
        )

        self._bridge = CvBridge()
        self._latest_msg: Optional[Image] = None
        self._frames = 0
        self._detections = 0

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(Image, feed_topic, self._on_image, qos)
        self.create_timer(1.0 / max(self._process_rate_hz, 1.0), self._process_latest_image)

        self._detected_pub = self.create_publisher(Bool, "/landing/experimental/square_detected", 10)
        self._pixel_pub = self.create_publisher(PointStamped, "/landing/experimental/square_pixel", 10)
        self._metric_pub = self.create_publisher(PointStamped, "/landing/experimental/square_metric", 10)
        self._range_pub = self.create_publisher(Float32, "/landing/experimental/square_range_m", 10)
        self._score_pub = self.create_publisher(Float32, "/landing/experimental/square_score", 10)
        self._method_pub = self.create_publisher(String, "/landing/experimental/square_method", 10)
        self._corners_pub = self.create_publisher(Float32MultiArray, "/landing/experimental/square_corners", 10)
        self._annotated_pub = self.create_publisher(Image, "/landing/experimental/square_image_annotated", 10)

        self.get_logger().info(
            f"experimental black-square detector on {feed_topic} at {self._process_rate_hz:.1f}Hz "
            f"threshold={self._detector.gray_threshold} "
            f"black_square={self._detector.black_square_size_m / 0.0254:.2f}in "
            f"outer_board={self._detector.outer_board_size_m / 0.0254:.2f}in"
        )

    def _on_image(self, msg: Image) -> None:
        self._latest_msg = msg

    def _process_latest_image(self) -> None:
        msg = self._latest_msg
        self._latest_msg = None
        if msg is None:
            return
        image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        result = self._detector.detect(image)
        self._frames += 1
        self._detected_pub.publish(Bool(data=result is not None))
        if result is not None:
            self._detections += 1
            pixel = PointStamped()
            pixel.header = msg.header
            pixel.point.x = float(result.cx)
            pixel.point.y = float(result.cy)
            pixel.point.z = float(result.yaw)
            self._pixel_pub.publish(pixel)

            metric = PointStamped()
            metric.header = msg.header
            metric.point.x = float(result.metric_forward_m)
            metric.point.y = float(result.metric_left_m)
            metric.point.z = float(result.yaw)
            self._metric_pub.publish(metric)

            self._range_pub.publish(Float32(data=float(result.range_m)))
            self._score_pub.publish(Float32(data=float(result.score)))
            self._method_pub.publish(String(data=result.method))
            self._corners_pub.publish(Float32MultiArray(data=result.corners.reshape(-1).astype(float).tolist()))

        if self._publish_annotated_image or self._annotated_pub.get_subscription_count() > 0:
            annotated = self._annotate(image, result)
            annotated_msg = self._bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            annotated_msg.header = msg.header
            self._annotated_pub.publish(annotated_msg)

    def _annotate(self, image: np.ndarray, result: Optional[SquareDetection]) -> np.ndarray:
        annotated = image.copy()
        cv2.circle(annotated, (self._cam_w // 2, self._cam_h // 2), 6, (0, 255, 0), 2)
        if result is None:
            cv2.putText(annotated, "experimental square: no", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return annotated
        cv2.polylines(
            annotated,
            [result.outer_board_corners.astype(np.int32).reshape(-1, 1, 2)],
            True,
            (0, 255, 0),
            1,
        )
        cv2.polylines(annotated, [result.corners.astype(np.int32).reshape(-1, 1, 2)], True, (255, 0, 0), 3)
        cv2.circle(annotated, (int(round(result.cx)), int(round(result.cy))), 6, (0, 0, 255), -1)
        lines = [
            f"experimental BLACK frame: yes method={result.method} score={result.score:.2f}",
            f"target=({result.cx:.1f},{result.cy:.1f}) range={result.range_m:.2f}m",
            f"metric fwd={result.metric_forward_m:+.2f} left={result.metric_left_m:+.2f}",
        ]
        overlay = annotated.copy()
        cv2.rectangle(overlay, (4, 4), (self._cam_w - 4, 86), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0.0, annotated)
        y = 27
        for line in lines:
            cv2.putText(annotated, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
            y += 24
        return annotated


def main(args=None):
    if rclpy is None:
        raise RuntimeError("ROS 2 Python modules are required to run square_detector_experimental as a node")
    rclpy.init(args=args)
    node = SquareDetectorExperimental()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
