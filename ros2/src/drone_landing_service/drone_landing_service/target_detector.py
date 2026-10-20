#!/usr/bin/env python3
"""Low-latency single-pass AprilTag detector for precision landing."""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np
import rclpy
from std_msgs.msg import Float32MultiArray, Int32MultiArray, String

from drone_landing_service.lightweight_landing_detector import (
    LightweightLandingDetector,
)
from drone_landing_service.target_detector_deprecated import (
    TargetDetector as BaseTargetDetector,
    TargetResult,
)


class TargetDetector(BaseTargetDetector):
    """Detect tags once per frame, at full resolution, in a tracked ROI."""

    def __init__(self):
        super().__init__()
        self.declare_parameter("tag_roi_min_padding_px", 96)
        self.declare_parameter("tag_roi_padding_tag_widths", 4.0)
        self._roi_min_padding_px = int(
            self.get_parameter("tag_roi_min_padding_px").value
        )
        self._roi_padding_tag_widths = float(
            self.get_parameter("tag_roi_padding_tag_widths").value
        )
        self._tag_roi: Optional[tuple[int, int, int, int]] = None
        self._aruco_detector = self._make_detector()
        self._lightweight = LightweightLandingDetector(
            self._cam_w,
            self._cam_h,
            focal_px=self._camera_focal_px,
            max_search_dimension=max(self._cam_w, self._cam_h),
            roi_padding_px=self._roi_min_padding_px,
            roi_padding_tag_widths=self._roi_padding_tag_widths,
            use_roi=True,
            white_threshold_on_miss=True,
            temporary_white_pose_retry=True,
        )

        self._tag_corners_pub = self.create_publisher(
            Float32MultiArray, "/landing/visible_tag_corners", 10
        )
        self._direct_tag_ids_pub = self.create_publisher(
            Int32MultiArray, "/landing/direct_visible_tag_ids", 10
        )
        self._standard_tag_ids_pub = self.create_publisher(
            Int32MultiArray, "/landing/direct_visible_standard_tag_ids", 10
        )
        self._landmark_tag_ids_pub = self.create_publisher(
            Int32MultiArray, "/landing/direct_visible_landmark_tag_ids", 10
        )
        self._tag_methods_pub = self.create_publisher(
            String, "/landing/tag_detection_methods", 10
        )
        self.get_logger().info(
            "lightweight full-resolution AprilTag detector enabled"
        )
        self._start_workers()

    def _configure_detector_params(self) -> None:
        params = self._aruco_params
        settings = {
            "minMarkerPerimeterRate": 0.01,
            "maxMarkerPerimeterRate": 4.0,
            "minDistanceToBorder": 0,
            "errorCorrectionRate": 1.0,
            "adaptiveThreshWinSizeMin": 3,
            "adaptiveThreshWinSizeMax": 31,
            "adaptiveThreshWinSizeStep": 4,
            "adaptiveThreshConstant": 7,
            "aprilTagQuadDecimate": 1.0,
            "aprilTagQuadSigma": 0.0,
            "aprilTagMinWhiteBlackDiff": 2,
            "aprilTagDeglitch": 1,
        }
        for name, value in settings.items():
            if hasattr(params, name):
                setattr(params, name, value)
        if hasattr(params, "cornerRefinementMethod"):
            params.cornerRefinementMethod = getattr(
                cv2.aruco, "CORNER_REFINE_NONE", 0
            )

    def _make_detector(self):
        detector_type = getattr(cv2.aruco, "ArucoDetector", None)
        if detector_type is None:
            return None
        return detector_type(self._aruco_dict, self._aruco_params)

    def _detect_markers(self, gray: np.ndarray):
        if self._aruco_detector is not None:
            return self._aruco_detector.detectMarkers(gray)
        return cv2.aruco.detectMarkers(
            gray, self._aruco_dict, parameters=self._aruco_params
        )

    def _suppress_highlights(self, image: np.ndarray) -> np.ndarray:
        return image

    def _detect_tag_target(self, image: np.ndarray) -> Optional[TargetResult]:
        frame_height, frame_width = image.shape[:2]
        active_roi = self._lightweight.roi
        if active_roi is None:
            roi_area = frame_height * frame_width
        else:
            x0, y0, x1, y1 = active_roi
            roi_area = max(0, x1 - x0) * max(0, y1 - y0)
        self._last_roi_area_frac = float(
            roi_area / max(1, frame_height * frame_width)
        )
        started = time.perf_counter()
        detection = self._lightweight.detect(image)
        elapsed_ms = 1000.0 * (time.perf_counter() - started)
        self._last_gray_conversion_ms = 0.0
        self._last_marker_search_ms = elapsed_ms
        self._last_pose_ms = 0.0

        tags = self._debug_tag_results(
            self._lightweight.last_tags, self._lightweight.last_search
        )
        self._publish_tag_details(
            tags,
            self._lightweight.last_search,
            landmark_ids=self._lightweight.last_landmark_ids,
        )
        if detection is None:
            return None
        primary_id = (
            self._center_tag_id
            if self._center_tag_id in detection.tag_corners_px
            else max(
                detection.tag_corners_px,
                key=lambda tag_id: abs(
                    cv2.contourArea(detection.tag_corners_px[tag_id])
                ),
            )
            if detection.tag_corners_px
            else -1
        )
        area_frac = abs(cv2.contourArea(detection.board_corners_px)) / max(
            1.0, float(frame_height * frame_width)
        )
        return TargetResult(
            cx=int(round(detection.center_px[0])),
            cy=int(round(detection.center_px[1])),
            yaw=float(detection.yaw_rad),
            area_frac=float(min(area_frac, 1.0)),
            corners=detection.board_corners_px,
            source=f"tag_homography_lightweight:{detection.search}",
            target_id=int(primary_id),
            tag_ids=tuple(int(tag_id) for tag_id in detection.ids),
            metric_forward_m=float(detection.forward_m),
            metric_left_m=float(detection.left_m),
            range_m=float(detection.range_m),
        )

    def _debug_tag_results(
        self, corners_by_id: dict[int, np.ndarray], method: str
    ) -> dict[int, TargetResult]:
        tags = {}
        for marker_id, corners in corners_by_id.items():
            center = np.mean(corners, axis=0)
            tags[int(marker_id)] = TargetResult(
                cx=int(round(float(center[0]))),
                cy=int(round(float(center[1]))),
                yaw=0.0,
                area_frac=abs(cv2.contourArea(corners)) / max(1.0, self._frame_area),
                corners=corners,
                source=f"tag_{marker_id}_{method}",
                target_id=int(marker_id),
                tag_ids=(int(marker_id),),
            )
        return tags

    def _detect_square_target(self, image: np.ndarray) -> Optional[TargetResult]:
        if not self._square_detection_enabled or self._square_template_detector is None:
            return None
        detection = self._square_template_detector.detect(image)
        if detection is None:
            return None
        return self._target_from_square_detection(detection)

    def _active_crop(self, image: np.ndarray) -> tuple[np.ndarray, int, int]:
        if self._tag_roi is None:
            return image, 0, 0
        x0, y0, x1, y1 = self._tag_roi
        if x1 <= x0 or y1 <= y0:
            self._tag_roi = None
            return image, 0, 0
        return image[y0:y1, x0:x1], x0, y0

    @staticmethod
    def _offset_corners(corners, x0: int, y0: int):
        if not corners or (x0 == 0 and y0 == 0):
            return corners
        offset = np.array([x0, y0], dtype=np.float32)
        return [corner.astype(np.float32) + offset for corner in corners]

    @staticmethod
    def _roi_from_tags(
        tags: dict[int, TargetResult],
        image_shape,
        min_padding_px: int,
        padding_tag_widths: float,
    ) -> tuple[int, int, int, int]:
        points = np.concatenate(
            [tag.corners.reshape(-1, 2) for tag in tags.values() if tag.corners is not None]
        )
        tag_widths = [
            max(
                np.linalg.norm(tag.corners[1] - tag.corners[0]),
                np.linalg.norm(tag.corners[2] - tag.corners[1]),
            )
            for tag in tags.values()
            if tag.corners is not None
        ]
        padding = max(min_padding_px, int(round(max(tag_widths) * padding_tag_widths)))
        height, width = image_shape[:2]
        x0 = max(0, int(np.floor(points[:, 0].min())) - padding)
        y0 = max(0, int(np.floor(points[:, 1].min())) - padding)
        x1 = min(width, int(np.ceil(points[:, 0].max())) + padding + 1)
        y1 = min(height, int(np.ceil(points[:, 1].max())) + padding + 1)
        return x0, y0, x1, y1

    def _publish_tag_details(
        self,
        tags: dict[int, TargetResult],
        method: str,
        landmark_ids: tuple[int, ...] = (),
    ) -> None:
        ids = sorted(tags)
        landmark_ids = sorted(int(marker_id) for marker_id in landmark_ids)
        if self._tag_corners_pub.get_subscription_count() > 0:
            values = []
            for marker_id, result in sorted(tags.items()):
                values.extend(
                    [
                        float(marker_id),
                        float(result.cx),
                        float(result.cy),
                        *result.corners.reshape(-1).astype(float).tolist(),
                    ]
                )
            self._tag_corners_pub.publish(Float32MultiArray(data=values))
        if self._direct_tag_ids_pub.get_subscription_count() > 0:
            self._direct_tag_ids_pub.publish(Int32MultiArray(data=ids))
        if self._standard_tag_ids_pub.get_subscription_count() > 0:
            self._standard_tag_ids_pub.publish(Int32MultiArray(data=ids))
        if self._landmark_tag_ids_pub.get_subscription_count() > 0:
            self._landmark_tag_ids_pub.publish(Int32MultiArray(data=landmark_ids))
        if self._tag_methods_pub.get_subscription_count() > 0:
            detail = (
                f"{method}:standard={'|'.join(str(marker_id) for marker_id in ids)};"
                f"landmark={'|'.join(str(marker_id) for marker_id in landmark_ids)}"
            )
            self._tag_methods_pub.publish(String(data=detail))

    def destroy_node(self):
        result = super().destroy_node()
        self._lightweight.close()
        return result


def main(args=None):
    rclpy.init(args=args)
    node = TargetDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
