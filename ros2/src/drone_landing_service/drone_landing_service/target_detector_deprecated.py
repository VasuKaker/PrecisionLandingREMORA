#!/usr/bin/env python3
"""Deprecated single-pass detector for the landing square and AprilTag grid."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, TwistStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, Float32MultiArray, Int32, Int32MultiArray, String

from drone_landing_service.square_detector_experimental import BlackSquareDetector, SquareDetection

ARUCO_DICTS = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_APRILTAG_16H5": cv2.aruco.DICT_APRILTAG_16H5,
    "DICT_APRILTAG_25H9": cv2.aruco.DICT_APRILTAG_25H9,
    "DICT_APRILTAG_36H10": cv2.aruco.DICT_APRILTAG_36H10,
    "DICT_APRILTAG_36H11": cv2.aruco.DICT_APRILTAG_36H11,
}

@dataclass
class TargetResult:
    cx: int
    cy: int
    yaw: float
    area_frac: float
    corners: Optional[np.ndarray]
    source: str
    target_id: int
    tag_ids: tuple[int, ...] = ()
    desired_vx: float = 0.0
    desired_vy: float = 0.0
    metric_forward_m: float = 0.0
    metric_left_m: float = 0.0
    range_m: float = 0.0
    score: float = 0.0


@dataclass
class SquareJob:
    header: object
    image: np.ndarray
    queued_at: float
    tag_generation: int


@dataclass
class SquareOutcome:
    header: object
    result: Optional[TargetResult]
    processing_ms: float
    completed_at: float
    queued_at: float
    tag_generation: int


class TargetDetector(Node):
    def __init__(self):
        super().__init__("target_detector")

        self.declare_parameter("video_feed_topic", "")
        self.declare_parameter("pixel_target_topic", "")
        self.declare_parameter("target_detected_topic", "")
        self.declare_parameter("aruco_dictionary", "DICT_APRILTAG_16H5")
        self.declare_parameter("grid_tag_ids", [1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.declare_parameter("center_tag_id", 5)
        self.declare_parameter("camera_width", 640)
        self.declare_parameter("camera_height", 480)
        self.declare_parameter("camera_focal_px", 760.0)
        self.declare_parameter("process_rate_hz", 30.0)
        self.declare_parameter("detection_hold_s", 0.35)
        self.declare_parameter("binary_white_threshold", 195)
        self.declare_parameter("highlight_suppression_enabled", True)
        self.declare_parameter("highlight_suppression_knee", 190)
        self.declare_parameter("highlight_suppression_compression", 0.20)
        self.declare_parameter("highlight_suppression_max_value", 230)
        self.declare_parameter("square_binary_threshold", 142)
        self.declare_parameter("square_detection_enabled", True)
        self.declare_parameter("square_template_enabled", True)
        self.declare_parameter("square_fallback_stale_s", 2.0)
        self.declare_parameter("square_fallback_prior_timeout_s", 8.0)
        self.declare_parameter("square_fallback_eval_period_s", 0.50)
        self.declare_parameter("square_fallback_max_range_jump_m", 1.0)
        self.declare_parameter("square_fallback_max_xy_jump_m", 0.75)
        self.declare_parameter("square_min_area_frac", 0.02)
        self.declare_parameter("square_max_area_frac", 0.95)
        self.declare_parameter("square_aspect_tolerance", 0.35)
        self.declare_parameter("square_min_tag_area_ratio", 5.0)
        self.declare_parameter("square_min_visible_tags", 1)
        self.declare_parameter("square_template_outer_board_size_inches", 18.0)
        self.declare_parameter("target_size_inches", 18.0)
        self.declare_parameter("outer_white_inches", 0.5)
        self.declare_parameter("black_border_inches", 0.75)
        self.declare_parameter("inner_padding_inches", 0.35)
        self.declare_parameter("tag_gap_inches", 0.35)
        self.declare_parameter("prefer_tag_homography", True)
        self.declare_parameter("publish_annotated_image", False)
        self.declare_parameter("stats_period_sec", 2.0)

        feed_topic = self.get_parameter("video_feed_topic").value
        target_topic = self.get_parameter("pixel_target_topic").value
        detected_topic = self.get_parameter("target_detected_topic").value
        dict_name = self.get_parameter("aruco_dictionary").value
        grid_tag_ids = list(self.get_parameter("grid_tag_ids").value)

        self._cam_w = int(self.get_parameter("camera_width").value)
        self._cam_h = int(self.get_parameter("camera_height").value)
        self._camera_focal_px = float(self.get_parameter("camera_focal_px").value)
        self._frame_area = float(self._cam_w * self._cam_h)
        self._grid_tag_ids = {int(marker_id) for marker_id in grid_tag_ids}
        self._center_tag_id = int(self.get_parameter("center_tag_id").value)
        self._process_rate_hz = float(self.get_parameter("process_rate_hz").value)
        self._detection_hold_s = float(self.get_parameter("detection_hold_s").value)
        self._binary_white_threshold = int(
            self.get_parameter("binary_white_threshold").value
        )
        self._highlight_suppression_enabled = bool(
            self.get_parameter("highlight_suppression_enabled").value
        )
        self._highlight_suppression_knee = int(
            self.get_parameter("highlight_suppression_knee").value
        )
        self._highlight_suppression_compression = float(
            self.get_parameter("highlight_suppression_compression").value
        )
        self._highlight_suppression_max_value = int(
            self.get_parameter("highlight_suppression_max_value").value
        )
        self._square_binary_threshold = int(
            self.get_parameter("square_binary_threshold").value
        )
        self._square_detection_enabled = bool(
            self.get_parameter("square_detection_enabled").value
        )
        self._square_template_enabled = bool(
            self.get_parameter("square_template_enabled").value
        )
        self._square_fallback_stale_s = float(
            self.get_parameter("square_fallback_stale_s").value
        )
        self._square_fallback_prior_timeout_s = float(
            self.get_parameter("square_fallback_prior_timeout_s").value
        )
        self._square_fallback_eval_period_s = float(
            self.get_parameter("square_fallback_eval_period_s").value
        )
        self._square_fallback_max_range_jump_m = float(
            self.get_parameter("square_fallback_max_range_jump_m").value
        )
        self._square_fallback_max_xy_jump_m = float(
            self.get_parameter("square_fallback_max_xy_jump_m").value
        )
        self._square_min_area_frac = float(
            self.get_parameter("square_min_area_frac").value
        )
        self._square_max_area_frac = float(
            self.get_parameter("square_max_area_frac").value
        )
        self._square_aspect_tolerance = float(
            self.get_parameter("square_aspect_tolerance").value
        )
        self._square_min_tag_area_ratio = float(
            self.get_parameter("square_min_tag_area_ratio").value
        )
        self._square_min_visible_tags = int(self.get_parameter("square_min_visible_tags").value)
        self._square_template_outer_board_size_inches = float(
            self.get_parameter("square_template_outer_board_size_inches").value
        )
        self._target_size_inches = float(
            self.get_parameter("target_size_inches").value
        )
        self._outer_white_inches = float(
            self.get_parameter("outer_white_inches").value
        )
        self._black_border_inches = float(
            self.get_parameter("black_border_inches").value
        )
        self._inner_padding_inches = float(
            self.get_parameter("inner_padding_inches").value
        )
        self._tag_gap_inches = float(self.get_parameter("tag_gap_inches").value)
        self._prefer_tag_homography = bool(
            self.get_parameter("prefer_tag_homography").value
        )
        self._publish_annotated_image = bool(
            self.get_parameter("publish_annotated_image").value
        )
        stats_period_sec = float(self.get_parameter("stats_period_sec").value)
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        dict_id = ARUCO_DICTS.get(dict_name, cv2.aruco.DICT_APRILTAG_36H11)
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        if hasattr(cv2.aruco, "DetectorParameters_create"):
            self._aruco_params = cv2.aruco.DetectorParameters_create()
        else:
            self._aruco_params = cv2.aruco.DetectorParameters()
        self._target_size_m = self._target_size_inches * 0.0254
        self._camera_matrix = np.array(
            [
                [self._camera_focal_px, 0.0, self._cam_w / 2.0],
                [0.0, self._camera_focal_px, self._cam_h / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        self._black_square_size_inches = self._target_size_inches - 2.0 * self._outer_white_inches
        self._black_square_size_m = self._black_square_size_inches * 0.0254
        self._inner_size_inches = self._black_square_size_inches - 2.0 * self._black_border_inches
        self._tag_size_inches = (
            self._inner_size_inches
            - 2.0 * self._inner_padding_inches
            - 2.0 * self._tag_gap_inches
        ) / 3.0
        self._tag_size_m = self._tag_size_inches * 0.0254
        self._configure_detector_params()
        self._tag_board_corners = self._build_tag_board_corners()
        half = self._target_size_m / 2.0
        self._target_board_corners = np.array(
            [
                [-half, -half],
                [half, -half],
                [half, half],
                [-half, half],
            ],
            dtype=np.float32,
        )
        black_square_half = self._black_square_size_m / 2.0
        self._black_square_board_corners = np.array(
            [
                [-black_square_half, -black_square_half],
                [black_square_half, -black_square_half],
                [black_square_half, black_square_half],
                [-black_square_half, black_square_half],
            ],
            dtype=np.float32,
        )
        self._square_template_detector = (
            BlackSquareDetector(
                width=self._cam_w,
                height=self._cam_h,
                focal_px=self._camera_focal_px,
                black_square_size_inches=self._black_square_size_inches,
                outer_board_size_inches=self._square_template_outer_board_size_inches,
                gray_threshold=self._square_binary_threshold,
            )
            if self._square_template_enabled
            else None
        )

        self._bridge = CvBridge()
        self._pub = self.create_publisher(PointStamped, target_topic, 10)
        self._detected_pub = self.create_publisher(Bool, detected_topic, 10)
        self._area_pub = self.create_publisher(Float32, "/landing/target_area", 10)
        self._metric_pub = self.create_publisher(PointStamped, "/landing/target_metric", 10)
        self._fresh_metric_pub = self.create_publisher(PointStamped, "/landing/target_metric_fresh", 10)
        self._range_pub = self.create_publisher(Float32, "/landing/target_range_m", 10)
        self._target_id_pub = self.create_publisher(Int32, "/landing/target_id", 10)
        self._target_source_pub = self.create_publisher(String, "/landing/target_source", 10)
        self._square_detected_pub = self.create_publisher(Bool, "/landing/square_detected", 10)
        self._square_pixel_pub = self.create_publisher(PointStamped, "/landing/square_pixel", 10)
        self._square_metric_pub = self.create_publisher(PointStamped, "/landing/square_metric", 10)
        self._square_range_pub = self.create_publisher(Float32, "/landing/square_range_m", 10)
        self._square_score_pub = self.create_publisher(Float32, "/landing/square_score", 10)
        self._square_method_pub = self.create_publisher(String, "/landing/square_method", 10)
        self._square_fallback_accepted_pub = self.create_publisher(
            Bool,
            "/landing/square_fallback_accepted",
            10,
        )
        self._square_fallback_reject_pub = self.create_publisher(
            String,
            "/landing/square_fallback_reject_reason",
            10,
        )
        self._square_corners_pub = self.create_publisher(
            Float32MultiArray,
            "/landing/square_corners",
            10,
        )
        self._visible_tag_ids_pub = self.create_publisher(
            Int32MultiArray,
            "/landing/visible_tag_ids",
            10,
        )
        self._guidance_velocity_pub = self.create_publisher(
            TwistStamped,
            "/landing/target_guidance_velocity",
            10,
        )
        self._processing_latency_pub = self.create_publisher(
            Float32,
            "/landing/detector_processing_latency_ms",
            10,
        )
        self._total_latency_pub = self.create_publisher(
            Float32,
            "/landing/detector_total_latency_ms",
            10,
        )
        self._timing_pub = self.create_publisher(
            Float32MultiArray,
            "/landing/detector_timing",
            10,
        )
        self._annotated_pub = self.create_publisher(Image, "/landing/image_annotated", 10)
        image_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, feed_topic, self._on_image, image_qos)

        self._latest_msg: Optional[Image] = None
        self._mailbox_condition = threading.Condition()
        self._square_condition = threading.Condition()
        self._worker_stop = False
        self._processing_thread: Optional[threading.Thread] = None
        self._square_thread: Optional[threading.Thread] = None
        self._square_job: Optional[SquareJob] = None
        self._square_outcome: Optional[SquareOutcome] = None
        self._last_result: Optional[TargetResult] = None
        self._last_tag_result: Optional[TargetResult] = None
        self._tag_generation = 0
        self._start_time = time.perf_counter()
        self._last_detection_time = 0.0
        self._last_tag_detection_time = 0.0
        self._last_square_eval_time = 0.0
        self._last_image_time = 0.0
        self._frames_received = 0
        self._frames_processed = 0
        self._frames_detected = 0
        self._processing_time_sum = 0.0
        self._message_age_sum = 0.0
        self._last_gray_conversion_ms = 0.0
        self._last_marker_search_ms = 0.0
        self._last_pose_ms = 0.0
        self._last_roi_area_frac = 1.0

        self.create_timer(max(stats_period_sec, 0.5), self._report_stats)

        self.get_logger().info(
            f"detecting square + {dict_name} grid ids={sorted(self._grid_tag_ids)} "
            f"center_id={self._center_tag_id} on {feed_topic} -> {target_topic} "
            f"(rate={self._process_rate_hz:.1f}Hz, hold={self._detection_hold_s:.2f}s, "
            f"target={self._target_size_inches:.1f}in, "
            f"tag_homography={self._prefer_tag_homography}, "
            f"square_fallback_stale={self._square_fallback_stale_s:.2f}s, "
            f"square_template={self._square_template_enabled})"
        )

    def _configure_detector_params(self) -> None:
        params = self._aruco_params
        if hasattr(params, "minMarkerPerimeterRate"):
            params.minMarkerPerimeterRate = 0.02
        if hasattr(params, "maxMarkerPerimeterRate"):
            params.maxMarkerPerimeterRate = 4.0
        if hasattr(params, "minDistanceToBorder"):
            params.minDistanceToBorder = 0
        if hasattr(params, "errorCorrectionRate"):
            params.errorCorrectionRate = 1.0
        if hasattr(params, "adaptiveThreshWinSizeMin"):
            params.adaptiveThreshWinSizeMin = 3
        if hasattr(params, "adaptiveThreshWinSizeMax"):
            params.adaptiveThreshWinSizeMax = 31
        if hasattr(params, "adaptiveThreshWinSizeStep"):
            params.adaptiveThreshWinSizeStep = 8
        if hasattr(params, "adaptiveThreshConstant"):
            params.adaptiveThreshConstant = 7
        if hasattr(params, "aprilTagQuadDecimate"):
            params.aprilTagQuadDecimate = 1.0
        if hasattr(params, "aprilTagQuadSigma"):
            params.aprilTagQuadSigma = 0.0
        if hasattr(params, "aprilTagMinWhiteBlackDiff"):
            params.aprilTagMinWhiteBlackDiff = 3
        if hasattr(params, "aprilTagDeglitch"):
            params.aprilTagDeglitch = 1
        if hasattr(params, "cornerRefinementMethod"):
            params.cornerRefinementMethod = getattr(cv2.aruco, "CORNER_REFINE_NONE", 0)

    def _on_image(self, msg: Image):
        with self._mailbox_condition:
            self._latest_msg = msg
            self._frames_received += 1
            self._last_image_time = time.perf_counter()
            self._mailbox_condition.notify()

    def _start_workers(self) -> None:
        if self._processing_thread is not None:
            return
        self._worker_stop = False
        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            name="target-detector",
            daemon=True,
        )
        self._processing_thread.start()
        if self._square_detection_enabled and self._square_template_detector is not None:
            self._square_thread = threading.Thread(
                target=self._square_loop,
                name="square-fallback",
                daemon=True,
            )
            self._square_thread.start()

    def _processing_loop(self) -> None:
        period_s = 1.0 / max(self._process_rate_hz, 1.0)
        next_start = 0.0
        while True:
            with self._mailbox_condition:
                while not self._worker_stop and self._latest_msg is None:
                    self._mailbox_condition.wait(timeout=0.5)
                if self._worker_stop:
                    return
                wait_s = next_start - time.perf_counter()
                if wait_s > 0.0:
                    self._mailbox_condition.wait(timeout=wait_s)
                    if self._worker_stop:
                        return
                msg = self._latest_msg
                self._latest_msg = None
            if msg is None:
                continue
            started = time.perf_counter()
            next_start = started + period_s
            try:
                self._process_image(msg)
            except Exception as exc:
                self.get_logger().error(f"detector worker failed: {exc}")

    def _square_loop(self) -> None:
        while True:
            with self._square_condition:
                while not self._worker_stop and self._square_job is None:
                    self._square_condition.wait(timeout=0.5)
                if self._worker_stop:
                    return
                job = self._square_job
                self._square_job = None
            if job is None:
                continue
            started = time.perf_counter()
            try:
                image = job.image
                if image.ndim == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                result = self._detect_square_target(image)
            except Exception as exc:
                self.get_logger().warning(f"square fallback worker failed: {exc}")
                result = None
            completed = time.perf_counter()
            outcome = SquareOutcome(
                header=job.header,
                result=result,
                processing_ms=1000.0 * (completed - started),
                completed_at=completed,
                queued_at=job.queued_at,
                tag_generation=job.tag_generation,
            )
            with self._square_condition:
                self._square_outcome = outcome

    def _queue_square_job(self, msg: Image, image: np.ndarray) -> None:
        job = SquareJob(
            header=msg.header,
            image=image.copy(),
            queued_at=time.perf_counter(),
            tag_generation=self._tag_generation,
        )
        with self._square_condition:
            self._square_job = job
            self._square_condition.notify()

    def _take_square_outcome(self) -> Optional[SquareOutcome]:
        with self._square_condition:
            outcome = self._square_outcome
            self._square_outcome = None
        return outcome

    def _process_latest_image(self):
        with self._mailbox_condition:
            msg = self._latest_msg
            self._latest_msg = None
        if msg is None:
            return
        self._process_image(msg)

    def _process_image(self, msg: Image):
        t0 = time.perf_counter()
        frame_age_start_s = self._message_age_seconds(msg)
        conversion_started = time.perf_counter()
        if str(msg.encoding).lower() in ("mono8", "8uc1"):
            image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        else:
            image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        image = self._suppress_highlights(image)
        bridge_conversion_ms = 1000.0 * (time.perf_counter() - conversion_started)
        now = time.perf_counter()

        tag_result = self._detect_tag_target(image)
        square_outcome = self._take_square_outcome()
        square_result = None
        square_ms = 0.0
        square_header = msg.header
        square_reject_reason = "no_square"
        if square_outcome is not None:
            square_ms = square_outcome.processing_ms
            square_header = square_outcome.header
            square_age_s = self._header_age_seconds(square_outcome.header)
            if square_age_s <= 0.0:
                square_age_s = time.perf_counter() - square_outcome.queued_at
            if square_outcome.tag_generation != self._tag_generation:
                square_reject_reason = "newer_tag_available"
            elif square_age_s > self._detection_hold_s:
                square_reject_reason = f"square_stale_{square_age_s:.2f}s"
            else:
                square_result = square_outcome.result
        self._publish_square_debug(square_header, square_result)
        square_ready, status_reason = self._square_fallback_status(now, square_result)
        if square_result is not None or square_reject_reason == "no_square":
            square_reject_reason = status_reason

        if tag_result is not None:
            result = tag_result
            result_header = msg.header
            self._tag_generation += 1
            self._last_tag_result = tag_result
            self._last_tag_detection_time = now
            self._last_result = result
            self._last_detection_time = now
        elif (
            square_result is not None
            and square_ready
        ):
            result = square_result
            result_header = square_header
            self._last_result = result
            self._last_detection_time = now
        elif self._last_result is not None and now - self._last_detection_time <= self._detection_hold_s:
            result = self._last_result
            result_header = msg.header
        else:
            result = None
            result_header = msg.header
            self._last_result = None

        if tag_result is None and self._should_evaluate_square_fallback(now):
            self._queue_square_job(msg, image)

        publishing_started = time.perf_counter()
        self._square_fallback_accepted_pub.publish(
            Bool(data=bool(result is not None and result.target_id == 0 and square_ready))
        )
        self._square_fallback_reject_pub.publish(String(data=square_reject_reason))
        self._detected_pub.publish(Bool(data=result is not None))
        self._publish_detection_debug(result_header, result)

        if result is not None:
            pt = PointStamped()
            pt.header = result_header
            pt.point.x = float(result.cx)
            pt.point.y = float(result.cy)
            pt.point.z = float(result.yaw)
            self._pub.publish(pt)
            self._area_pub.publish(Float32(data=float(result.area_frac)))
            metric = PointStamped()
            metric.header = result_header
            metric.point.x = float(result.metric_forward_m)
            metric.point.y = float(result.metric_left_m)
            metric.point.z = float(result.range_m)
            self._metric_pub.publish(metric)
            if tag_result is not None:
                self._fresh_metric_pub.publish(metric)
            self._range_pub.publish(Float32(data=float(result.range_m)))
            self._frames_detected += 1

        should_publish_annotated = self._publish_annotated_image or (
            self._annotated_pub.get_subscription_count() > 0
        )
        if should_publish_annotated:
            annotated = self._build_annotated_image(image, result)
            annotated_msg = self._bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            annotated_msg.header = msg.header
            self._annotated_pub.publish(annotated_msg)

        publish_ms = 1000.0 * (time.perf_counter() - publishing_started)
        processing_s = time.perf_counter() - t0
        total_latency_s = self._message_age_seconds(msg)
        self._processing_latency_pub.publish(
            Float32(data=float(processing_s * 1000.0))
        )
        self._total_latency_pub.publish(
            Float32(data=float(total_latency_s * 1000.0))
        )
        self._timing_pub.publish(
            Float32MultiArray(
                data=[
                    float(frame_age_start_s * 1000.0),
                    float(bridge_conversion_ms + self._last_gray_conversion_ms),
                    float(self._last_marker_search_ms),
                    float(self._last_pose_ms),
                    float(square_ms),
                    float(publish_ms),
                    float(processing_s * 1000.0),
                    float(self._last_roi_area_frac),
                    1.0,
                ]
            )
        )
        self._frames_processed += 1
        self._processing_time_sum += processing_s
        self._message_age_sum += total_latency_s

    def _square_fallback_status(
        self,
        now: float,
        square_result: Optional[TargetResult],
    ) -> tuple[bool, str]:
        if not self._square_detection_enabled:
            return False, "disabled"
        if square_result is None:
            return False, "no_square"
        if self._last_tag_detection_time <= 0.0:
            return False, "no_tag_prior"

        tag_age = now - self._last_tag_detection_time
        if tag_age < self._square_fallback_stale_s:
            return False, f"tag_fresh_{tag_age:.2f}s"
        if tag_age > self._square_fallback_prior_timeout_s:
            return False, f"tag_prior_too_old_{tag_age:.2f}s"
        if self._last_tag_result is None:
            return False, "missing_tag_prior"

        range_jump = abs(square_result.range_m - self._last_tag_result.range_m)
        if range_jump > self._square_fallback_max_range_jump_m:
            return False, f"range_jump_{range_jump:.2f}m"
        xy_jump = math.hypot(
            square_result.metric_forward_m - self._last_tag_result.metric_forward_m,
            square_result.metric_left_m - self._last_tag_result.metric_left_m,
        )
        if xy_jump > self._square_fallback_max_xy_jump_m:
            return False, f"xy_jump_{xy_jump:.2f}m"
        return True, "accepted"

    def _should_evaluate_square_fallback(self, now: float) -> bool:
        if not self._square_detection_enabled:
            return False
        if self._last_tag_detection_time <= 0.0:
            return False
        tag_age = now - self._last_tag_detection_time
        if tag_age < self._square_fallback_stale_s:
            return False
        if tag_age > self._square_fallback_prior_timeout_s:
            return False
        if now - self._last_square_eval_time < self._square_fallback_eval_period_s:
            return False
        self._last_square_eval_time = now
        return True

    def _detect_target(self, image: np.ndarray) -> Optional[TargetResult]:
        tag_result = self._detect_tag_target(image)
        if tag_result is not None:
            return tag_result
        return self._detect_square_target(image)

    def _suppress_highlights(self, image: np.ndarray) -> np.ndarray:
        if not self._highlight_suppression_enabled:
            return image
        if image.ndim == 2:
            knee = float(np.clip(self._highlight_suppression_knee, 0, 255))
            compression = float(np.clip(self._highlight_suppression_compression, 0.0, 1.0))
            max_value = float(np.clip(self._highlight_suppression_max_value, 0, 255))
            output = image.astype(np.float32)
            highlights = output > knee
            output[highlights] = knee + (output[highlights] - knee) * compression
            return np.clip(output, 0, max_value).astype(np.uint8)
        knee = float(np.clip(self._highlight_suppression_knee, 0, 255))
        compression = float(np.clip(self._highlight_suppression_compression, 0.0, 1.0))
        max_value = float(np.clip(self._highlight_suppression_max_value, 0, 255))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        value = hsv[:, :, 2].astype(np.float32)
        highlights = value > knee
        value[highlights] = knee + (value[highlights] - knee) * compression
        value = np.minimum(value, max_value)
        hsv[:, :, 2] = np.clip(value, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _detect_tag_target(self, image: np.ndarray) -> Optional[TargetResult]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        tag_binary = self._threshold_white(gray, morph_open=False)

        tags = self._detect_grid_tags(gray, tag_binary)
        return self._estimate_target_from_tags(tags)

    def _detect_square_target(self, image: np.ndarray) -> Optional[TargetResult]:
        if not self._square_detection_enabled:
            return None

        if self._square_template_detector is not None:
            square_detection = self._square_template_detector.detect(image)
            if square_detection is not None:
                return self._target_from_square_detection(square_detection)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        tag_binary = self._threshold_white(gray, morph_open=False)
        tags = self._detect_grid_tags(gray, tag_binary)
        largest_tag_area = max((tag.area_frac for tag in tags.values()), default=0.0)
        for square_threshold in self._square_threshold_values():
            square_binary = self._threshold_white(
                gray,
                threshold=square_threshold,
                morph_open=True,
            )
            square = self._detect_square(square_binary, largest_tag_area, len(tags))
            if square is not None:
                square.tag_ids = tuple(sorted(tags))
                return square

        return None

    def _target_from_square_detection(
        self,
        detection: SquareDetection,
    ) -> TargetResult:
        return TargetResult(
            cx=int(round(float(detection.cx))),
            cy=int(round(float(detection.cy))),
            yaw=float(detection.yaw),
            area_frac=float(detection.area_frac),
            corners=detection.corners.astype(np.float32),
            source=f"square_{detection.method}",
            target_id=0,
            tag_ids=(),
            metric_forward_m=float(detection.metric_forward_m),
            metric_left_m=float(detection.metric_left_m),
            range_m=float(detection.range_m),
            score=float(detection.score),
        )

    def _square_threshold_values(self) -> tuple[int, ...]:
        values = [
            self._square_binary_threshold,
            160,
            175,
            190,
            205,
        ]
        return tuple(dict.fromkeys(int(max(0, min(255, value))) for value in values))

    def _threshold_white(
        self,
        gray: np.ndarray,
        threshold: Optional[int] = None,
        morph_open: bool = True,
    ) -> np.ndarray:
        threshold_value = self._binary_white_threshold if threshold is None else int(threshold)
        binary = (gray >= threshold_value).astype(np.uint8) * 255
        if not morph_open:
            return binary
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))

    def _build_tag_board_corners(self) -> dict[int, np.ndarray]:
        inner_origin = self._outer_white_inches + self._black_border_inches
        start = inner_origin + self._inner_padding_inches
        board = {}
        for index, marker_id in enumerate(sorted(self._grid_tag_ids)):
            row, col = divmod(index, 3)
            x0 = start + col * (self._tag_size_inches + self._tag_gap_inches)
            y0 = start + row * (self._tag_size_inches + self._tag_gap_inches)
            x1 = x0 + self._tag_size_inches
            y1 = y0 + self._tag_size_inches
            board[marker_id] = (
                np.array(
                [
                    [x0, y0],
                    [x1, y0],
                    [x1, y1],
                    [x0, y1],
                ],
                dtype=np.float32,
                )
                - (self._target_size_inches / 2.0)
            ) * 0.0254
        return board

    def _detect_square(
        self,
        binary_white: np.ndarray,
        largest_tag_area_frac: float,
        visible_tag_count: int,
    ) -> Optional[TargetResult]:
        if not self._square_detection_enabled:
            return None
        if visible_tag_count > 0 and visible_tag_count < self._square_min_visible_tags:
            return None

        black_square = self._detect_black_frame_square(binary_white, largest_tag_area_frac)
        if black_square is not None:
            return black_square

        return None

    def _detect_black_frame_square(
        self,
        binary_white: np.ndarray,
        largest_tag_area_frac: float,
    ) -> Optional[TargetResult]:
        height, width = binary_white.shape[:2]
        frame_area = float(height * width)

        closed_white = cv2.morphologyEx(
            binary_white,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (71, 71)),
            iterations=2,
        )
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            closed_white,
            connectivity=8,
        )

        board = self._select_white_board_component(stats, frame_area, width, height)
        if board is None:
            return None
        board_label = int(board["label"])
        white_region = np.where(labels == board_label, 255, 0).astype(np.uint8)
        board_area = float(cv2.countNonZero(white_region))
        if board_area <= 0.0:
            return None

        black_on_board = cv2.bitwise_and(cv2.bitwise_not(binary_white), white_region)
        black_on_board = cv2.morphologyEx(
            black_on_board,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
            iterations=1,
        )
        contours, _ = cv2.findContours(
            black_on_board,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        best_quad = None
        best_area = 0.0
        best_score = -1.0
        for contour in contours:
            contour_area = float(abs(cv2.contourArea(contour)))
            if contour_area < 500.0:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0.0:
                continue
            approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
            if len(approx) != 4:
                hull = cv2.convexHull(contour)
                hull_perimeter = cv2.arcLength(hull, True)
                if hull_perimeter <= 0.0:
                    continue
                approx = cv2.approxPolyDP(hull, 0.03 * hull_perimeter, True)
            if len(approx) != 4:
                continue

            quad = self._order_corners(approx.reshape(-1, 2).astype(np.float32))
            x, y, box_w, box_h = cv2.boundingRect(quad.astype(np.int32))
            aspect = box_w / max(1.0, float(box_h))
            if not 0.65 <= aspect <= 1.45:
                continue

            area_frac = contour_area / max(1.0, frame_area)
            if area_frac < self._square_min_area_frac or area_frac > self._square_max_area_frac:
                continue

            area_ratio_to_board = contour_area / max(1.0, board_area)
            if area_ratio_to_board < 0.03:
                continue

            if (
                largest_tag_area_frac > 0.0
                and area_frac < largest_tag_area_frac * self._square_min_tag_area_ratio
            ):
                continue

            pattern_score = self._black_frame_pattern_score(binary_white, quad)
            if pattern_score is None:
                continue

            score = contour_area * (1.0 - min(abs(1.0 - aspect), 0.9)) * pattern_score
            if score > best_score:
                best_score = score
                best_area = area_frac
                best_quad = quad

        if best_quad is None:
            return None

        homography, _ = cv2.findHomography(self._black_square_board_corners, best_quad, 0)
        if homography is None:
            return None
        square_metric = self._metric_from_object_image_points(
            self._black_square_board_corners,
            best_quad,
        )
        return self._target_from_homography(
            homography,
            area_frac=best_area,
            source="square",
            target_id=0,
            tag_ids=(),
            range_m=self._range_from_square_corners(best_quad, self._black_square_size_m),
            metric_override=square_metric,
        )

    def _select_white_board_component(
        self,
        stats: np.ndarray,
        frame_area: float,
        width: int,
        height: int,
    ) -> Optional[dict[str, float]]:
        best = None
        for label in range(1, stats.shape[0]):
            x, y, component_w, component_h, area = stats[label]
            area_frac = float(area) / max(1.0, frame_area)
            if area_frac < 0.02:
                continue
            touches_edge = (
                x <= 1
                or y <= 1
                or x + component_w >= width - 1
                or y + component_h >= height - 1
            )
            aspect = float(component_w) / max(1.0, float(component_h))
            score = float(area)
            if touches_edge:
                score *= 0.35
            if not 0.55 <= aspect <= 1.85:
                score *= 0.4
            candidate = {
                "score": score,
                "label": float(label),
            }
            if best is None or score > best["score"]:
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

    def _contour_touches_image_edge(self, contour: np.ndarray) -> bool:
        x, y, width, height = cv2.boundingRect(contour)
        return (
            x <= 1
            or y <= 1
            or x + width >= self._cam_w - 1
            or y + height >= self._cam_h - 1
        )

    def _is_valid_black_frame_pattern(
        self,
        binary_white: np.ndarray,
        corners: np.ndarray,
    ) -> bool:
        return self._black_frame_pattern_score(binary_white, corners) is not None

    def _black_frame_pattern_score(
        self,
        binary_white: np.ndarray,
        corners: np.ndarray,
    ) -> Optional[float]:
        warp_size = 240
        dst = np.array(
            [
                [0.0, 0.0],
                [warp_size - 1.0, 0.0],
                [warp_size - 1.0, warp_size - 1.0],
                [0.0, warp_size - 1.0],
            ],
            dtype=np.float32,
        )
        perspective = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
        warped_white = cv2.warpPerspective(
            binary_white,
            perspective,
            (warp_size, warp_size),
            flags=cv2.INTER_NEAREST,
        )
        black_bool = warped_white < 128

        black_square_inches = self._target_size_inches - 2.0 * self._outer_white_inches
        border_px = max(
            6,
            int(round(warp_size * self._black_border_inches / black_square_inches)),
        )
        padding_px = max(
            5,
            int(round(warp_size * self._inner_padding_inches / black_square_inches)),
        )

        border_bands = (
            black_bool[:border_px, :],
            black_bool[-border_px:, :],
            black_bool[:, :border_px],
            black_bool[:, -border_px:],
        )
        border_min = min(float(np.mean(band)) for band in border_bands)
        if border_min < 0.62:
            return None

        moat_start = border_px + 2
        moat_stop = min(warp_size // 3, border_px + padding_px + 4)
        if moat_stop <= moat_start + 2:
            return None
        core = slice(border_px + 4, warp_size - border_px - 4)
        white_moats = (
            black_bool[moat_start:moat_stop, core],
            black_bool[warp_size - moat_stop : warp_size - moat_start, core],
            black_bool[core, moat_start:moat_stop],
            black_bool[core, warp_size - moat_stop : warp_size - moat_start],
        )
        moat_max = max(float(np.mean(moat)) for moat in white_moats)
        if moat_max > 0.30:
            return None

        inner = black_bool[moat_stop : warp_size - moat_stop, moat_stop : warp_size - moat_stop]
        inner_black_ratio = float(np.mean(inner))
        if not 0.05 <= inner_black_ratio <= 0.65:
            return None
        return border_min + (1.0 - moat_max) + 0.25 * inner_black_ratio

    def _detect_grid_tags(
        self,
        gray: np.ndarray,
        binary_white: np.ndarray,
    ) -> dict[int, TargetResult]:
        results: dict[int, TargetResult] = {}
        for method_name, tag_image in self._tag_detection_images(gray, binary_white):
            corners, ids, _ = cv2.aruco.detectMarkers(
                tag_image,
                self._aruco_dict,
                parameters=self._aruco_params,
            )
            self._merge_tag_detections(results, corners, ids, method_name)
            if len(results) >= 4:
                break
        return results

    def _tag_detection_images(
        self,
        gray: np.ndarray,
        binary_white: np.ndarray,
    ):
        yield "white_threshold", binary_white
        yield "gray", gray
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=17, sigmaY=17)
        flat = cv2.divide(gray, blurred, scale=128)
        yield "flat", flat
        yield "flat_clahe", self._clahe.apply(flat)
        yield "clahe", self._clahe.apply(gray)

    def _merge_tag_detections(
        self,
        results: dict[int, TargetResult],
        corners,
        ids,
        method_name: str,
    ) -> None:
        if ids is None:
            return
        for i, marker_id_raw in enumerate(ids.flatten()):
            marker_id = int(marker_id_raw)
            if marker_id not in self._grid_tag_ids:
                continue
            pts = corners[i][0].astype(np.float32)
            area_frac = abs(cv2.contourArea(pts)) / max(1.0, self._frame_area)
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            dx = pts[1][0] - pts[0][0]
            dy = pts[1][1] - pts[0][1]
            yaw = math.atan2(dy, dx)
            result = TargetResult(
                cx=cx,
                cy=cy,
                yaw=yaw,
                area_frac=area_frac,
                corners=pts,
                source=f"tag_{marker_id}_{method_name}",
                target_id=marker_id,
                tag_ids=(marker_id,),
            )
            previous = results.get(marker_id)
            if previous is None or result.area_frac > previous.area_frac:
                results[marker_id] = result

    def _estimate_target_from_tags(
        self,
        tags: dict[int, TargetResult],
    ) -> Optional[TargetResult]:
        if not tags:
            return None

        board_points = []
        image_points = []
        point_marker_ids = []
        for marker_id, result in sorted(tags.items()):
            board_corners = self._tag_board_corners.get(marker_id)
            if board_corners is None or result.corners is None:
                continue
            board_points.extend(board_corners)
            image_points.extend(result.corners.astype(np.float32))
            point_marker_ids.extend([marker_id] * len(board_corners))

        if len(board_points) < 4:
            return None

        board_arr = np.asarray(board_points, dtype=np.float32)
        image_arr = np.asarray(image_points, dtype=np.float32)
        method = 0 if len(board_points) == 4 else cv2.RANSAC
        homography, mask = cv2.findHomography(board_arr, image_arr, method)
        if homography is None:
            return None
        pose_board_arr = board_arr
        pose_image_arr = image_arr
        pose_tag_ids = set(tags)
        if mask is not None and len(board_arr) > 4:
            inliers = mask.reshape(-1).astype(bool)
            if int(np.count_nonzero(inliers)) >= 4:
                pose_board_arr = board_arr[inliers]
                pose_image_arr = image_arr[inliers]
                pose_tag_ids = {
                    int(marker_id)
                    for marker_id, keep in zip(point_marker_ids, inliers)
                    if keep
                } or pose_tag_ids
        primary_id = (
            self._center_tag_id
            if self._center_tag_id in tags
            else max(tags.items(), key=lambda item: item[1].area_frac)[0]
        )
        target_corners = self._project_board_points(homography, self._target_board_corners)
        area_frac = abs(cv2.contourArea(target_corners)) / max(1.0, self._frame_area)
        pose_metric = self._metric_from_object_image_points(pose_board_arr, pose_image_arr)
        range_m = pose_metric[2] if pose_metric is not None else self._range_from_tags(tags)
        return self._target_from_homography(
            homography,
            area_frac=float(min(area_frac, 1.0)),
            source="tag_homography",
            target_id=int(primary_id),
            tag_ids=tuple(sorted(pose_tag_ids)),
            range_m=range_m,
            metric_override=pose_metric,
        )

    def _target_from_homography(
        self,
        homography: np.ndarray,
        area_frac: float,
        source: str,
        target_id: int,
        tag_ids: tuple[int, ...],
        range_m: Optional[float] = None,
        metric_override: Optional[tuple[float, float, float]] = None,
    ) -> TargetResult:
        center = self._project_board_points(
            homography,
            np.array([[0.0, 0.0]], dtype=np.float32),
        )[0]
        target_corners = self._project_board_points(homography, self._target_board_corners)
        x_axis = self._project_board_points(
            homography,
            np.array([[0.0, 0.0], [0.10, 0.0]], dtype=np.float32),
        )
        yaw = math.atan2(x_axis[1][1] - x_axis[0][1], x_axis[1][0] - x_axis[0][0])
        metric_forward_m, metric_left_m, homography_range_m = self._metric_from_homography(homography)
        if metric_override is not None:
            metric_forward_m, metric_left_m, pose_range_m = metric_override
            if pose_range_m > 0.0:
                range_m = pose_range_m
        if range_m is None or range_m <= 0.0:
            range_m = homography_range_m
        return TargetResult(
            cx=int(round(float(center[0]))),
            cy=int(round(float(center[1]))),
            yaw=yaw,
            area_frac=area_frac,
            corners=target_corners,
            source=source,
            target_id=target_id,
            tag_ids=tag_ids,
            metric_forward_m=metric_forward_m,
            metric_left_m=metric_left_m,
            range_m=range_m,
        )

    def _range_from_tags(self, tags: dict[int, TargetResult]) -> float:
        ranges = [
            self._range_from_square_corners(result.corners, self._tag_size_m)
            for result in tags.values()
            if result.corners is not None
        ]
        valid_ranges = [range_m for range_m in ranges if range_m > 0.0]
        if not valid_ranges:
            return 0.0
        return float(np.median(valid_ranges))

    def _range_from_square_corners(
        self,
        corners: Optional[np.ndarray],
        physical_side_m: float,
    ) -> float:
        if corners is None or physical_side_m <= 0.0:
            return 0.0
        pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
        if len(pts) != 4:
            return 0.0
        side_lengths = [
            float(np.linalg.norm(pts[(index + 1) % 4] - pts[index]))
            for index in range(4)
        ]
        valid_sides = [side for side in side_lengths if side > 1.0]
        if not valid_sides:
            return 0.0
        side_px = float(np.median(valid_sides))
        return float(self._camera_focal_px * physical_side_m / side_px)

    @staticmethod
    def _project_board_points(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
        projected = cv2.perspectiveTransform(points.reshape(-1, 1, 2), homography)
        return projected.reshape(-1, 2).astype(np.float32)

    def _metric_from_homography(self, homography: np.ndarray) -> tuple[float, float, float]:
        try:
            inverse = np.linalg.inv(homography)
        except np.linalg.LinAlgError:
            return 0.0, 0.0, 0.0

        image_center = np.array([[self._cam_w / 2.0, self._cam_h / 2.0]], dtype=np.float32)
        board_at_camera = self._project_board_points(inverse, image_center)[0]
        delta_px = 8.0
        image_up = np.array([[self._cam_w / 2.0, self._cam_h / 2.0 - delta_px]], dtype=np.float32)
        image_left = np.array([[self._cam_w / 2.0 - delta_px, self._cam_h / 2.0]], dtype=np.float32)
        board_up = self._project_board_points(inverse, image_up)[0] - board_at_camera
        board_left = self._project_board_points(inverse, image_left)[0] - board_at_camera

        up_norm = float(np.linalg.norm(board_up))
        left_norm = float(np.linalg.norm(board_left))
        if up_norm <= 1e-6 or left_norm <= 1e-6:
            return 0.0, 0.0, 0.0

        basis = np.column_stack((board_up / up_norm, board_left / left_norm))
        board_error = -board_at_camera.astype(np.float64)
        try:
            forward_m, left_m = np.linalg.solve(basis.astype(np.float64), board_error)
        except np.linalg.LinAlgError:
            forward_m, left_m = np.linalg.lstsq(basis.astype(np.float64), board_error, rcond=None)[0]

        scale_points = self._project_board_points(
            homography,
            np.array([[0.0, 0.0], [0.10, 0.0], [0.0, 0.10]], dtype=np.float32),
        )
        px_per_m_x = float(np.linalg.norm(scale_points[1] - scale_points[0]) / 0.10)
        px_per_m_y = float(np.linalg.norm(scale_points[2] - scale_points[0]) / 0.10)
        px_per_m = 0.5 * (px_per_m_x + px_per_m_y)
        range_m = self._camera_focal_px / px_per_m if px_per_m > 1e-6 else 0.0
        return float(forward_m), float(left_m), float(range_m)

    def _metric_from_object_image_points(
        self,
        board_points_m: np.ndarray,
        image_points_px: np.ndarray,
    ) -> Optional[tuple[float, float, float]]:
        board_arr = np.asarray(board_points_m, dtype=np.float64)
        image_arr = np.asarray(image_points_px, dtype=np.float64)
        if board_arr.ndim != 2 or image_arr.ndim != 2 or len(board_arr) < 4:
            return None
        if board_arr.shape[1] == 2:
            object_points = np.column_stack(
                (board_arr, np.zeros((len(board_arr),), dtype=np.float64))
            )
        elif board_arr.shape[1] == 3:
            object_points = board_arr
        else:
            return None

        candidates = []

        def add_candidate(rvec: np.ndarray, tvec: np.ndarray, label: str, error=None) -> None:
            tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
            if not np.all(np.isfinite(tvec)) or tvec[2] <= 0.0:
                return
            if error is None:
                projected, _ = cv2.projectPoints(
                    object_points,
                    rvec,
                    tvec.reshape(3, 1),
                    self._camera_matrix,
                    self._dist_coeffs,
                )
                reprojection = projected.reshape(-1, 2) - image_arr
                error = float(np.mean(np.linalg.norm(reprojection, axis=1)))
            candidates.append((float(error), label, tvec))

        if hasattr(cv2, "solvePnPGeneric"):
            try:
                ok, rvecs, tvecs, errors = cv2.solvePnPGeneric(
                    object_points,
                    image_arr,
                    self._camera_matrix,
                    self._dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE,
                )
                if ok:
                    if errors is None:
                        errors = [None] * len(rvecs)
                    for rvec, tvec, error in zip(rvecs, tvecs, errors):
                        if error is not None:
                            error = float(np.asarray(error).reshape(-1)[0])
                        add_candidate(rvec, tvec, "ippe", error)
            except cv2.error:
                pass

        best_ippe_error = min((item[0] for item in candidates), default=float("inf"))
        if best_ippe_error > 3.0:
            try:
                ok, rvec, tvec = cv2.solvePnP(
                    object_points,
                    image_arr,
                    self._camera_matrix,
                    self._dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if ok:
                    add_candidate(rvec, tvec, "iterative")
            except cv2.error:
                pass

        if not candidates:
            return None

        _, _, tvec = min(candidates, key=lambda item: item[0])
        # OpenCV camera coordinates: +x image-right, +y image-down, +z optical axis.
        # Controller convention: +forward is image-up, +left is image-left.
        return float(-tvec[1]), float(-tvec[0]), float(tvec[2])

    def _publish_detection_debug(
        self,
        header,
        result: Optional[TargetResult],
    ) -> None:
        guidance = TwistStamped()
        guidance.header = header
        if result is None:
            self._target_id_pub.publish(Int32(data=-1))
            self._target_source_pub.publish(String(data="none"))
            self._visible_tag_ids_pub.publish(Int32MultiArray(data=[]))
            self._guidance_velocity_pub.publish(guidance)
            return

        self._target_id_pub.publish(Int32(data=int(result.target_id)))
        self._target_source_pub.publish(String(data=result.source))
        self._visible_tag_ids_pub.publish(
            Int32MultiArray(data=[int(marker_id) for marker_id in result.tag_ids])
        )
        guidance.twist.linear.x = float(result.desired_vx)
        guidance.twist.linear.y = float(result.desired_vy)
        self._guidance_velocity_pub.publish(guidance)

    def _publish_square_debug(
        self,
        header,
        result: Optional[TargetResult],
    ) -> None:
        self._square_detected_pub.publish(Bool(data=result is not None))
        if result is None:
            if self._square_method_pub.get_subscription_count() > 0:
                self._square_method_pub.publish(String(data="none"))
            self._square_score_pub.publish(Float32(data=0.0))
            if self._square_corners_pub.get_subscription_count() > 0:
                self._square_corners_pub.publish(Float32MultiArray(data=[]))
            return

        pixel = PointStamped()
        pixel.header = header
        pixel.point.x = float(result.cx)
        pixel.point.y = float(result.cy)
        pixel.point.z = float(result.yaw)
        self._square_pixel_pub.publish(pixel)

        metric = PointStamped()
        metric.header = header
        metric.point.x = float(result.metric_forward_m)
        metric.point.y = float(result.metric_left_m)
        metric.point.z = float(result.range_m)
        self._square_metric_pub.publish(metric)

        self._square_range_pub.publish(Float32(data=float(result.range_m)))
        if self._square_method_pub.get_subscription_count() > 0:
            self._square_method_pub.publish(String(data=result.source))
        self._square_score_pub.publish(Float32(data=float(result.score)))
        if self._square_corners_pub.get_subscription_count() > 0:
            corners = []
            if result.corners is not None:
                corners = result.corners.reshape(-1).astype(float).tolist()
            self._square_corners_pub.publish(Float32MultiArray(data=corners))

    @staticmethod
    def _order_corners(corners: np.ndarray) -> np.ndarray:
        pts = corners.astype(np.float32)
        sums = pts.sum(axis=1)
        diffs = pts[:, 0] - pts[:, 1]
        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = pts[np.argmin(sums)]
        ordered[2] = pts[np.argmax(sums)]
        ordered[1] = pts[np.argmax(diffs)]
        ordered[3] = pts[np.argmin(diffs)]
        return ordered

    def _build_annotated_image(
        self,
        image: np.ndarray,
        result: Optional[TargetResult],
    ) -> np.ndarray:
        annotated = (
            cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if image.ndim == 2
            else image.copy()
        )
        h, w = annotated.shape[:2]
        frame_center = (w // 2, h // 2)
        cv2.circle(annotated, frame_center, 6, (0, 255, 0), -1)

        if result is None:
            cv2.putText(
                annotated,
                "target=(none)",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            return annotated

        if result.corners is not None:
            corners = result.corners.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [corners], True, (255, 0, 0), 2)

        target = (int(result.cx), int(result.cy))
        cv2.circle(annotated, target, 6, (0, 0, 255), -1)
        cv2.arrowedLine(annotated, frame_center, target, (0, 0, 255), 2)
        label = (
            f"{result.source} ids={list(result.tag_ids)} "
            f"target=({result.cx},{result.cy}) yaw={result.yaw:.2f} "
            f"fwd={result.metric_forward_m:.2f} left={result.metric_left_m:.2f} "
            f"range={result.range_m:.2f}"
        )
        cv2.putText(
            annotated,
            label,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated

    def _message_age_seconds(self, msg: Image) -> float:
        return self._header_age_seconds(msg.header)

    def _header_age_seconds(self, header) -> float:
        stamp = header.stamp
        if stamp.sec == 0 and stamp.nanosec == 0:
            return 0.0

        now = self.get_clock().now().nanoseconds
        then = stamp.sec * 1_000_000_000 + stamp.nanosec
        return max(0.0, (now - then) / 1_000_000_000.0)

    def _report_stats(self):
        since_last_image_sec = (
            time.perf_counter() - self._last_image_time if self._last_image_time else None
        )
        if self._frames_processed == 0:
            if since_last_image_sec is not None and since_last_image_sec > 1.0:
                self.get_logger().warning(
                    f"no images received on detector input for {since_last_image_sec:.1f}s"
                )
            return

        avg_proc_ms = 1000.0 * self._processing_time_sum / self._frames_processed
        avg_age_ms = 1000.0 * self._message_age_sum / self._frames_processed
        msg = (
            f"perf recv={self._frames_received} proc={self._frames_processed} "
            f"detect={self._frames_detected} avg_proc_ms={avg_proc_ms:.1f} "
            f"avg_age_ms={avg_age_ms:.1f}"
        )
        if since_last_image_sec is not None:
            msg += f" last_image_sec={since_last_image_sec:.2f}"
        self.get_logger().info(msg)
        self._frames_received = 0
        self._frames_processed = 0
        self._frames_detected = 0
        self._processing_time_sum = 0.0
        self._message_age_sum = 0.0

    def destroy_node(self):
        self._worker_stop = True
        with self._mailbox_condition:
            self._mailbox_condition.notify_all()
        with self._square_condition:
            self._square_condition.notify_all()
        current = threading.current_thread()
        for worker in (self._processing_thread, self._square_thread):
            if worker is not None and worker is not current:
                worker.join(timeout=2.0)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TargetDetector()
    node._start_workers()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
