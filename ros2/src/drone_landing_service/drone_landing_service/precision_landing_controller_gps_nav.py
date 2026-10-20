#!/usr/bin/env python3
"""GPS-assisted precision landing controller using metric target estimates.

This mode merges two flows:

* /landing/start: fly a global GPS setpoint at a fixed relative altitude until
  the AprilTag/square target estimator publishes a fresh metric target, then
  immediately switch into visual ALIGN and DESCEND.
* /landing/begin_landing: skip GPS and start the same visual ALIGN/DESCEND
  controller from the current camera view.

/landing/kill_offboard is the explicit escape hatch back to manual/POSCTL.
"""

from __future__ import annotations

import math
from enum import Enum

import rclpy
from geometry_msgs.msg import PointStamped, TwistStamped
from mavros_msgs.msg import GlobalPositionTarget, RCOut, State as MavrosState
from mavros_msgs.srv import SetMode
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Bool, Float32, String
from std_srvs.srv import Trigger


EARTH_RADIUS_M = 6378137.0


class State(Enum):
    MANUAL = "manual"
    GPS_NAV = "gps_nav"
    GPS_SPIRAL = "gps_spiral"
    ALIGN = "align"
    DESCEND = "descend"
    DISARMED = "disarmed"


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def step_scalar(current: float, target: float, max_step: float) -> float:
    if max_step <= 0.0:
        return target
    delta = target - current
    if abs(delta) <= max_step:
        return target
    return current + math.copysign(max_step, delta)


def step_latlon(
    lat: float,
    lon: float,
    target_lat: float,
    target_lon: float,
    max_step_m: float,
) -> tuple[float, float]:
    distance_m = haversine_m(lat, lon, target_lat, target_lon)
    if max_step_m <= 0.0 or distance_m <= max_step_m or distance_m <= 1e-6:
        return target_lat, target_lon
    ratio = max_step_m / distance_m
    return lat + (target_lat - lat) * ratio, lon + (target_lon - lon) * ratio


class PrecisionLandingControllerGpsNav(Node):
    def __init__(self):
        super().__init__("precision_landing_controller_gps_nav")

        self.declare_parameter("target_metric_topic", "/landing/target_metric")
        self.declare_parameter("target_range_topic", "/landing/target_range_m")
        self.declare_parameter("target_detected_topic", "/landing/target_detected")
        self.declare_parameter("metric_error_topic", "/landing/metric_error")
        self.declare_parameter("drone_global_position_topic", "/landing/drone_position_global")
        self.declare_parameter("boat_position_global_topic", "/landing/boat_position_global")
        self.declare_parameter("drone_gps_setpoint_topic", "/mavros/setpoint_raw/global")
        self.declare_parameter("drone_speed_setpoint_topic", "/landing/cmd/auto")
        self.declare_parameter("drone_speed_topic", "/landing/auto_velocity_setpoint")
        self.declare_parameter("kp", 0.90)
        self.declare_parameter("metric_vx_sign", 1.0)
        self.declare_parameter("metric_vy_sign", 1.0)
        self.declare_parameter("velocity_mav_frame", "BODY_NED")
        self.declare_parameter("gps_nav_altitude_m", 5.0)
        self.declare_parameter("gps_nav_horizontal_setpoint_speed_mps", 1.0)
        self.declare_parameter("gps_nav_vertical_setpoint_speed_mps", 0.75)
        self.declare_parameter("gps_nav_arrival_radius_m", 0.5)
        self.declare_parameter("gps_nav_altitude_tolerance_m", 0.75)
        self.declare_parameter("gps_spiral_enabled", True)
        self.declare_parameter("gps_spiral_start_radius_m", 0.4)
        self.declare_parameter("gps_spiral_max_radius_m", 10.0)
        self.declare_parameter("gps_spiral_radial_speed_mps", 0.25)
        self.declare_parameter("gps_spiral_tangential_speed_mps", 1.50)
        self.declare_parameter("max_horizontal_speed", 0.25)
        self.declare_parameter("min_horizontal_speed", 0.0)
        self.declare_parameter("max_vertical_speed", 0.30)
        self.declare_parameter("descent_speed", 0.15)
        self.declare_parameter("align_metric_tolerance_m", 0.30)
        self.declare_parameter("align_stable_s", 1.0)
        self.declare_parameter("target_metric_timeout_s", 1.00)
        self.declare_parameter("align_missed_sample_limit", 15)
        self.declare_parameter("descend_missed_sample_limit", 15)
        self.declare_parameter("touchdown_range_m", 0.60)
        self.declare_parameter("touchdown_xy_tolerance_m", 0.12)
        self.declare_parameter("offboard_warmup_setpoints", 20)
        self.declare_parameter("mode_request_interval_s", 1.0)
        self.declare_parameter("offboard_mode", "OFFBOARD")
        self.declare_parameter("position_mode", "POSCTL")
        self.declare_parameter("start_on_launch", False)

        metric_topic = self.get_parameter("target_metric_topic").value
        range_topic = self.get_parameter("target_range_topic").value
        detected_topic = self.get_parameter("target_detected_topic").value
        metric_error_topic = self.get_parameter("metric_error_topic").value
        drone_topic = self.get_parameter("drone_global_position_topic").value
        boat_topic = self.get_parameter("boat_position_global_topic").value
        gps_setpoint_topic = self.get_parameter("drone_gps_setpoint_topic").value
        mavros_vel_topic = self.get_parameter("drone_speed_setpoint_topic").value
        landing_vel_topic = self.get_parameter("drone_speed_topic").value

        self._kp = float(self.get_parameter("kp").value)
        self._metric_vx_sign = 1.0 if float(self.get_parameter("metric_vx_sign").value) >= 0.0 else -1.0
        self._metric_vy_sign = 1.0 if float(self.get_parameter("metric_vy_sign").value) >= 0.0 else -1.0
        self._velocity_mav_frame = str(self.get_parameter("velocity_mav_frame").value).strip() or "BODY_NED"
        self._gps_target_alt = float(self.get_parameter("gps_nav_altitude_m").value)
        self._gps_horizontal_speed = abs(float(self.get_parameter("gps_nav_horizontal_setpoint_speed_mps").value))
        self._gps_vertical_speed = abs(float(self.get_parameter("gps_nav_vertical_setpoint_speed_mps").value))
        self._gps_arrival_radius_m = abs(float(self.get_parameter("gps_nav_arrival_radius_m").value))
        self._gps_altitude_tolerance_m = abs(float(self.get_parameter("gps_nav_altitude_tolerance_m").value))
        self._gps_spiral_enabled = bool(self.get_parameter("gps_spiral_enabled").value)
        self._gps_spiral_start_radius_m = abs(float(self.get_parameter("gps_spiral_start_radius_m").value))
        self._gps_spiral_max_radius_m = abs(float(self.get_parameter("gps_spiral_max_radius_m").value))
        self._gps_spiral_radial_speed_mps = abs(float(self.get_parameter("gps_spiral_radial_speed_mps").value))
        self._gps_spiral_tangential_speed_mps = abs(float(self.get_parameter("gps_spiral_tangential_speed_mps").value))
        self._vxy_max = abs(float(self.get_parameter("max_horizontal_speed").value))
        self._vxy_min = min(abs(float(self.get_parameter("min_horizontal_speed").value)), self._vxy_max)
        self._vz_max = abs(float(self.get_parameter("max_vertical_speed").value))
        self._descent_speed = min(abs(float(self.get_parameter("descent_speed").value)), self._vz_max)
        self._align_tol_m = float(self.get_parameter("align_metric_tolerance_m").value)
        self._align_stable_s = float(self.get_parameter("align_stable_s").value)
        self._target_timeout_s = float(self.get_parameter("target_metric_timeout_s").value)
        self._align_missed_limit = int(self.get_parameter("align_missed_sample_limit").value)
        self._descend_missed_limit = int(self.get_parameter("descend_missed_sample_limit").value)
        self._touchdown_range_m = float(self.get_parameter("touchdown_range_m").value)
        self._touchdown_xy_tol_m = float(self.get_parameter("touchdown_xy_tolerance_m").value)
        self._offboard_warmup_setpoints = int(self.get_parameter("offboard_warmup_setpoints").value)
        self._mode_request_interval_s = float(self.get_parameter("mode_request_interval_s").value)
        self._offboard_mode = str(self.get_parameter("offboard_mode").value).strip().upper() or "OFFBOARD"
        self._position_mode = str(self.get_parameter("position_mode").value).strip().upper() or "POSCTL"
        self._start_on_launch = bool(self.get_parameter("start_on_launch").value)

        self._state = State.MANUAL
        self._landing_started = False
        self._return_to_gps_on_loss = False
        self._target_metric: PointStamped | None = None
        self._target_metric_time = 0.0
        self._target_detected = False
        self._target_range_m: float | None = None
        self._target_range_time = 0.0
        self._drone_fix: NavSatFix | None = None
        self._boat_fix: NavSatFix | None = None
        self._cmd_lat: float | None = None
        self._cmd_lon: float | None = None
        self._cmd_rel_alt_m: float | None = None
        self._last_tick_s: float | None = None
        self._spiral_start_s: float | None = None
        self._spiral_angle_rad = 0.0
        self._spiral_radius_m = 0.0
        self._spiral_center_lat: float | None = None
        self._spiral_center_lon: float | None = None
        self._spiral_center_alt_m: float | None = None
        self._mavros_mode = ""
        self._mavros_connected = False
        self._mavros_armed = False
        self._motors_low_count = 0
        self._setpoint_count = 0
        self._last_mode_request_s = 0.0
        self._requested_mode = ""
        self._qgc_takeover = False
        self._align_stable_start_time = 0.0
        self._align_missed_count = 0
        self._descend_missed_count = 0
        self._last_error = (0.0, 0.0)
        self._last_descend_cmd = (0.0, 0.0, -self._descent_speed, 0.0)
        self._velocity_frame_set = False
        self._last_velocity_frame_request_time = 0.0

        takeover_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(PointStamped, metric_topic, self._on_metric, 10)
        self.create_subscription(Float32, range_topic, self._on_range, 10)
        self.create_subscription(Bool, detected_topic, self._on_detected, 10)
        self.create_subscription(NavSatFix, drone_topic, self._on_drone_pos, qos_profile_sensor_data)
        self.create_subscription(NavSatFix, boat_topic, self._on_boat_pos, 10)
        self.create_subscription(Bool, "/landing/qgc_takeover", self._on_qgc_takeover, takeover_qos)
        self.create_subscription(MavrosState, "/mavros/state", self._on_mavros_state, 10)
        self.create_subscription(RCOut, "/mavros/rc/out", self._on_rc_out, 10)

        self._gps_setpoint_pub = self.create_publisher(GlobalPositionTarget, gps_setpoint_topic, 10)
        self._mavros_vel_pub = self.create_publisher(TwistStamped, mavros_vel_topic, 10)
        self._landing_vel_pub = self.create_publisher(TwistStamped, landing_vel_topic, 10)
        self._metric_error_pub = self.create_publisher(PointStamped, metric_error_topic, 10)
        self._phase_pub = self.create_publisher(String, "/landing/phase", 10)
        self._owner_set_pub = self.create_publisher(String, "/landing/control_owner/set", 10)

        self._set_mode_client = self.create_client(SetMode, "/mavros/set_mode")
        self._set_velocity_frame_client = self.create_client(
            SetParameters,
            "/mavros/setpoint_velocity/set_parameters",
        )

        self.create_service(Trigger, "/landing/start", self._handle_start)
        self.create_service(Trigger, "/landing/start_gps_landing", self._handle_start_gps_landing)
        self.create_service(Trigger, "/landing/start_landing", self._handle_start_landing)
        self.create_service(Trigger, "/landing/start_manual_landing", self._handle_start_landing)
        self.create_service(Trigger, "/landing/begin_landing", self._handle_begin_landing)
        self.create_service(Trigger, "/landing/kill_offboard", self._handle_kill_offboard)
        self.create_service(Trigger, "/gps_spiral", self._handle_gps_spiral)
        self.create_timer(0.1, self._tick)
        self._publish_phase()
        if self._start_on_launch:
            self._start_gps_nav()
            self.get_logger().info("gps-based landing auto-started")

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _on_metric(self, msg: PointStamped) -> None:
        self._target_metric = msg
        self._target_metric_time = self._now_s()

    def _on_range(self, msg: Float32) -> None:
        value = float(msg.data)
        if math.isfinite(value) and value > 0.0:
            self._target_range_m = value
            self._target_range_time = self._now_s()

    def _on_detected(self, msg: Bool) -> None:
        self._target_detected = bool(msg.data)
        if not self._target_detected:
            self._target_metric = None
            self._target_metric_time = 0.0

    def _on_drone_pos(self, msg: NavSatFix) -> None:
        self._drone_fix = msg

    def _on_boat_pos(self, msg: NavSatFix) -> None:
        self._boat_fix = msg

    def _on_mavros_state(self, msg: MavrosState) -> None:
        was_armed = self._mavros_armed
        self._mavros_mode = msg.mode
        self._mavros_connected = bool(msg.connected)
        self._mavros_armed = bool(msg.armed)
        if self._mavros_armed and not was_armed and self._state == State.DISARMED:
            self.get_logger().info("vehicle re-armed: resetting landing controller to manual")
            self._state = State.MANUAL
            self._landing_started = False
            self._return_to_gps_on_loss = False
            self._motors_low_count = 0
            self._setpoint_count = 0
            self._align_missed_count = 0
            self._descend_missed_count = 0
            self._align_stable_start_time = 0.0
            self._reset_gps_ramp()
            self._set_control_owner("teleop")
            self._publish_phase()

    def _on_qgc_takeover(self, msg: Bool) -> None:
        self._qgc_takeover = bool(msg.data)
        if self._qgc_takeover:
            self.get_logger().warn("QGC takeover active: killing gps-based landing")
            self._kill_to_manual(request_position=False)

    def _on_rc_out(self, msg: RCOut) -> None:
        low = sum(1 for value in list(msg.channels[:4]) if value < 500)
        self._motors_low_count = self._motors_low_count + 1 if low >= 2 else 0

    def _has_metric(self) -> bool:
        return self._target_metric is not None and self._now_s() - self._target_metric_time <= self._target_timeout_s

    def _has_range(self) -> bool:
        return self._target_range_m is not None and self._now_s() - self._target_range_time <= self._target_timeout_s

    def _has_target(self) -> bool:
        return self._target_detected and self._has_metric()

    def _rel_alt_m(self) -> float | None:
        if self._drone_fix is None:
            return None
        value = float(self._drone_fix.altitude)
        return value if math.isfinite(value) else None

    def _metric_errors(self) -> tuple[float, float, float]:
        if self._target_metric is None:
            return 0.0, 0.0, 0.0
        return (
            float(self._target_metric.point.x),
            float(self._target_metric.point.y),
            float(self._target_metric.point.z),
        )

    def _range_for_metric_error(self) -> float:
        if self._has_range() and self._target_range_m is not None:
            return float(self._target_range_m)
        return 0.0

    def _publish_metric_error(self, ex_m: float, ey_m: float, yaw_err: float) -> None:
        self._last_error = (float(ex_m), float(ey_m))
        msg = PointStamped()
        if self._target_metric is not None:
            msg.header = self._target_metric.header
        else:
            msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x = float(ex_m)
        msg.point.y = float(ey_m)
        msg.point.z = self._range_for_metric_error()
        self._metric_error_pub.publish(msg)

    def _axis_cmd(self, error_m: float) -> float:
        raw = self._clamp(self._kp * error_m, -self._vxy_max, self._vxy_max)
        if self._vxy_min <= 0.0 or abs(raw) >= self._vxy_min or raw == 0.0:
            return raw
        return self._vxy_min if raw > 0.0 else -self._vxy_min

    def _metric_cmd(self, ex_m: float, ey_m: float, z: float) -> tuple[float, float, float, float]:
        vx = self._metric_vx_sign * self._axis_cmd(ex_m)
        vy = self._metric_vy_sign * self._axis_cmd(ey_m)
        return vx, vy, z, 0.0

    def _aligned(self, ex_m: float, ey_m: float) -> bool:
        return math.hypot(ex_m, ey_m) <= self._align_tol_m

    def _near_touchdown(self) -> bool:
        if not self._has_range() or self._target_range_m is None:
            return False
        if float(self._target_range_m) > self._touchdown_range_m:
            return False
        ex_m, ey_m = self._last_error
        return abs(ex_m) <= self._touchdown_xy_tol_m and abs(ey_m) <= self._touchdown_xy_tol_m

    def _descend_loss_is_tolerable(self) -> bool:
        return math.hypot(self._last_error[0], self._last_error[1]) <= self._align_tol_m

    def _publish_cmd(self, x: float, y: float, z: float, yaw_rate: float) -> None:
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.twist.linear.x = float(x)
        cmd.twist.linear.y = float(y)
        cmd.twist.linear.z = float(z)
        cmd.twist.angular.z = float(yaw_rate)
        self._mavros_vel_pub.publish(cmd)
        self._landing_vel_pub.publish(cmd)

    def _publish_gps_setpoint(self) -> bool:
        if self._boat_fix is None or self._drone_fix is None:
            self.get_logger().warn("GPS_NAV waiting for drone/boat GPS fix", throttle_duration_sec=2.0)
            return False
        now_s = self._now_s()
        dt = 0.0 if self._last_tick_s is None else max(0.0, min(now_s - self._last_tick_s, 0.5))
        self._last_tick_s = now_s

        rel_alt = self._rel_alt_m()
        if self._cmd_lat is None:
            self._cmd_lat = float(self._drone_fix.latitude)
        if self._cmd_lon is None:
            self._cmd_lon = float(self._drone_fix.longitude)
        if self._cmd_rel_alt_m is None:
            self._cmd_rel_alt_m = rel_alt if rel_alt is not None else self._gps_target_alt

        self._cmd_rel_alt_m = step_scalar(
            self._cmd_rel_alt_m,
            self._gps_target_alt,
            self._gps_vertical_speed * dt,
        )
        self._cmd_lat, self._cmd_lon = step_latlon(
            self._cmd_lat,
            self._cmd_lon,
            float(self._boat_fix.latitude),
            float(self._boat_fix.longitude),
            self._gps_horizontal_speed * dt,
        )

        msg = GlobalPositionTarget()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.coordinate_frame = GlobalPositionTarget.FRAME_GLOBAL_REL_ALT
        msg.type_mask = (
            GlobalPositionTarget.IGNORE_VX
            | GlobalPositionTarget.IGNORE_VY
            | GlobalPositionTarget.IGNORE_VZ
            | GlobalPositionTarget.IGNORE_AFX
            | GlobalPositionTarget.IGNORE_AFY
            | GlobalPositionTarget.IGNORE_AFZ
            | GlobalPositionTarget.IGNORE_YAW
            | GlobalPositionTarget.IGNORE_YAW_RATE
        )
        msg.latitude = float(self._cmd_lat)
        msg.longitude = float(self._cmd_lon)
        msg.altitude = float(self._cmd_rel_alt_m)
        self._gps_setpoint_pub.publish(msg)
        return True

    def _publish_gps_spiral_setpoint(self) -> bool:
        center_lat = self._spiral_center_lat
        center_lon = self._spiral_center_lon
        center_alt = self._spiral_center_alt_m
        if center_lat is None or center_lon is None:
            if self._boat_fix is None:
                self.get_logger().warn("GPS_SPIRAL waiting for spiral center GPS fix", throttle_duration_sec=2.0)
                return False
            center_lat = float(self._boat_fix.latitude)
            center_lon = float(self._boat_fix.longitude)
        if center_alt is None:
            center_alt = self._gps_target_alt
        if center_lat is None or center_lon is None:
            return False
        now_s = self._now_s()
        dt = 0.0 if self._last_tick_s is None else max(0.0, min(now_s - self._last_tick_s, 0.5))
        self._last_tick_s = now_s
        if self._spiral_start_s is None:
            self._spiral_start_s = now_s
            self._spiral_radius_m = min(self._gps_spiral_start_radius_m, self._gps_spiral_max_radius_m)
            self._spiral_angle_rad = 0.0

        self._spiral_radius_m = min(
            self._gps_spiral_max_radius_m,
            max(self._gps_spiral_start_radius_m, self._spiral_radius_m + self._gps_spiral_radial_speed_mps * dt),
        )
        if self._spiral_radius_m > 1e-3:
            angular_rate = self._gps_spiral_tangential_speed_mps / self._spiral_radius_m
            self._spiral_angle_rad = (self._spiral_angle_rad + angular_rate * dt) % (2.0 * math.pi)

        north_m = self._spiral_radius_m * math.sin(self._spiral_angle_rad)
        east_m = self._spiral_radius_m * math.cos(self._spiral_angle_rad)
        lat0 = math.radians(center_lat)
        lat = center_lat + math.degrees(north_m / EARTH_RADIUS_M)
        lon = center_lon + math.degrees(east_m / (EARTH_RADIUS_M * max(math.cos(lat0), 1e-6)))
        self._cmd_lat = lat
        self._cmd_lon = lon
        self._cmd_rel_alt_m = center_alt

        msg = GlobalPositionTarget()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.coordinate_frame = GlobalPositionTarget.FRAME_GLOBAL_REL_ALT
        msg.type_mask = (
            GlobalPositionTarget.IGNORE_VX
            | GlobalPositionTarget.IGNORE_VY
            | GlobalPositionTarget.IGNORE_VZ
            | GlobalPositionTarget.IGNORE_AFX
            | GlobalPositionTarget.IGNORE_AFY
            | GlobalPositionTarget.IGNORE_AFZ
            | GlobalPositionTarget.IGNORE_YAW
            | GlobalPositionTarget.IGNORE_YAW_RATE
        )
        msg.latitude = lat
        msg.longitude = lon
        msg.altitude = center_alt
        self._gps_setpoint_pub.publish(msg)
        return True

    def _gps_distance_to_boat_m(self) -> float | None:
        if self._drone_fix is None or self._boat_fix is None:
            return None
        return haversine_m(
            float(self._drone_fix.latitude),
            float(self._drone_fix.longitude),
            float(self._boat_fix.latitude),
            float(self._boat_fix.longitude),
        )

    def _reset_gps_ramp(self) -> None:
        self._cmd_lat = None
        self._cmd_lon = None
        self._cmd_rel_alt_m = None
        self._last_tick_s = None
        self._spiral_start_s = None
        self._spiral_angle_rad = 0.0
        self._spiral_radius_m = 0.0
        self._spiral_center_lat = None
        self._spiral_center_lon = None
        self._spiral_center_alt_m = None

    def _gps_nav_arrived(self) -> bool:
        distance_m = self._gps_distance_to_boat_m()
        rel_alt = self._rel_alt_m()
        if distance_m is None or rel_alt is None:
            return False
        return (
            distance_m <= self._gps_arrival_radius_m
            and abs(rel_alt - self._gps_target_alt) <= self._gps_altitude_tolerance_m
        )

    def _start_gps_nav(self) -> None:
        self._landing_started = True
        self._return_to_gps_on_loss = True
        self._state = State.GPS_NAV
        self._setpoint_count = 0
        self._motors_low_count = 0
        self._align_missed_count = 0
        self._descend_missed_count = 0
        self._align_stable_start_time = 0.0
        self._reset_gps_ramp()
        self._set_control_owner("auto")
        self._publish_phase()

    def _start_visual_landing(self, return_to_gps_on_loss: bool) -> None:
        self._landing_started = True
        self._return_to_gps_on_loss = return_to_gps_on_loss
        self._state = State.ALIGN
        self._setpoint_count = 0
        self._motors_low_count = 0
        self._align_missed_count = 0
        self._descend_missed_count = 0
        self._align_stable_start_time = 0.0
        self._set_control_owner("auto")
        self._publish_phase()

    def _start_direct_spiral(self) -> tuple[bool, str]:
        if self._drone_fix is None:
            return False, "no /landing/drone_position_global sample yet"
        lat = float(self._drone_fix.latitude)
        lon = float(self._drone_fix.longitude)
        alt = self._rel_alt_m()
        if not math.isfinite(lat) or not math.isfinite(lon):
            return False, "current drone GPS is invalid"
        if alt is None:
            return False, "no relative altitude sample yet"
        self._landing_started = True
        self._return_to_gps_on_loss = False
        self._state = State.GPS_SPIRAL
        self._setpoint_count = 0
        self._motors_low_count = 0
        self._align_missed_count = 0
        self._descend_missed_count = 0
        self._align_stable_start_time = 0.0
        self._reset_gps_ramp()
        self._spiral_center_lat = lat
        self._spiral_center_lon = lon
        self._spiral_center_alt_m = alt
        self._set_control_owner("auto")
        self._publish_phase()
        return True, (
            "gps spiral started around current position: "
            f"lat={lat:.7f} lon={lon:.7f} alt={alt:.2f}m "
            f"radius={self._gps_spiral_max_radius_m:.1f}m "
            f"radial={self._gps_spiral_radial_speed_mps:.2f}m/s "
            f"tangential={self._gps_spiral_tangential_speed_mps:.2f}m/s"
        )

    def _return_after_target_loss(self) -> None:
        self._publish_cmd(0.0, 0.0, 0.0, 0.0)
        self._setpoint_count = 0
        if self._return_to_gps_on_loss:
            self._state = State.GPS_NAV
            self._reset_gps_ramp()
            self._set_control_owner("auto")
        else:
            self._state = State.MANUAL
            self._landing_started = False
            self._set_control_owner("teleop")
            self._request_mode(self._position_mode)
        self._publish_phase()

    def _finish_landing(self, reason: str) -> None:
        range_text = "none" if self._target_range_m is None else f"{self._target_range_m:.3f}m"
        self.get_logger().info(
            "finish_landing trigger: "
            f"reason={reason} phase={self._state.value} range={range_text} "
            f"last_error=({self._last_error[0]:.3f}m,{self._last_error[1]:.3f}m) "
            f"missed={self._descend_missed_count} motors_low={self._motors_low_count}"
        )
        self._publish_cmd(0.0, 0.0, 0.0, 0.0)
        self._request_mode("AUTO.LAND")
        self._state = State.DISARMED
        self._landing_started = False
        self._set_control_owner("teleop")
        self._publish_phase()

    def _kill_to_manual(self, request_position: bool = True) -> None:
        self._publish_cmd(0.0, 0.0, 0.0, 0.0)
        self._state = State.MANUAL
        self._landing_started = False
        self._return_to_gps_on_loss = False
        self._setpoint_count = 0
        self._reset_gps_ramp()
        self._set_control_owner("teleop")
        self._publish_phase()
        if request_position:
            self._request_mode(self._position_mode)

    def _handle_start(self, _req, resp):
        return self._start_gps_landing_response(resp)

    def _handle_start_gps_landing(self, _req, resp):
        return self._start_gps_landing_response(resp)

    def _start_gps_landing_response(self, resp):
        if self._qgc_takeover:
            resp.success = False
            resp.message = "QGC takeover active; call /landing/resume_auto first"
            return resp
        if self._state == State.DISARMED:
            resp.success = False
            resp.message = "already disarmed"
            return resp
        if self._landing_started and self._state != State.MANUAL:
            resp.success = False
            resp.message = f"already in {self._state.value}"
            return resp
        self._start_gps_nav()
        resp.success = True
        resp.message = "gps-based landing started; will enter ALIGN on first fresh target"
        return resp

    def _handle_begin_landing(self, _req, resp):
        return self._start_visual_landing_response(resp)

    def _handle_start_landing(self, _req, resp):
        return self._start_visual_landing_response(resp)

    def _start_visual_landing_response(self, resp):
        if self._qgc_takeover:
            resp.success = False
            resp.message = "QGC takeover active; call /landing/resume_auto first"
            return resp
        if self._state == State.DISARMED:
            resp.success = False
            resp.message = "already disarmed"
            return resp
        if not self._has_target():
            resp.success = False
            resp.message = "no fresh metric target; cannot begin visual landing"
            return resp
        self._start_visual_landing(return_to_gps_on_loss=False)
        resp.success = True
        resp.message = "visual landing started from current target"
        return resp

    def _handle_kill_offboard(self, _req, resp):
        self._kill_to_manual(request_position=True)
        resp.success = True
        resp.message = f"killed offboard; requested {self._position_mode}"
        return resp

    def _handle_gps_spiral(self, _req, resp):
        if self._qgc_takeover:
            resp.success = False
            resp.message = "QGC takeover active; call /landing/resume_auto first"
            return resp
        if self._state == State.DISARMED:
            resp.success = False
            resp.message = "already disarmed"
            return resp
        if self._landing_started and self._state not in {State.MANUAL, State.GPS_SPIRAL}:
            resp.success = False
            resp.message = f"already in {self._state.value}; call /landing/kill_offboard first"
            return resp
        resp.success, resp.message = self._start_direct_spiral()
        return resp

    def _request_velocity_frame(self) -> None:
        if self._velocity_frame_set:
            return
        now = self._now_s()
        if now - self._last_velocity_frame_request_time < 1.0:
            return
        self._last_velocity_frame_request_time = now
        if not self._set_velocity_frame_client.wait_for_service(timeout_sec=0.01):
            return
        value = ParameterValue()
        value.type = ParameterType.PARAMETER_STRING
        value.string_value = self._velocity_mav_frame
        parameter = Parameter(name="mav_frame", value=value)
        req = SetParameters.Request(parameters=[parameter])
        future = self._set_velocity_frame_client.call_async(req)
        future.add_done_callback(self._on_velocity_frame_response)

    def _on_velocity_frame_response(self, future) -> None:
        try:
            resp = future.result()
        except Exception as exc:
            self.get_logger().warn(f"set velocity frame failed: {exc}", throttle_duration_sec=2.0)
            return
        ok = bool(resp.results) and bool(resp.results[0].successful)
        if ok:
            self._velocity_frame_set = True
            self.get_logger().info(f"MAVROS setpoint_velocity mav_frame -> {self._velocity_mav_frame}")
        else:
            reason = resp.results[0].reason if resp.results else "empty result"
            self.get_logger().warn(f"MAVROS rejected mav_frame={self._velocity_mav_frame}: {reason}")

    def _request_mode(self, mode: str) -> None:
        if self._qgc_takeover:
            return
        mode = mode.strip().upper()
        if not mode or (self._mavros_mode or "").strip().upper() == mode:
            return
        now = self._now_s()
        if self._requested_mode == mode and now - self._last_mode_request_s < self._mode_request_interval_s:
            return
        if not self._mavros_connected and mode == self._offboard_mode:
            self.get_logger().warn("cannot request OFFBOARD: MAVROS disconnected", throttle_duration_sec=2.0)
            return
        if not self._set_mode_client.wait_for_service(timeout_sec=0.2):
            return
        self._requested_mode = mode
        self._last_mode_request_s = now
        req = SetMode.Request()
        req.base_mode = 0
        req.custom_mode = mode
        future = self._set_mode_client.call_async(req)
        future.add_done_callback(lambda fut, requested=mode: self._on_mode_response(fut, requested))

    def _on_mode_response(self, future, requested: str) -> None:
        try:
            result = future.result()
        except Exception as exc:
            self.get_logger().error(f"set_mode {requested} failed: {exc}")
            return
        if result.mode_sent:
            self.get_logger().info(f"set_mode {requested} accepted")
        else:
            self.get_logger().warn(f"set_mode {requested} rejected")

    def _tick(self) -> None:
        self._request_velocity_frame()
        self._publish_phase()
        if self._qgc_takeover or self._state == State.DISARMED:
            return

        if self._state == State.MANUAL:
            return

        if self._landing_started:
            self._set_control_owner("auto")

        if self._state in {State.GPS_NAV, State.GPS_SPIRAL}:
            if not self._landing_started:
                return
            if self._has_target():
                ex_m, ey_m, yaw_err = self._metric_errors()
                self._publish_metric_error(ex_m, ey_m, yaw_err)
                self._start_visual_landing(return_to_gps_on_loss=True)
                self.get_logger().info(f"{self._state.value.upper()} -> ALIGN: fresh metric target visible")
                return
            if self._state == State.GPS_SPIRAL:
                if self._publish_gps_spiral_setpoint():
                    self._setpoint_count += 1
                    if self._setpoint_count >= self._offboard_warmup_setpoints:
                        self._request_mode(self._offboard_mode)
                self.get_logger().info(
                    "GPS_SPIRAL "
                    f"radius={self._spiral_radius_m:.2f}m "
                    f"target_seen={self._target_detected}",
                    throttle_duration_sec=1.0,
                )
                return
            if self._publish_gps_setpoint():
                self._setpoint_count += 1
                if self._setpoint_count >= self._offboard_warmup_setpoints:
                    self._request_mode(self._offboard_mode)
            distance_m = self._gps_distance_to_boat_m()
            rel_alt = self._rel_alt_m()
            if self._gps_spiral_enabled and self._gps_nav_arrived():
                self._state = State.GPS_SPIRAL
                self._reset_gps_ramp()
                self._publish_phase()
                self.get_logger().info("GPS_NAV -> GPS_SPIRAL: arrived at boat GPS; searching slowly")
                return
            self.get_logger().info(
                "GPS_NAV "
                f"dist={distance_m if distance_m is not None else float('nan'):.2f}m "
                f"rel_alt={rel_alt if rel_alt is not None else float('nan'):.2f}m "
                f"target_seen={self._target_detected}",
                throttle_duration_sec=1.0,
            )
            return

        self._setpoint_count += 1
        if self._setpoint_count >= self._offboard_warmup_setpoints:
            self._request_mode(self._offboard_mode)

        if self._state == State.ALIGN:
            if not self._has_target():
                self._align_missed_count += 1
                if self._align_missed_count < self._align_missed_limit:
                    return
                self.get_logger().info("leaving ALIGN: target lost")
                self._return_after_target_loss()
                return
            self._align_missed_count = 0
            ex_m, ey_m, yaw_err = self._metric_errors()
            self._publish_metric_error(ex_m, ey_m, yaw_err)
            cmd = self._metric_cmd(ex_m, ey_m, 0.0)
            self._publish_cmd(*cmd)
            self.get_logger().info(
                f"ALIGN metric: ex={ex_m:.3f}m ey={ey_m:.3f}m yaw={yaw_err:.2f} "
                f"vx={cmd[0]:.2f} vy={cmd[1]:.2f}",
                throttle_duration_sec=0.5,
            )
            now = self._now_s()
            if self._aligned(ex_m, ey_m) and self._align_stable_start_time == 0.0:
                self._align_stable_start_time = now
            elif not self._aligned(ex_m, ey_m):
                self._align_stable_start_time = 0.0
            if self._align_stable_start_time != 0.0 and now - self._align_stable_start_time >= self._align_stable_s:
                self._state = State.DESCEND
                self._descend_missed_count = 0
                self._publish_phase()
            return

        if self._state == State.DESCEND:
            if self._motors_low_count >= 10:
                self._finish_landing("motors_low_descend")
                return
            if self._near_touchdown():
                self._finish_landing("target_range_touchdown_descend")
                return
            if not self._has_target():
                self._descend_missed_count += 1
                if (
                    self._descend_loss_is_tolerable()
                    and self._descend_missed_count < self._descend_missed_limit
                ):
                    self._publish_cmd(*self._last_descend_cmd)
                    return
                self.get_logger().info(
                    "leaving DESCEND: target lost "
                    f"missed={self._descend_missed_count} "
                    f"last_error=({self._last_error[0]:.3f}m,{self._last_error[1]:.3f}m) "
                    f"range={self._target_range_m if self._target_range_m is not None else 'none'}"
                )
                self._return_after_target_loss()
                return
            self._descend_missed_count = 0
            ex_m, ey_m, yaw_err = self._metric_errors()
            self._publish_metric_error(ex_m, ey_m, yaw_err)
            self._last_descend_cmd = self._metric_cmd(ex_m, ey_m, -self._descent_speed)
            self._publish_cmd(*self._last_descend_cmd)

    def _publish_phase(self) -> None:
        msg = String()
        msg.data = self._state.value
        self._phase_pub.publish(msg)

    def _set_control_owner(self, owner: str) -> None:
        msg = String()
        msg.data = owner
        self._owner_set_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PrecisionLandingControllerGpsNav()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
