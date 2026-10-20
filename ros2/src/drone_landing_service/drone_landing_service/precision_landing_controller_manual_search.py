#!/usr/bin/env python3
"""Manual-start precision landing controller using homography metric target error."""

from __future__ import annotations

import math
from enum import Enum

import rclpy
from geometry_msgs.msg import PointStamped, TwistStamped
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from mavros_msgs.msg import RCOut, State as MavrosState
from mavros_msgs.srv import SetMode
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Bool, Float32, String
from std_srvs.srv import Trigger


class State(Enum):
    MANUAL = "manual"
    ALIGN_READY = "align_ready"
    ALIGN = "align"
    DESCEND = "descend"
    DISARMED = "disarmed"


class PrecisionLandingControllerManualSearch(Node):
    def __init__(self):
        super().__init__("precision_landing_controller_manual_search")

        self.declare_parameter("target_metric_topic", "/landing/target_metric")
        self.declare_parameter("target_range_topic", "/landing/target_range_m")
        self.declare_parameter("target_detected_topic", "/landing/target_detected")
        self.declare_parameter("metric_error_topic", "/landing/metric_error")
        self.declare_parameter("drone_speed_setpoint_topic", "/mavros/setpoint_velocity/cmd_vel")
        self.declare_parameter("drone_speed_topic", "/landing/drone_velocity_setpoint")
        self.declare_parameter("kp", 0.45)
        self.declare_parameter("metric_vx_sign", 1.0)
        self.declare_parameter("metric_vy_sign", 1.0)
        self.declare_parameter("velocity_mav_frame", "BODY_NED")
        self.declare_parameter("max_horizontal_speed", 0.20)
        self.declare_parameter("min_horizontal_speed", 0.0)
        self.declare_parameter("max_vertical_speed", 0.30)
        self.declare_parameter("descent_speed", 0.10)
        self.declare_parameter("align_metric_tolerance_m", 0.025)
        self.declare_parameter("align_entry_metric_threshold_m", 0.45)
        self.declare_parameter("align_stable_s", 1.0)
        self.declare_parameter("entry_grace_s", 0.8)
        self.declare_parameter("target_metric_timeout_s", 0.40)
        self.declare_parameter("align_missed_sample_limit", 5)
        self.declare_parameter("descend_missed_sample_limit", 5)
        self.declare_parameter("touchdown_range_m", 0.40)
        self.declare_parameter("touchdown_xy_tolerance_m", 0.12)
        self.declare_parameter("start_on_launch", False)

        metric_topic = self.get_parameter("target_metric_topic").value
        range_topic = self.get_parameter("target_range_topic").value
        detected_topic = self.get_parameter("target_detected_topic").value
        metric_error_topic = self.get_parameter("metric_error_topic").value
        mavros_vel_topic = self.get_parameter("drone_speed_setpoint_topic").value
        landing_vel_topic = self.get_parameter("drone_speed_topic").value

        self._kp = float(self.get_parameter("kp").value)
        self._metric_vx_sign = 1.0 if float(self.get_parameter("metric_vx_sign").value) >= 0.0 else -1.0
        self._metric_vy_sign = 1.0 if float(self.get_parameter("metric_vy_sign").value) >= 0.0 else -1.0
        self._velocity_mav_frame = str(self.get_parameter("velocity_mav_frame").value).strip() or "BODY_NED"
        self._vxy_max = abs(float(self.get_parameter("max_horizontal_speed").value))
        self._vxy_min = min(abs(float(self.get_parameter("min_horizontal_speed").value)), self._vxy_max)
        self._vz_max = abs(float(self.get_parameter("max_vertical_speed").value))
        self._descent_speed = min(abs(float(self.get_parameter("descent_speed").value)), self._vz_max)
        self._align_tol_m = float(self.get_parameter("align_metric_tolerance_m").value)
        self._entry_threshold_m = float(self.get_parameter("align_entry_metric_threshold_m").value)
        self._align_stable_s = float(self.get_parameter("align_stable_s").value)
        self._entry_grace_s = float(self.get_parameter("entry_grace_s").value)
        self._target_timeout_s = float(self.get_parameter("target_metric_timeout_s").value)
        self._align_missed_limit = int(self.get_parameter("align_missed_sample_limit").value)
        self._descend_missed_limit = int(self.get_parameter("descend_missed_sample_limit").value)
        self._touchdown_range_m = float(self.get_parameter("touchdown_range_m").value)
        self._touchdown_xy_tol_m = float(self.get_parameter("touchdown_xy_tolerance_m").value)
        self._start_on_launch = bool(self.get_parameter("start_on_launch").value)

        self._state = State.MANUAL
        self._landing_started = False
        self._target_metric: PointStamped | None = None
        self._target_metric_time = 0.0
        self._target_detected = False
        self._mavros_mode = ""
        self._mavros_connected = False
        self._mavros_armed = False
        self._motors_low_count = 0
        self._setpoint_count = 0
        self._last_offboard_request_time = 0.0
        self._last_position_request_time = 0.0
        self._qgc_takeover = False
        self._last_entry_target_time = 0.0
        self._align_stable_start_time = 0.0
        self._align_missed_count = 0
        self._descend_missed_count = 0
        self._last_error = (0.0, 0.0)
        self._last_descend_cmd = (0.0, 0.0, -self._descent_speed, 0.0)
        self._target_range_m: float | None = None
        self._target_range_time = 0.0
        self._velocity_frame_set = False
        self._last_velocity_frame_request_time = 0.0

        takeover_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(PointStamped, metric_topic, self._on_metric, 10)
        self.create_subscription(Float32, range_topic, self._on_range, 10)
        self.create_subscription(Bool, detected_topic, self._on_detected, 10)
        self.create_subscription(Bool, "/landing/qgc_takeover", self._on_qgc_takeover, takeover_qos)
        self.create_subscription(MavrosState, "/mavros/state", self._on_mavros_state, 10)
        self.create_subscription(RCOut, "/mavros/rc/out", self._on_rc_out, 10)

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
        self.create_service(Trigger, "/landing/begin_landing", self._handle_begin_landing)
        self.create_timer(0.1, self._tick)
        self._publish_phase()
        if self._start_on_launch:
            self._start_manual_search()
            self.get_logger().info("metric landing auto-started in POSCTL; waiting for align_ready")

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

    def _on_mavros_state(self, msg: MavrosState) -> None:
        self._mavros_mode = msg.mode
        self._mavros_connected = bool(msg.connected)
        self._mavros_armed = bool(msg.armed)

    def _on_qgc_takeover(self, msg: Bool) -> None:
        self._qgc_takeover = bool(msg.data)
        if self._qgc_takeover:
            self.get_logger().warn("QGC takeover active: stopping metric landing")
            self._state = State.MANUAL
            self._landing_started = False
            self._setpoint_count = 0
            self._publish_phase()

    def _on_rc_out(self, msg: RCOut) -> None:
        low = sum(1 for value in list(msg.channels[:4]) if value < 500)
        self._motors_low_count = self._motors_low_count + 1 if low >= 2 else 0

    def _has_metric(self) -> bool:
        return self._target_metric is not None and self._now_s() - self._target_metric_time <= self._target_timeout_s

    def _has_range(self) -> bool:
        return self._target_range_m is not None and self._now_s() - self._target_range_time <= self._target_timeout_s

    def _has_target(self) -> bool:
        return self._target_detected and self._has_metric()

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
        raw = self._kp * error_m
        raw = self._clamp(raw, -self._vxy_max, self._vxy_max)
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

    def _finish_landing(self, reason: str) -> None:
        range_text = "none"
        if self._target_range_m is not None:
            range_text = f"{self._target_range_m:.3f}m"
        self.get_logger().info(
            "finish_landing trigger: "
            f"reason={reason} phase={self._state.value} range={range_text} "
            f"last_error=({self._last_error[0]:.3f}m,{self._last_error[1]:.3f}m) "
            f"missed={self._descend_missed_count} motors_low={self._motors_low_count}"
        )
        self._publish_cmd(0.0, 0.0, 0.0, 0.0)
        self._request_land_mode()
        self._state = State.DISARMED
        self._landing_started = False
        self._set_control_owner("teleop")
        self._publish_phase()

    def _start_manual_search(self) -> None:
        self._landing_started = True
        self._setpoint_count = 0
        self._motors_low_count = 0
        self._request_position_mode()
        self._publish_phase()

    def _handle_start(self, _req, resp):
        if self._qgc_takeover:
            resp.success = False
            resp.message = "QGC takeover active; call /landing/resume_auto first"
            return resp
        if self._state == State.DISARMED:
            resp.success = False
            resp.message = "already disarmed"
            return resp
        if self._state not in (State.MANUAL, State.ALIGN_READY):
            resp.success = False
            resp.message = f"already in {self._state.value}"
            return resp
        if self._landing_started:
            resp.success = False
            resp.message = "metric landing already started"
            return resp
        self._start_manual_search()
        resp.success = True
        resp.message = "metric landing started in POSCTL; waiting for align_ready"
        return resp

    def _handle_begin_landing(self, _req, resp):
        if self._qgc_takeover:
            resp.success = False
            resp.message = "QGC takeover active; call /landing/resume_auto first"
            return resp
        if self._state != State.ALIGN_READY:
            resp.success = False
            resp.message = f"not align_ready (state={self._state.value})"
            return resp
        self._set_control_owner("auto")
        self._setpoint_count = 0
        self._align_stable_start_time = 0.0
        self._align_missed_count = 0
        self._descend_missed_count = 0
        self._state = State.ALIGN
        self._publish_phase()
        resp.success = True
        resp.message = "metric landing alignment started; requesting OFFBOARD after setpoint warmup"
        return resp

    def _request_offboard(self) -> None:
        if self._qgc_takeover or (self._mavros_mode or "").strip().upper() == "AUTO.LAND":
            return
        if not self._mavros_connected:
            self.get_logger().warn("cannot request OFFBOARD: MAVROS disconnected", throttle_duration_sec=2.0)
            return
        now = self._now_s()
        if now - self._last_offboard_request_time < 1.0:
            return
        self._last_offboard_request_time = now
        if not self._set_mode_client.wait_for_service(timeout_sec=0.2):
            return
        req = SetMode.Request()
        req.base_mode = 0
        req.custom_mode = "OFFBOARD"
        self.get_logger().info(
            f"requesting OFFBOARD: connected={self._mavros_connected} "
            f"armed={self._mavros_armed} mode=\"{self._mavros_mode}\" setpoints={self._setpoint_count}",
            throttle_duration_sec=1.0,
        )
        future = self._set_mode_client.call_async(req)
        future.add_done_callback(self._on_offboard_response)

    def _request_land_mode(self) -> None:
        if not self._set_mode_client.wait_for_service(timeout_sec=0.2):
            return
        req = SetMode.Request()
        req.custom_mode = "AUTO.LAND"
        self._set_mode_client.call_async(req)

    def _request_position_mode(self) -> None:
        if self._qgc_takeover or (self._mavros_mode or "").strip().upper() == "POSCTL":
            return
        now = self._now_s()
        if now - self._last_position_request_time < 1.0:
            return
        self._last_position_request_time = now
        if not self._set_mode_client.wait_for_service(timeout_sec=0.2):
            return
        req = SetMode.Request()
        req.base_mode = 0
        req.custom_mode = "POSCTL"
        self._set_mode_client.call_async(req)

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

    def _tick(self) -> None:
        self._request_velocity_frame()
        self._publish_phase()
        if self._qgc_takeover:
            return
        if self._state == State.DISARMED:
            return

        if self._state == State.MANUAL:
            self.get_logger().info(
                f"MANUAL landing_started={self._landing_started} target_detected={self._target_detected}",
                throttle_duration_sec=1.0,
            )
            if not self._landing_started:
                return
            self._request_position_mode()
            if not self._has_target():
                return
            ex_m, ey_m, yaw_err = self._metric_errors()
            self._publish_metric_error(ex_m, ey_m, yaw_err)
            if math.hypot(ex_m, ey_m) > self._entry_threshold_m:
                return
            self._last_entry_target_time = self._now_s()
            self._state = State.ALIGN_READY
            self._publish_phase()
            self.get_logger().info("ALIGN_READY: call /landing/begin_landing to enter OFFBOARD")
            return

        if self._state == State.ALIGN_READY:
            self._request_position_mode()
            now = self._now_s()
            if not self._has_target():
                if now - self._last_entry_target_time <= self._entry_grace_s:
                    return
                self.get_logger().info("leaving ALIGN_READY: target lost")
                self._state = State.MANUAL
                self._request_position_mode()
                self._publish_phase()
                return
            ex_m, ey_m, yaw_err = self._metric_errors()
            self._publish_metric_error(ex_m, ey_m, yaw_err)
            if math.hypot(ex_m, ey_m) > self._entry_threshold_m:
                if now - self._last_entry_target_time <= self._entry_grace_s:
                    return
                self._state = State.MANUAL
                self._request_position_mode()
                self._publish_phase()
                return
            self._last_entry_target_time = now
            return

        self._setpoint_count += 1
        if not self._is_offboard_mode() and self._setpoint_count >= 20:
            self._request_offboard()

        if self._state == State.ALIGN:
            if not self._has_target():
                self._align_missed_count += 1
                if self._align_missed_count < self._align_missed_limit:
                    return
                self._state = State.MANUAL
                self._landing_started = True
                self._setpoint_count = 0
                self._request_position_mode()
                self._publish_phase()
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
                self._state = State.MANUAL
                self._landing_started = True
                self._setpoint_count = 0
                self._request_position_mode()
                self._publish_phase()
                return
            self._descend_missed_count = 0
            ex_m, ey_m, yaw_err = self._metric_errors()
            self._publish_metric_error(ex_m, ey_m, yaw_err)
            self._last_descend_cmd = self._metric_cmd(ex_m, ey_m, -self._descent_speed)
            self._publish_cmd(*self._last_descend_cmd)
            return

    def _publish_phase(self) -> None:
        msg = String()
        msg.data = self._state.value
        self._phase_pub.publish(msg)

    def _set_control_owner(self, owner: str) -> None:
        msg = String()
        msg.data = owner
        self._owner_set_pub.publish(msg)

    def _on_offboard_response(self, future) -> None:
        try:
            resp = future.result()
            if resp.mode_sent:
                self.get_logger().info("OFFBOARD mode accepted")
            else:
                self.get_logger().warn(
                    f"OFFBOARD mode rejected; connected={self._mavros_connected} "
                    f"armed={self._mavros_armed} mode=\"{self._mavros_mode}\""
                )
        except Exception as exc:
            self.get_logger().error(f"set_mode failed: {exc}")

    def _is_offboard_mode(self) -> bool:
        return (self._mavros_mode or "").strip().upper() == "OFFBOARD"


def main(args=None):
    rclpy.init(args=args)
    node = PrecisionLandingControllerManualSearch()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
