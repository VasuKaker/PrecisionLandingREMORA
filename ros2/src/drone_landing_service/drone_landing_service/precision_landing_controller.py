#!/usr/bin/env python3
"""
Precision landing state machine.

States: IDLE -> ALTITUDE -> APPROACH -> SEARCH -> ALIGN -> DESCEND -> DISARMED

IDLE:     waiting for /landing/start service call
ALTITUDE: climb/descend to relative_altitude_boat_drone above boat
APPROACH: fly toward boat GPS until within approach_tolerance
SEARCH:   loiter, wait for target_detected == true
ALIGN:    hold altitude, align pixel + yaw to target
DESCEND:  descend while keeping pixel near centre
DISARMED: done
"""

import math
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import PointStamped, TwistStamped
from mavros_msgs.msg import ExtendedState, ParamValue, RCOut, State as MavrosState
from mavros_msgs.srv import ParamSet, SetMode
from std_msgs.msg import Bool, Float64, String
from std_srvs.srv import Trigger


class State(Enum):
    IDLE = 'idle'
    ALTITUDE = 'altitude'
    APPROACH = 'approach'
    SEARCH = 'search'
    ALIGN = 'align'
    DESCEND = 'descend'
    DISARMED = 'disarmed'


def haversine_m(lat1, lon1, lat2, lon2):
    """Horizontal distance in meters between two GPS coords."""
    R = 6371000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class PrecisionLandingController(Node):
    def __init__(self):
        super().__init__('precision_landing_controller')

        # -- params from YAML --
        self.declare_parameter('pixel_target_topic', '')
        self.declare_parameter('boat_position_global_topic', '')
        self.declare_parameter('boat_orientation_topic', '')
        self.declare_parameter('drone_global_position_topic', '')
        self.declare_parameter('drone_orientation_topic', '/mavros/global_position/compass_hdg')
        self.declare_parameter('target_detected_topic', '')
        self.declare_parameter('drone_speed_setpoint_topic', '')
        self.declare_parameter('drone_speed_topic', '')
        self.declare_parameter('relative_altitude_boat_drone', 3.0)
        self.declare_parameter('altitude_tolerance', 1.0)
        self.declare_parameter('approach_tolerance', 3.0)
        self.declare_parameter('kp', 0.02)
        self.declare_parameter('approach_yaw_tolerance_deg', 3.0)
        self.declare_parameter('camera_width', 640)
        self.declare_parameter('camera_height', 480)
        self.declare_parameter('max_horizontal_speed', 0.5)
        self.declare_parameter('max_vertical_speed', 0.5)
        self.declare_parameter('descent_speed', 0.3)
        self.declare_parameter('align_pixel_tolerance', 20.0)
        self.declare_parameter('align_yaw_tolerance', 0.15)

        # read them all
        pixel_topic = self.get_parameter('pixel_target_topic').value
        boat_topic = self.get_parameter('boat_position_global_topic').value
        boat_orientation_topic = self.get_parameter('boat_orientation_topic').value
        drone_topic = self.get_parameter('drone_global_position_topic').value
        drone_orientation_topic = self.get_parameter('drone_orientation_topic').value
        detected_topic = self.get_parameter('target_detected_topic').value
        mavros_vel_topic = self.get_parameter('drone_speed_setpoint_topic').value
        landing_vel_topic = self.get_parameter('drone_speed_topic').value
        self._target_alt = self.get_parameter('relative_altitude_boat_drone').value
        self._alt_tol = self.get_parameter('altitude_tolerance').value
        self._approach_tol = self.get_parameter('approach_tolerance').value
        self._kp = self.get_parameter('kp').value
        self._approach_yaw_tol_deg = self.get_parameter('approach_yaw_tolerance_deg').value
        self._cam_w = self.get_parameter('camera_width').value
        self._cam_h = self.get_parameter('camera_height').value
        self._vxy_max = self.get_parameter('max_horizontal_speed').value
        self._vz_max = self.get_parameter('max_vertical_speed').value
        # Positive speed magnitude in m/s; we apply negative sign for downward z command.
        self._descent_speed = min(abs(float(self.get_parameter('descent_speed').value)), self._vz_max)
        self._align_px_tol = self.get_parameter('align_pixel_tolerance').value
        self._align_yaw_tol = self.get_parameter('align_yaw_tolerance').value

        # -- state --
        self._state = State.IDLE
        self._drone_fix = None   # NavSatFix
        self._boat_fix = None    # NavSatFix
        self._boat_orientation_deg = None
        self._target_px = None   # PointStamped (pixel x, y)
        self._target_detected = False
        self._drone_compass_hdg_deg = None
        self._mavros_mode = ''
        self._mavros_connected = False
        self._mavros_armed = False
        self._landed = False
        self._motors_low_count = 0  # consecutive ticks with all motors < 500
        self._target_lost_ticks = 0
        self._setpoint_count = 0  # PX4 needs ~20 setpoints before accepting OFFBOARD
        self._last_offboard_request_time = 0.0
        self._qgc_takeover = False

        # -- subscribers --
        takeover_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(NavSatFix, drone_topic, self._on_drone_pos, 10)
        self.create_subscription(NavSatFix, boat_topic, self._on_boat_pos, 10)
        self.create_subscription(Float64, boat_orientation_topic, self._on_boat_orientation, 10)
        self.create_subscription(Float64, drone_orientation_topic, self._on_drone_orientation, qos_profile_sensor_data)
        self.create_subscription(PointStamped, pixel_topic, self._on_pixel, 10)
        self.create_subscription(Bool, detected_topic, self._on_detected, 10)
        self.create_subscription(MavrosState, '/mavros/state', self._on_mavros_state, 10)
        self.create_subscription(Bool, '/landing/qgc_takeover', self._on_qgc_takeover, takeover_qos)
        self.create_subscription(ExtendedState, '/mavros/extended_state', self._on_extended_state, 10)
        self.create_subscription(RCOut, '/mavros/rc/out', self._on_rc_out, 10)

        # -- publishers --
        self._mavros_vel_pub = self.create_publisher(TwistStamped, mavros_vel_topic, 10)
        self._landing_vel_pub = self.create_publisher(TwistStamped, landing_vel_topic, 10)
        self._phase_pub = self.create_publisher(String, '/landing/phase', 10)
        self._owner_set_pub = self.create_publisher(String, '/landing/control_owner/set', 10)

        # -- MAVROS services --
        self._set_mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self._param_set_client = self.create_client(ParamSet, '/mavros/param/set')

        # TODO: Set PX4 OFFBOARD loss failsafe to "Land" (2) instead of RTL
        # self._px4_param_timer = self.create_timer(2.0, self._set_px4_params_once)

        # -- service: call /landing/start to begin --
        self.create_service(Trigger, '/landing/start', self._handle_start)

        # -- 10 Hz control loop --
        self.create_timer(0.1, self._tick)

        self._publish_phase()
        self.get_logger().info('precision landing controller ready, call /landing/start')

    # ---- callbacks ----

    def _on_drone_pos(self, msg: NavSatFix):
        self._drone_fix = msg

    def _on_boat_pos(self, msg: NavSatFix):
        self._boat_fix = msg

    def _on_boat_orientation(self, msg: Float64):
        self._boat_orientation_deg = msg.data

    def _on_drone_orientation(self, msg: Float64):
        self._drone_compass_hdg_deg = msg.data

    def _on_pixel(self, msg: PointStamped):
        self._target_px = msg

    def _on_detected(self, msg: Bool):
        self._target_detected = msg.data

    def _on_mavros_state(self, msg: MavrosState):
        self._mavros_mode = msg.mode
        self._mavros_connected = bool(msg.connected)
        self._mavros_armed = bool(msg.armed)

    def _on_qgc_takeover(self, msg: Bool):
        self._qgc_takeover = bool(msg.data)
        if self._qgc_takeover and self._state != State.IDLE:
            self.get_logger().warn('QGC takeover active: stopping precision landing')
            self._state = State.IDLE
            self._setpoint_count = 0
            self._publish_phase()

    def _on_extended_state(self, msg: ExtendedState):
        self._landed = (msg.landed_state == ExtendedState.LANDED_STATE_ON_GROUND)

    def _on_rc_out(self, msg: RCOut):
        # First 4 channels are motors. Count how many are below 500.
        motors = list(msg.channels[:4])
        low = sum(1 for m in motors if m < 500)
        if low >= 2:
            self._motors_low_count += 1
        else:
            self._motors_low_count = 0

    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    @staticmethod
    def _finite(value) -> bool:
        return value is not None and math.isfinite(value)

    def _current_altitude(self):
        if self._drone_fix is None or not self._finite(self._drone_fix.altitude):
            self.get_logger().warn(
                'waiting for finite drone altitude',
                throttle_duration_sec=5.0,
            )
            return None
        return float(self._drone_fix.altitude)

    @staticmethod
    def _wrap_angle_deg(angle_deg: float) -> float:
        return (angle_deg + 180.0) % 360.0 - 180.0

    def _pixel_errors(self):
        """Return pixel errors relative to frame center and marker yaw error."""
        ex = self._target_px.point.x - (self._cam_w / 2.0)
        ey = -1.0 * (self._target_px.point.y - (self._cam_h / 2.0))
        yaw_err = self._target_px.point.z
        return ex, ey, yaw_err

    def _handle_start(self, req, resp):
        if self._qgc_takeover:
            resp.success = False
            resp.message = 'QGC takeover active; call /landing/resume_auto first'
            return resp
        if self._state in (State.IDLE, State.DISARMED):
            self._set_state(State.ALTITUDE)
            self._setpoint_count = 0
            self._motors_low_count = 0
            self._set_control_owner('auto')
            resp.success = True
            resp.message = 'landing sequence started'
        else:
            resp.success = False
            resp.message = f'already in {self._state.value}'
        return resp

    # ---- state machine tick @ 10 Hz ----

    def _tick(self):
        # Always publish phase so we can monitor
        self._publish_phase()
        if self._qgc_takeover:
            return

        if self._state == State.IDLE:
            return

        # In DISARMED (AUTO.LAND sent), PX4 handles the rest. Stay here
        # until user calls /landing/start again.
        if self._state == State.DISARMED:
            return

        # PX4 requires setpoints streaming before it accepts OFFBOARD mode.
        # Keep requesting OFFBOARD whenever we're not in it.
        self._setpoint_count += 1
        if (not self._is_offboard_mode()) and self._setpoint_count >= 20:
            self._request_offboard()

        if self._state == State.ALTITUDE:
            altitude = self._current_altitude()
            if altitude is None:
                return
            # Command drone to target altitude, zero horizontal velocity
            alt_error = self._target_alt - altitude
            cmd = TwistStamped()
            cmd.header.stamp = self.get_clock().now().to_msg()
            # Use a faster climb/descent cap in ALTITUDE phase so we reach target height promptly.
            cmd.twist.linear.z = max(-1.0, min(1.0, alt_error))
            self._mavros_vel_pub.publish(cmd)
            self._landing_vel_pub.publish(cmd)

            if abs(alt_error) <= self._alt_tol:
                self._set_state(State.APPROACH)

        elif self._state == State.APPROACH:
            altitude = self._current_altitude()
            if self._drone_fix is None or self._boat_fix is None or altitude is None:
                return
            # Fly toward boat GPS using proportional control
            dist = haversine_m(
                self._drone_fix.latitude, self._drone_fix.longitude,
                self._boat_fix.latitude, self._boat_fix.longitude)

            # Compute north/east error in meters
            dlat = math.radians(self._boat_fix.latitude - self._drone_fix.latitude) * 6371000.0
            dlon = math.radians(self._boat_fix.longitude - self._drone_fix.longitude) * 6371000.0 * math.cos(math.radians(self._drone_fix.latitude))
            
            cmd = TwistStamped()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.twist.linear.x = -max(-self._vxy_max, min(self._vxy_max, dlat))   # north
            cmd.twist.linear.y = -max(-self._vxy_max, min(self._vxy_max, dlon))   # east
            cmd.twist.angular.z = 0.0
            # Hold altitude
            alt_error = self._target_alt - altitude
            cmd.twist.linear.z = max(-self._vz_max, min(self._vz_max, alt_error))
            self._mavros_vel_pub.publish(cmd)
            self._landing_vel_pub.publish(cmd)

            if dist <= self._approach_tol:
                self._set_state(State.SEARCH)

        elif self._state == State.SEARCH:
            # Loiter, wait for marker detection
            if self._target_detected:
                self._set_state(State.ALIGN)

        elif self._state == State.ALIGN:
            # Hold altitude and align target to image center + yaw before descent.
            # if (not self._target_detected) or (self._target_px is None) or (self._drone_fix is None):
            #     self._set_state(State.SEARCH)
            #     return

            altitude = self._current_altitude()
            if self._target_px is None or altitude is None:
                return

            ex, ey, yaw_err = self._pixel_errors()
            alt_error = self._target_alt - altitude

            cmd = TwistStamped()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.twist.linear.x = self._clamp(self._kp * ex, -self._vxy_max, self._vxy_max)
            cmd.twist.linear.y = self._clamp(self._kp * ey, -self._vxy_max, self._vxy_max)
            cmd.twist.linear.z = self._clamp(alt_error, -self._vz_max, self._vz_max)  # NO descent in ALIGN
            cmd.twist.angular.z = 0.0
            self._mavros_vel_pub.publish(cmd)
            self._landing_vel_pub.publish(cmd)

            if (
                abs(ex) <= self._align_px_tol
                and abs(ey) <= self._align_px_tol
            ):
                self._set_state(State.DESCEND)

        elif self._state == State.DESCEND:
            if self._motors_low_count >= 10:
                self._set_state(State.DISARMED)
                return

            # If marker is lost, descend straight down (no XY guesswork).
            if (not self._target_detected) or (self._target_px is None):
                cmd = TwistStamped()
                cmd.header.stamp = self.get_clock().now().to_msg()
                cmd.twist.linear.x = 0.0
                cmd.twist.linear.y = 0.0
                cmd.twist.linear.z = -self._descent_speed
                cmd.twist.angular.z = 0.0
                self._mavros_vel_pub.publish(cmd)
                self._landing_vel_pub.publish(cmd)
                return
            ex, ey, yaw_err = self._pixel_errors()

            cmd = TwistStamped()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.twist.linear.x = self._clamp(self._kp * ex, -self._vxy_max, self._vxy_max)
            cmd.twist.linear.y = self._clamp(self._kp * ey, -self._vxy_max, self._vxy_max)
            cmd.twist.linear.z = -self._descent_speed
            cmd.twist.angular.z = 0.0

            self._mavros_vel_pub.publish(cmd)
            self._landing_vel_pub.publish(cmd)

    # ---- helpers ----

    def _set_state(self, new_state: State):
        self.get_logger().info(f'{self._state.value} -> {new_state.value}')
        self._state = new_state
        if new_state == State.DESCEND:
            self._motors_low_count = 0
        self._publish_phase()
        if new_state == State.DISARMED:
            self._request_land_mode()

    def _request_land_mode(self):
        """Switch PX4 to AUTO.LAND — it handles touchdown + disarm."""
        if not self._set_mode_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn('set_mode service not available')
            return
        req = SetMode.Request()
        req.custom_mode = 'AUTO.LAND'
        future = self._set_mode_client.call_async(req)
        future.add_done_callback(
            lambda f: self.get_logger().info(
                f'AUTO.LAND {"accepted" if f.result().mode_sent else "rejected"}'))

    def _publish_phase(self):
        msg = String()
        msg.data = self._state.value
        self._phase_pub.publish(msg)

    def _set_control_owner(self, owner: str):
        msg = String()
        msg.data = owner
        self._owner_set_pub.publish(msg)

    def _set_px4_params_once(self):
        """Set OFFBOARD loss failsafe to Land (2) on startup, then cancel timer."""
        if not self._param_set_client.wait_for_service(timeout_sec=0.5):
            return
        req = ParamSet.Request()
        req.param_id = 'COM_OBL_RC_ACT'
        req.value = ParamValue(integer=1, real=0.0)
        future = self._param_set_client.call_async(req)
        future.add_done_callback(
            lambda f: self.get_logger().info(f'COM_OBL_RC_ACT=1 {"ok" if f.result().success else "failed"}'))
        # Cancel this timer so it only runs once
        self._px4_param_timer.cancel()

    def _request_offboard(self):
        if self._qgc_takeover:
            return
        if not self._mavros_connected:
            self.get_logger().warn(
                'cannot request OFFBOARD: /mavros/state reports disconnected',
                throttle_duration_sec=2.0,
            )
            return
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self._last_offboard_request_time < 1.0:
            return
        self._last_offboard_request_time = now
        if not self._set_mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('set_mode service not available')
            return
        req = SetMode.Request()
        req.base_mode = 0
        req.custom_mode = 'OFFBOARD'
        self.get_logger().info(
            f'requesting OFFBOARD: connected={self._mavros_connected} '
            f'armed={self._mavros_armed} mode="{self._mavros_mode}" '
            f'setpoints={self._setpoint_count}',
            throttle_duration_sec=1.0,
        )
        future = self._set_mode_client.call_async(req)
        future.add_done_callback(self._on_offboard_response)

    def _on_offboard_response(self, future):
        try:
            resp = future.result()
            if resp.mode_sent:
                self.get_logger().info('OFFBOARD mode accepted')
            else:
                self.get_logger().warn(
                    f'OFFBOARD mode rejected; connected={self._mavros_connected} '
                    f'armed={self._mavros_armed} mode="{self._mavros_mode}"'
                )
        except Exception as e:
            self.get_logger().error(f'set_mode failed: {e}')

    def _is_offboard_mode(self) -> bool:
        return (self._mavros_mode or '').strip().upper() == 'OFFBOARD'


def main(args=None):
    rclpy.init(args=args)
    node = PrecisionLandingController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
