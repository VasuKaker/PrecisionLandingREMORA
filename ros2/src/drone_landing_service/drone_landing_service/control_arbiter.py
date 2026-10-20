#!/usr/bin/env python3
"""Select exactly one command source to forward to MAVROS."""

import copy
import math

import rclpy
from geometry_msgs.msg import TwistStamped
from mavros_msgs.msg import State as MavrosState
from mavros_msgs.srv import SetMode
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Bool, String


class ControlArbiter(Node):
    def __init__(self):
        super().__init__('control_arbiter')

        self.declare_parameter('teleop_cmd_topic', '/landing/cmd/teleop')
        self.declare_parameter('auto_cmd_topic', '/landing/cmd/auto')
        self.declare_parameter('mavros_cmd_topic', '/mavros/setpoint_velocity/cmd_vel')
        self.declare_parameter('selected_cmd_topic', '/landing/drone_velocity_setpoint')
        self.declare_parameter('owner_set_topic', '/landing/control_owner/set')
        self.declare_parameter('owner_topic', '/landing/control_owner')
        self.declare_parameter('phase_topic', '/landing/phase')
        self.declare_parameter('default_owner', 'teleop')
        self.declare_parameter('publish_rate_hz', 20.0)
        self.declare_parameter('enable_mode_switching', True)
        self.declare_parameter('offboard_mode', 'OFFBOARD')
        self.declare_parameter('position_mode', 'POSCTL')
        self.declare_parameter('command_timeout_s', 0.35)
        self.declare_parameter('idle_position_delay_s', 0.5)
        self.declare_parameter('mode_request_interval_s', 1.0)
        self.declare_parameter('active_command_epsilon', 1e-3)
        self.declare_parameter('offboard_warmup_setpoints', 20)

        teleop_cmd_topic = self.get_parameter('teleop_cmd_topic').value
        auto_cmd_topic = self.get_parameter('auto_cmd_topic').value
        mavros_cmd_topic = self.get_parameter('mavros_cmd_topic').value
        selected_cmd_topic = self.get_parameter('selected_cmd_topic').value
        owner_set_topic = self.get_parameter('owner_set_topic').value
        owner_topic = self.get_parameter('owner_topic').value
        phase_topic = self.get_parameter('phase_topic').value
        self._requested_owner = str(self.get_parameter('default_owner').value).strip().lower() or 'teleop'
        self._owner = self._requested_owner
        self._phase = 'manual'
        publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self._enable_mode_switching = bool(self.get_parameter('enable_mode_switching').value)
        self._offboard_mode = str(self.get_parameter('offboard_mode').value).strip().upper() or 'OFFBOARD'
        self._position_mode = str(self.get_parameter('position_mode').value).strip().upper() or 'POSCTL'
        self._command_timeout_s = float(self.get_parameter('command_timeout_s').value)
        self._idle_position_delay_s = float(self.get_parameter('idle_position_delay_s').value)
        self._mode_request_interval_s = float(self.get_parameter('mode_request_interval_s').value)
        self._active_command_epsilon = float(self.get_parameter('active_command_epsilon').value)
        self._offboard_warmup_setpoints = int(self.get_parameter('offboard_warmup_setpoints').value)

        owner_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        takeover_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._owner_pub = self.create_publisher(String, owner_topic, owner_qos)
        self._mavros_pub = self.create_publisher(TwistStamped, mavros_cmd_topic, 10)
        self._selected_cmd_pub = self.create_publisher(TwistStamped, selected_cmd_topic, 10)
        self._set_mode_client = self.create_client(SetMode, '/mavros/set_mode')

        self.create_subscription(TwistStamped, teleop_cmd_topic, self._on_teleop_cmd, 10)
        self.create_subscription(TwistStamped, auto_cmd_topic, self._on_auto_cmd, 10)
        self.create_subscription(String, owner_set_topic, self._on_owner_set, 10)
        self.create_subscription(String, phase_topic, self._on_phase, 10)
        self.create_subscription(Bool, '/landing/qgc_takeover', self._on_qgc_takeover, takeover_qos)
        self.create_subscription(MavrosState, '/mavros/state', self._on_mavros_state, 10)

        self._last_teleop_cmd = TwistStamped()
        self._last_auto_cmd = TwistStamped()
        self._last_teleop_cmd_time = None
        self._last_auto_cmd_time = None
        self._mavros_connected = False
        self._mavros_mode = ''
        self._active_setpoint_count = 0
        self._last_active_time = None
        self._last_mode_request_time = 0.0
        self._requested_mode = None
        self._qgc_takeover = False

        self.create_timer(1.0 / publish_rate_hz, self._tick)
        self._publish_owner()
        self.get_logger().info(f'control arbiter ready: owner={self._owner}')

    def _on_teleop_cmd(self, msg: TwistStamped):
        self._last_teleop_cmd = copy.deepcopy(msg)
        self._last_teleop_cmd_time = self._now_s()

    def _on_auto_cmd(self, msg: TwistStamped):
        self._last_auto_cmd = copy.deepcopy(msg)
        self._last_auto_cmd_time = self._now_s()

    def _on_mavros_state(self, msg: MavrosState):
        self._mavros_connected = bool(msg.connected)
        self._mavros_mode = (msg.mode or '').strip().upper()

    def _on_qgc_takeover(self, msg: Bool):
        self._qgc_takeover = bool(msg.data)

    def _on_owner_set(self, msg: String):
        requested = (msg.data or '').strip().lower()
        if requested not in {'teleop', 'auto'}:
            self.get_logger().warn(f'ignoring invalid owner request "{msg.data}"')
            return
        self._requested_owner = requested
        self._update_owner()

    def _on_phase(self, msg: String):
        self._phase = (msg.data or '').strip().lower()
        self._update_owner()

    def _update_owner(self):
        # During manual-search, let teleop keep ownership until the controller
        # actually transitions out of the manual phase.
        effective_owner = 'teleop' if self._phase == 'manual' else self._requested_owner
        if effective_owner == self._owner:
            return
        self._owner = effective_owner
        self.get_logger().info(
            f'control owner -> {self._owner} '
            f'(requested={self._requested_owner}, phase={self._phase})'
        )
        self._publish_owner()

    def _publish_owner(self):
        msg = String()
        msg.data = self._owner
        self._owner_pub.publish(msg)

    def _tick(self):
        now = self._now_s()
        cmd, cmd_time = self._selected_command()
        active = (not self._qgc_takeover) and self._is_active_command(cmd, cmd_time, now)
        cmd = copy.deepcopy(cmd if active else TwistStamped())
        cmd.header.stamp = self.get_clock().now().to_msg()
        if self._phase != 'gps_nav':
            self._mavros_pub.publish(cmd)
        self._selected_cmd_pub.publish(cmd)
        self._publish_owner()
        if self._phase != 'gps_nav':
            self._update_mode(active, now)

    def _selected_command(self):
        if self._owner == 'teleop':
            return self._last_teleop_cmd, self._last_teleop_cmd_time
        return self._last_auto_cmd, self._last_auto_cmd_time

    def _is_active_command(self, cmd: TwistStamped, cmd_time, now: float) -> bool:
        if cmd_time is None or now - cmd_time > self._command_timeout_s:
            return False
        if self._owner == 'auto' and self._phase in {'align', 'descend', 'descend_blind'}:
            return True
        values = (
            cmd.twist.linear.x,
            cmd.twist.linear.y,
            cmd.twist.linear.z,
            cmd.twist.angular.x,
            cmd.twist.angular.y,
            cmd.twist.angular.z,
        )
        return math.sqrt(sum(float(v) * float(v) for v in values)) > self._active_command_epsilon

    def _update_mode(self, active: bool, now: float):
        if not self._enable_mode_switching:
            return
        if self._qgc_takeover:
            return
        if not self._mavros_connected or self._mavros_mode == 'AUTO.LAND':
            return

        if active:
            self._last_active_time = now
            self._active_setpoint_count += 1
            if (
                self._active_setpoint_count >= self._offboard_warmup_setpoints
                and self._mavros_mode != self._offboard_mode
            ):
                self._request_mode(self._offboard_mode, now)
            return

        self._active_setpoint_count = 0
        if self._last_active_time is None:
            return
        if now - self._last_active_time < self._idle_position_delay_s:
            return
        if self._mavros_mode == self._offboard_mode:
            self._request_mode(self._position_mode, now)

    def _request_mode(self, mode: str, now: float):
        if self._requested_mode == mode and now - self._last_mode_request_time < self._mode_request_interval_s:
            return
        if not self._set_mode_client.wait_for_service(timeout_sec=0.01):
            self.get_logger().warn('set_mode service not available', throttle_duration_sec=2.0)
            return

        self._requested_mode = mode
        self._last_mode_request_time = now
        req = SetMode.Request()
        req.base_mode = 0
        req.custom_mode = mode
        future = self._set_mode_client.call_async(req)
        future.add_done_callback(lambda fut, requested=mode: self._on_mode_response(fut, requested))

    def _on_mode_response(self, future, requested: str):
        try:
            result = future.result()
        except Exception as exc:
            self.get_logger().error(f'set_mode {requested} failed: {exc}')
            return
        if result.mode_sent:
            self.get_logger().info(f'set_mode {requested} accepted')
        else:
            self.get_logger().warn(f'set_mode {requested} rejected')

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9


def main(args=None):
    rclpy.init(args=args)
    node = ControlArbiter()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
