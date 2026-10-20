#!/usr/bin/env python3
"""Fuse /mavros/altitude and /mavros/global_position/global into a single lat/lon/alt topic."""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float64


class DroneGlobalPositionInterpreter(Node):
    def __init__(self):
        super().__init__('drone_global_position_interpreter')

        self.declare_parameter('altitude_topic', '')
        self.declare_parameter('drone_gps_topic', '')
        self.declare_parameter('drone_global_position_topic', '')

        alt_topic = self.get_parameter('altitude_topic').value
        gps_topic = self.get_parameter('drone_gps_topic').value
        out_topic = self.get_parameter('drone_global_position_topic').value

        self._altitude = None

        self._pub = self.create_publisher(NavSatFix, out_topic, 10)
        self.create_subscription(Float64, alt_topic, self._on_altitude, qos_profile_sensor_data)
        self.create_subscription(NavSatFix, gps_topic, self._on_gps, qos_profile_sensor_data)

        self.get_logger().info(f'fusing {alt_topic} + {gps_topic} -> {out_topic}')

    def _on_altitude(self, msg: Float64):
        if math.isfinite(msg.data):
            self._altitude = msg.data

    def _on_gps(self, msg: NavSatFix):
        out = NavSatFix()
        out.header = msg.header
        out.latitude = msg.latitude
        out.longitude = msg.longitude
        if self._altitude is not None:
            out.altitude = self._altitude
        elif math.isfinite(msg.altitude):
            out.altitude = msg.altitude
        else:
            self.get_logger().warn(
                'dropping GPS fix with non-finite altitude',
                throttle_duration_sec=5.0,
            )
            return
        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(DroneGlobalPositionInterpreter())
    rclpy.shutdown()
