#!/usr/bin/env python3
"""Publish the boat's global position as NavSatFix.

When *boat_global_position_API* is set, the node polls that HTTP endpoint at
1 Hz and publishes the returned lat/lon/alt.  When the parameter is null (or
the request fails), it falls back to the static *boat_latitude*,
*boat_longitude*, *boat_altitude* parameters.
"""

import urllib.request
import json

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float64

POLL_PERIOD_SEC = 1.0


class GlobalBoatPositionInterpreter(Node):
    def __init__(self):
        super().__init__('global_boat_position_interpreter')

        self.declare_parameter('boat_position_global_topic', '')
        self.declare_parameter('boat_orientation_topic', '')
        self.declare_parameter('boat_latitude', 0.0)
        self.declare_parameter('boat_longitude', 0.0)
        self.declare_parameter('boat_altitude', 0.0)
        self.declare_parameter('boat_orientation_deg', 0.0)
        self.declare_parameter('boat_global_position_API', '')

        out_topic = self.get_parameter('boat_position_global_topic').value
        orientation_topic = self.get_parameter('boat_orientation_topic').value
        self._default_lat = self.get_parameter('boat_latitude').value
        self._default_lon = self.get_parameter('boat_longitude').value
        self._default_alt = self.get_parameter('boat_altitude').value
        self._default_orientation_deg = self.get_parameter('boat_orientation_deg').value

        api_raw = self.get_parameter('boat_global_position_API').value
        self._api_url = api_raw if api_raw else None

        self._pub = self.create_publisher(NavSatFix, out_topic, 10)
        self._orientation_pub = self.create_publisher(Float64, orientation_topic, 10)
        self.create_timer(POLL_PERIOD_SEC, self._tick)

        if self._api_url:
            self.get_logger().info(
                f'polling {self._api_url} -> {out_topic}')
        else:
            self.get_logger().info(
                f'static position ({self._default_lat}, {self._default_lon}, '
                f'{self._default_alt}) -> {out_topic}')

    def _fetch_from_api(self):
        """GET the API and return (lat, lon, alt) or None on failure."""
        try:
            with urllib.request.urlopen(self._api_url, timeout=2) as resp:
                data = json.loads(resp.read())
            return (
                float(data['latitude']),
                float(data['longitude']),
                float(data.get('altitude', self._default_alt)),
                float(data.get('orientation_deg', self._default_orientation_deg)),
            )
        except Exception as exc:
            self.get_logger().warn(
                f'API request failed ({exc}); using default position',
                throttle_duration_sec=10.0,
            )
            return None

    def _tick(self):
        if self._api_url:
            result = self._fetch_from_api()
            if result:
                lat, lon, alt, orientation_deg = result
            else:
                lat, lon, alt, orientation_deg = (
                    self._default_lat,
                    self._default_lon,
                    self._default_alt,
                    self._default_orientation_deg,
                )
        else:
            lat, lon, alt, orientation_deg = (
                self._default_lat,
                self._default_lon,
                self._default_alt,
                self._default_orientation_deg,
            )

        msg = NavSatFix()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.latitude = lat
        msg.longitude = lon
        msg.altitude = alt
        self._pub.publish(msg)

        orientation_msg = Float64()
        orientation_msg.data = orientation_deg
        self._orientation_pub.publish(orientation_msg)


def main(args=None):
    rclpy.init(args=args)
    node = GlobalBoatPositionInterpreter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
