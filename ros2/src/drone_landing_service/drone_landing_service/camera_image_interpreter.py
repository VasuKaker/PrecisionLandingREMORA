#!/usr/bin/env python3
"""SIM: relay image_topic -> video_feed_topic. REAL: capture camera -> publish."""

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class CameraImageInterpreter(Node):
    def __init__(self):
        super().__init__('camera_image_interpreter')

        self.declare_parameter('image_topic', '')
        self.declare_parameter('video_feed_topic', '')
        self.declare_parameter('Camera_Parameters.video_device', '')
        self.declare_parameter('Camera_Parameters.width', 640)
        self.declare_parameter('Camera_Parameters.height', 480)
        self.declare_parameter('Camera_Parameters.fps', 30)

        image_topic = self.get_parameter('image_topic').value
        video_feed_topic = self.get_parameter('video_feed_topic').value
        video_device = self.get_parameter('Camera_Parameters.video_device').value
        width = int(self.get_parameter('Camera_Parameters.width').value)
        height = int(self.get_parameter('Camera_Parameters.height').value)
        fps = int(self.get_parameter('Camera_Parameters.fps').value)

        self._bridge = CvBridge()
        self._image_topic = image_topic
        self._video_feed_topic = video_feed_topic
        self._relay_enabled = bool(video_feed_topic) and video_feed_topic != image_topic
        self._pub = self.create_publisher(Image, video_feed_topic, 10) if self._relay_enabled else None
        self._raw_pub = self.create_publisher(Image, image_topic, 10)
        self._cap = None

        # REAL mode: Camera_Parameters provided (e.g. /dev/video0), capture and publish.
        if video_device:
            self._cap = cv2.VideoCapture(video_device, cv2.CAP_V4L2)
            if not self._cap.isOpened():
                self._cap = cv2.VideoCapture(video_device)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._cap.set(cv2.CAP_PROP_FPS, fps)
            if hasattr(cv2, 'CAP_PROP_BUFFERSIZE'):
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if hasattr(cv2, 'CAP_PROP_FOURCC'):
                self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            period = 1.0 / float(max(1, fps))
            self.create_timer(period, self._on_camera_tick)
            if self._relay_enabled:
                self.get_logger().info(
                    f'camera {video_device} ({width}x{height}@{fps}) -> '
                    f'{image_topic} (relay {video_feed_topic})'
                )
            else:
                self.get_logger().info(
                    f'camera {video_device} ({width}x{height}@{fps}) -> {image_topic}'
                )
            if not self._cap.isOpened():
                self.get_logger().error(f'failed to open camera device {video_device}')
        else:
            # SIM mode: just relay bridged images.
            if self._relay_enabled:
                self.create_subscription(Image, image_topic, self._on_image, 10)
                self.get_logger().info(f'{image_topic} -> {video_feed_topic}')
            else:
                self.get_logger().info(f'using upstream image topic directly: {image_topic}')

    def _on_image(self, msg: Image):
        if self._pub is not None:
            self._pub.publish(msg)

    def _on_camera_tick(self):
        if self._cap is None:
            return
        ok, frame = self._cap.read()
        if not ok:
            return
        msg = self._bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        self._raw_pub.publish(msg)
        if self._pub is not None:
            self._pub.publish(msg)

    def destroy_node(self):
        if self._cap is not None:
            self._cap.release()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(CameraImageInterpreter())
    rclpy.shutdown()
