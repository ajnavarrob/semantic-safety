#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class SegmentationAdapter(Node):
    """
    Converts the colorized DepthAI segmentation output into a mono8 class map.

    Temporary POC behavior:
      black pixel     -> class 0, background
      non-black pixel -> class 1, segmented foreground

    The neural-network inference still runs entirely on the OAK-1.
    """

    def __init__(self) -> None:
        super().__init__("segmentation_adapter")

        self.declare_parameter(
            "input_topic",
            "/oak_left/nn/image_raw",
        )
        self.declare_parameter(
            "output_topic",
            "/oak_left/segmentation/class_map",
        )

        self.declare_parameter(
            "minimum_color_intensity",
            5,
        )

        input_topic = (
            self.get_parameter("input_topic")
            .get_parameter_value()
            .string_value
        )

        output_topic = (
            self.get_parameter("output_topic")
            .get_parameter_value()
            .string_value
        )

        self.minimum_color_intensity = int(
            self.get_parameter("minimum_color_intensity")
            .get_parameter_value()
            .integer_value
        )

        self.publisher = self.create_publisher(
            Image,
            output_topic,
            1,
        )

        self.subscription = self.create_subscription(
            Image,
            input_topic,
            self.image_callback,
            rclpy.qos.qos_profile_sensor_data,
        )

        self.get_logger().info(
            f"Segmentation adapter: {input_topic} -> {output_topic}"
        )

    def image_callback(self, msg: Image) -> None:
        if msg.encoding.lower() != "bgr8":
            self.get_logger().error(
                f"Expected bgr8 input, received '{msg.encoding}'"
            )
            return

        required_size = int(msg.height) * int(msg.step)

        if len(msg.data) < required_size:
            self.get_logger().error(
                "Image buffer is smaller than height * step."
            )
            return

        raw = np.frombuffer(msg.data, dtype=np.uint8)
        raw = raw.reshape((msg.height, msg.step))

        packed_width = int(msg.width) * 3
        bgr = raw[:, :packed_width]
        bgr = bgr.reshape((msg.height, msg.width, 3))

        # Ignore tiny nonzero values in case the visualization contains
        # compression or interpolation artifacts.
        foreground = np.max(bgr, axis=2) >= self.minimum_color_intensity

        class_map = np.zeros(
            (msg.height, msg.width),
            dtype=np.uint8,
        )
        class_map[foreground] = 1

        output = Image()
        output.header = msg.header
        output.height = msg.height
        output.width = msg.width
        output.encoding = "mono8"
        output.is_bigendian = 0
        output.step = msg.width
        output.data = class_map.tobytes()

        self.publisher.publish(output)


def main(args=None) -> None:
    rclpy.init(args=args)

    node = SegmentationAdapter()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()