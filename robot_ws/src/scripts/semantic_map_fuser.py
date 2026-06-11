#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid


class SemanticMapFuser(Node):
    def __init__(self):
        super().__init__('semantic_map_fuser')

        self.front_class_map = None
        self.rear_class_map = None
        self.front_visibility_map = None
        self.rear_visibility_map = None

        self.create_subscription(
            OccupancyGrid,
            '/class_map_front',
            self.front_class_callback,
            10
        )

        self.create_subscription(
            OccupancyGrid,
            '/class_map_rear',
            self.rear_class_callback,
            10
        )

        self.create_subscription(
            OccupancyGrid,
            '/visibility_map_front',
            self.front_visibility_callback,
            10
        )

        self.create_subscription(
            OccupancyGrid,
            '/visibility_map_rear',
            self.rear_visibility_callback,
            10
        )

        self.class_map_pub = self.create_publisher(
            OccupancyGrid,
            '/class_map',
            10
        )

        self.visibility_map_pub = self.create_publisher(
            OccupancyGrid,
            '/visibility_map',
            10
        )

        self.timer = self.create_timer(0.1, self.publish_fused_maps)

        self.get_logger().info('Semantic map fuser initialized')

    def front_class_callback(self, msg):
        self.front_class_map = msg

    def rear_class_callback(self, msg):
        self.rear_class_map = msg

    def front_visibility_callback(self, msg):
        self.front_visibility_map = msg

    def rear_visibility_callback(self, msg):
        self.rear_visibility_map = msg

    def fuse_maps(self, map_a, map_b):
        if map_a is None and map_b is None:
            return None

        if map_a is None:
            return map_b

        if map_b is None:
            return map_a

        if len(map_a.data) != len(map_b.data):
            self.get_logger().warn(
                'Map sizes do not match; using front map only',
                throttle_duration_sec=2.0
            )
            return map_a

        fused = OccupancyGrid()
        fused.header = map_a.header
        fused.info = map_a.info

        fused.data = [
            max(a, b)
            for a, b in zip(map_a.data, map_b.data)
        ]

        return fused

    def publish_fused_maps(self):
        fused_class = self.fuse_maps(
            self.front_class_map,
            self.rear_class_map
        )

        fused_visibility = self.fuse_maps(
            self.front_visibility_map,
            self.rear_visibility_map
        )

        if fused_class is not None:
            fused_class.header.stamp = self.get_clock().now().to_msg()
            fused_class.header.frame_id = 'body_link'
            self.class_map_pub.publish(fused_class)

        if fused_visibility is not None:
            fused_visibility.header.stamp = self.get_clock().now().to_msg()
            fused_visibility.header.frame_id = 'body_link'
            self.visibility_map_pub.publish(fused_visibility)


def main(args=None):
    rclpy.init(args=args)
    node = SemanticMapFuser()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()