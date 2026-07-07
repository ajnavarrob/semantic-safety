import rclpy
from rclpy.node import Node

import cv2
import numpy as np

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from cv_bridge import CvBridge

import sensor_msgs_py.point_cloud2 as pc2

import tf2_ros
from geometry_msgs.msg import TransformStamped


class ProjectLidarDebugNode(Node):
    def __init__(self):
        super().__init__("project_lidar_debug_node")

        self.declare_parameter("lidar_topic", "/livox/lidar")
        self.declare_parameter("front_image_topic", "/oak_front/rgb/image_rect")
        self.declare_parameter("front_camera_info_topic", "/oak_front/rgb/camera_info")
        self.declare_parameter("front_output_image_topic", "/debug/oak_front_projected_lidar_image")
        self.declare_parameter("front_camera_frame", "oak_front_camera_optical_frame")
        self.declare_parameter("min_projection_depth_m", 0.15)
        self.declare_parameter("max_projection_depth_m", 12.0)
        self.declare_parameter("point_stride", 3)

        self.lidar_topic = self.get_parameter("lidar_topic").value
        self.image_topic = self.get_parameter("front_image_topic").value
        self.camera_info_topic = self.get_parameter("front_camera_info_topic").value
        self.output_topic = self.get_parameter("front_output_image_topic").value
        self.camera_frame = self.get_parameter("front_camera_frame").value
        self.min_depth = float(self.get_parameter("min_projection_depth_m").value)
        self.max_depth = float(self.get_parameter("max_projection_depth_m").value)
        self.point_stride = int(self.get_parameter("point_stride").value)

        self.bridge = CvBridge()

        self.latest_cloud = None
        self.latest_camera_info = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10,
        )

        self.info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            10,
        )

        self.cloud_sub = self.create_subscription(
            PointCloud2,
            self.lidar_topic,
            self.cloud_callback,
            10,
        )

        self.debug_pub = self.create_publisher(Image, self.output_topic, 10)

        self.get_logger().info(f"Projecting {self.lidar_topic} into {self.image_topic}")
        self.get_logger().info(f"Camera frame: {self.camera_frame}")

    def camera_info_callback(self, msg):
        self.latest_camera_info = msg

    def cloud_callback(self, msg):
        self.latest_cloud = msg
    
    def transform_to_matrix(self, transform_msg: TransformStamped):
        q = transform_msg.transform.rotation
        tx = transform_msg.transform.translation.x
        ty = transform_msg.transform.translation.y
        tz = transform_msg.transform.translation.z

        x = q.x
        y = q.y
        z = q.z
        w = q.w

        R = np.array([
            [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,         2*x*z + 2*y*w],
            [2*x*y + 2*z*w,         1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
            [2*x*z - 2*y*w,         2*y*z + 2*x*w,         1 - 2*x*x - 2*y*y],
        ])

        t = np.array([tx, ty, tz])

        return R, t

    def image_callback(self, image_msg):
        if self.latest_cloud is None or self.latest_camera_info is None:
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                self.camera_frame,
                self.latest_cloud.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05),
            )
        except Exception as exc:
            self.get_logger().warn(f"TF lookup failed: {exc}", throttle_duration_sec=1.0)
            return

        cloud_cam = self.latest_cloud

        image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")

        k = np.array(self.latest_camera_info.k).reshape(3, 3)
        fx = k[0, 0]
        fy = k[1, 1]
        cx = k[0, 2]
        cy = k[1, 2]

        height, width = image.shape[:2]

        projected_count = 0

        R, t = self.transform_to_matrix(transform)

        points = pc2.read_points(
            self.latest_cloud,
            field_names=("x", "y", "z"),
            skip_nans=True,
        )

        for i, p in enumerate(points):
            if i % self.point_stride != 0:
                continue

            p_lidar = np.array([float(p[0]), float(p[1]), float(p[2])])
            p_cam = R @ p_lidar + t

            x, y, z = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])

            if z <= self.min_depth or z >= self.max_depth:
                continue

            u = int((fx * x / z) + cx)
            v = int((fy * y / z) + cy)

            if 0 <= u < width and 0 <= v < height:
                cv2.circle(image, (u, v), 1, (0, 255, 0), -1)
                projected_count += 1

        cv2.putText(
            image,
            f"projected points: {projected_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        out_msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        out_msg.header = image_msg.header
        self.debug_pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ProjectLidarDebugNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()