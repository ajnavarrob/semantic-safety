#!/usr/bin/env python3
"""
YOLO semantic-observation node for the Unitree Go2.

Responsibilities
----------------
1. Subscribe to a rectified RGB image and its organized registered point cloud.
2. Run the six-class YOLO segmentation model.
3. Preserve every semantic class independently.
4. Project each class mask into a robot-centered 2-D OccupancyGrid.
5. Publish one raw observation grid per semantic class.

This node deliberately does NOT apply language-rule buffers, temporal semantic
persistence, homotopy insertion, or semantic-layer union. Those operations
belong in the semantic map fuser/synthesis stage.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Bool
from tf2_ros import TransformException
from ultralytics import YOLO


# YOLO model class IDs are part of the perception/control interface.
# Do not reorder these without updating the dataset and downstream consumers.
SEMANTIC_CLASSES: Dict[int, str] = {
    0: "human",
    1: "traffic_cone",
    2: "caution_tape",
    3: "floor_danger_tape",
    4: "wet_floor_sign",
    5: "spill",
}

# Display colors only, in BGR order.
CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 255),
    1: (0, 165, 255),
    2: (255, 255, 0),
    3: (255, 0, 255),
    4: (0, 255, 255),
    5: (255, 0, 0),
}


class YOLODetectorNode(Node):
    """Publish raw class-specific semantic observations from RGB-D data."""

    def __init__(self) -> None:
        super().__init__("yolo_detector")

        self._declare_parameters()
        self._load_parameters()
        self._initialize_runtime_state()
        self._load_model()
        self._initialize_ros_interfaces()

        self.get_logger().info(self._startup_summary())

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _declare_parameters(self) -> None:
        # Model and inference.
        self.declare_parameter("model_path", "yolo11n-seg.pt")
        self.declare_parameter("use_tensorrt", False)
        self.declare_parameter("device", "0")
        self.declare_parameter("image_size", 640)
        self.declare_parameter("mask_threshold", 0.5)

        # The model must run at the minimum class threshold. Class-specific
        # filtering is then applied after inference.
        self.declare_parameter("human_confidence_threshold", 0.80)
        self.declare_parameter("traffic_cone_confidence_threshold", 0.15)
        self.declare_parameter("caution_tape_confidence_threshold", 0.25)
        self.declare_parameter("floor_danger_tape_confidence_threshold", 0.30)
        self.declare_parameter("wet_floor_sign_confidence_threshold", 0.30)
        self.declare_parameter("spill_confidence_threshold", 0.25)

        # Workload control.
        self.declare_parameter("enable_rule_based_gating", True) # Change to True to enable gating based on semantic_enable_topict 
        self.declare_parameter(
            "semantic_enable_topic", "/semantic_perception_required"
        )
        self.declare_parameter("process_every_n_frames", 3)
        self.declare_parameter("inference_enabled_at_startup", False) #Change to False to disable inference at startup


        # Sensor topics.
        self.declare_parameter(
            "image_topic", "/camera_front/image_rect_color"
        )
        self.declare_parameter(
            "pointcloud_topic",
            "/camera_front/point_cloud/cloud_registered",
        )

        # Output topics. Use a camera-specific prefix when running front and
        # rear nodes simultaneously, e.g. /semantic_observations/front.
        self.declare_parameter(
            "semantic_observation_prefix", "/semantic_observations/front"
        )
        self.declare_parameter(
            "segmentation_mask_topic", "/yolo/segmentation_mask"
        )
        self.declare_parameter("annotated_image_topic", "/yolo/annotated_image")
        self.declare_parameter("human_centroid_topic", "/human_tracking/centroid")
        self.declare_parameter("visibility_map_topic", "/visibility_map")

        # Temporary backward-compatibility output. Values are 0 for empty and
        # YOLO class_id + 1 for occupied semantic cells.
        self.declare_parameter("publish_legacy_class_map", True)
        self.declare_parameter("class_map_topic", "/class_map")
        self.declare_parameter("publish_annotated_image", False)

        # Projection/grid parameters. Must match poisson.h.
        self.declare_parameter("target_frame", "body_link")
        self.declare_parameter("grid_imax", 100)
        self.declare_parameter("grid_jmax", 100)
        self.declare_parameter("grid_ds", 0.05)
        self.declare_parameter("grid_size", 5.0)

        # Semantic points are already selected by image masks. In particular,
        # floor tape and spills must NOT be removed by generic ground filtering.
        self.declare_parameter("semantic_z_min", -1.0)
        self.declare_parameter("semantic_z_max", 3.5)
        self.declare_parameter("pointcloud_max_age_sec", 0.20)
        self.declare_parameter("tf_timeout_sec", 0.10)

        self.declare_parameter("logging_publish_hz", 10.0)

    def _load_parameters(self) -> None:
        self.model_path = str(self.get_parameter("model_path").value)
        self.use_tensorrt = bool(self.get_parameter("use_tensorrt").value)
        self.device = str(self.get_parameter("device").value)
        self.image_size = int(self.get_parameter("image_size").value)
        self.mask_threshold = float(self.get_parameter("mask_threshold").value)

        self.class_confidence_thresholds: Dict[int, float] = {
            0: float(self.get_parameter("human_confidence_threshold").value),
            1: float(
                self.get_parameter("traffic_cone_confidence_threshold").value
            ),
            2: float(self.get_parameter("caution_tape_confidence_threshold").value),
            3: float(
                self.get_parameter(
                    "floor_danger_tape_confidence_threshold"
                ).value
            ),
            4: float(
                self.get_parameter("wet_floor_sign_confidence_threshold").value
            ),
            5: float(self.get_parameter("spill_confidence_threshold").value),
        }
        self.global_confidence_threshold = min(
            self.class_confidence_thresholds.values()
        )

        self.enable_rule_based_gating = bool(
            self.get_parameter("enable_rule_based_gating").value
        )
        self.semantic_enable_topic = str(
            self.get_parameter("semantic_enable_topic").value
        )
        self.process_every_n_frames = max(
            1, int(self.get_parameter("process_every_n_frames").value)
        )
        self.inference_enabled = bool(
            self.get_parameter("inference_enabled_at_startup").value
        )

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.pointcloud_topic = str(self.get_parameter("pointcloud_topic").value)
        self.semantic_observation_prefix = str(
            self.get_parameter("semantic_observation_prefix").value
        ).rstrip("/")
        self.segmentation_mask_topic = str(
            self.get_parameter("segmentation_mask_topic").value
        )
        self.annotated_image_topic = str(
            self.get_parameter("annotated_image_topic").value
        )
        self.human_centroid_topic = str(
            self.get_parameter("human_centroid_topic").value
        )
        self.visibility_map_topic = str(
            self.get_parameter("visibility_map_topic").value
        )
        self.publish_legacy_class_map = bool(
            self.get_parameter("publish_legacy_class_map").value
        )
        self.class_map_topic = str(self.get_parameter("class_map_topic").value)
        self.publish_annotated_image = bool(
            self.get_parameter("publish_annotated_image").value
        )

        self.target_frame = str(self.get_parameter("target_frame").value)
        self.grid_imax = int(self.get_parameter("grid_imax").value)
        self.grid_jmax = int(self.get_parameter("grid_jmax").value)
        self.grid_ds = float(self.get_parameter("grid_ds").value)
        self.grid_size = float(self.get_parameter("grid_size").value)
        self.semantic_z_min = float(self.get_parameter("semantic_z_min").value)
        self.semantic_z_max = float(self.get_parameter("semantic_z_max").value)
        self.pointcloud_max_age_sec = float(
            self.get_parameter("pointcloud_max_age_sec").value
        )
        self.tf_timeout_sec = float(self.get_parameter("tf_timeout_sec").value)

        logging_hz = float(self.get_parameter("logging_publish_hz").value)
        self.logging_publish_period = 1.0 / logging_hz if logging_hz > 0.0 else 0.0

        if self.grid_imax <= 0 or self.grid_jmax <= 0 or self.grid_ds <= 0.0:
            raise ValueError("Grid dimensions and resolution must be positive")
        if self.semantic_z_min >= self.semantic_z_max:
            raise ValueError("semantic_z_min must be less than semantic_z_max")

    def _initialize_runtime_state(self) -> None:
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.latest_pointcloud: Optional[np.ndarray] = None
        self.latest_pc_stamp = None
        self.latest_pc_frame: Optional[str] = None

        self.frame_counter = 0
        self.received_image_count = 0
        self.inference_count = 0
        self.skipped_disabled_count = 0
        self.skipped_stride_count = 0
        self.skipped_missing_pc_count = 0
        self.skipped_stale_pc_count = 0

        self.last_image_header = None
        self.last_image_shape: Optional[Tuple[int, int]] = None
        self.last_logging_publish_time = self.get_clock().now()

    def _load_model(self) -> None:
        self.get_logger().info(
            f"Loading YOLO model: {self.model_path}; TensorRT={self.use_tensorrt}"
        )

        if self.use_tensorrt:
            if not self.model_path.endswith(".pt"):
                raise ValueError(
                    "TensorRT export expects model_path to reference a .pt checkpoint"
                )
            engine_path = os.path.splitext(self.model_path)[0] + ".engine"
            if not os.path.exists(engine_path):
                self.get_logger().info(f"Exporting TensorRT engine: {engine_path}")
                source_model = YOLO(self.model_path)
                exported_path = source_model.export(
                    format="engine",
                    device=self.device,
                    imgsz=self.image_size,
                )
                engine_path = str(exported_path)
            self.model = YOLO(engine_path)
        else:
            self.model = YOLO(self.model_path)

        model_names = {
            int(class_id): str(name)
            for class_id, name in dict(self.model.names).items()
        }
        if model_names != SEMANTIC_CLASSES:
            raise RuntimeError(
                "YOLO class mapping does not match the required semantic interface. "
                f"Expected {SEMANTIC_CLASSES}, received {model_names}"
            )

    def _initialize_ros_interfaces(self) -> None:
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.image_callback, sensor_qos
        )
        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            self.pointcloud_topic,
            self.pointcloud_callback,
            sensor_qos,
        )
        self.semantic_enable_sub = self.create_subscription(
            Bool,
            self.semantic_enable_topic,
            self.semantic_enable_callback,
            10,
        )

        self.seg_mask_pub = self.create_publisher(
            Image, self.segmentation_mask_topic, 10
        )
        self.annotated_image_pub = self.create_publisher(
            Image, self.annotated_image_topic, 10
        )
        self.human_centroid_pub = self.create_publisher(
            Point, self.human_centroid_topic, 10
        )
        self.visibility_map_pub = self.create_publisher(
            OccupancyGrid, self.visibility_map_topic, 10
        )

        self.class_map_pub = None
        if self.publish_legacy_class_map:
            self.class_map_pub = self.create_publisher(
                OccupancyGrid, self.class_map_topic, 10
            )

        self.semantic_grid_publishers = {
            class_id: self.create_publisher(
                OccupancyGrid,
                f"{self.semantic_observation_prefix}/{class_name}",
                10,
            )
            for class_id, class_name in SEMANTIC_CLASSES.items()
        }

    def _startup_summary(self) -> str:
        class_topics = "\n".join(
            f"  {name}: {self.semantic_observation_prefix}/{name}"
            for name in SEMANTIC_CLASSES.values()
        )
        return (
            "YOLO semantic-observation node initialized\n"
            f"Model: {self.model_path}\n"
            f"Classes: {SEMANTIC_CLASSES}\n"
            f"Image: {self.image_topic}\n"
            f"Point cloud: {self.pointcloud_topic}\n"
            f"Target frame: {self.target_frame}\n"
            f"Rule gating: {self.enable_rule_based_gating}\n"
            f"Inference initially enabled: {self.inference_enabled}\n"
            f"Process every N frames: {self.process_every_n_frames}\n"
            f"Semantic observation topics:\n{class_topics}"
        )

    # ------------------------------------------------------------------
    # Gating and empty-output handling
    # ------------------------------------------------------------------

    def semantic_enable_callback(self, msg: Bool) -> None:
        requested_state = bool(msg.data)
        if requested_state == self.inference_enabled:
            return

        self.inference_enabled = requested_state
        self.frame_counter = 0

        if self.inference_enabled:
            self.get_logger().info(
                "YOLO inference ENABLED; next selected camera frame will be processed"
            )
        else:
            self.get_logger().info(
                "YOLO inference DISABLED; clearing semantic observation outputs"
            )
            self.publish_empty_semantic_outputs()

    def should_process_image(self, msg: Image) -> bool:
        self.received_image_count += 1
        self.last_image_header = msg.header
        self.last_image_shape = (msg.height, msg.width)

        if self.enable_rule_based_gating and not self.inference_enabled:
            self.skipped_disabled_count += 1
            return False

        selected = (self.frame_counter % self.process_every_n_frames) == 0
        self.frame_counter += 1
        if not selected:
            self.skipped_stride_count += 1
            return False

        self.inference_count += 1
        return True

    def publish_empty_semantic_outputs(self) -> None:
        if self.last_image_header is None:
            return

        header = self.last_image_header
        header.stamp = self.get_clock().now().to_msg()

        empty_grid = np.zeros((self.grid_imax, self.grid_jmax), dtype=np.int8)
        for publisher in self.semantic_grid_publishers.values():
            publisher.publish(self._make_occupancy_grid(empty_grid, header))

        self.visibility_map_pub.publish(self._make_occupancy_grid(empty_grid, header))

        if self.class_map_pub is not None:
            self.class_map_pub.publish(self._make_occupancy_grid(empty_grid, header))

        if self.last_image_shape is not None:
            height, width = self.last_image_shape
            empty_mask = np.zeros((height, width), dtype=np.uint8)
            mask_msg = self.bridge.cv2_to_imgmsg(empty_mask, encoding="mono8")
            mask_msg.header = header
            self.seg_mask_pub.publish(mask_msg)

    # ------------------------------------------------------------------
    # Point-cloud parsing and transforms
    # ------------------------------------------------------------------

    def pointcloud_callback(self, msg: PointCloud2) -> None:
        try:
            if msg.height <= 1:
                self.get_logger().warn(
                    "Received unorganized point cloud; semantic projection requires HxW data",
                    throttle_duration_sec=5.0,
                )
                self.latest_pointcloud = None
                return

            field_offsets = {
                field.name: field.offset
                for field in msg.fields
                if field.name in ("x", "y", "z")
            }
            if set(field_offsets) != {"x", "y", "z"}:
                raise ValueError(
                    f"PointCloud2 lacks x/y/z fields: {sorted(field_offsets)}"
                )

            raw = np.frombuffer(msg.data, dtype=np.uint8)
            x_off = field_offsets["x"]
            y_off = field_offsets["y"]
            z_off = field_offsets["z"]

            if y_off == x_off + 4 and z_off == x_off + 8:
                points = np.ndarray(
                    shape=(msg.height, msg.width, 3),
                    dtype=np.float32,
                    buffer=raw,
                    strides=(msg.row_step, msg.point_step, 4),
                    offset=x_off,
                ).copy()
            else:
                axes: List[np.ndarray] = []
                for offset in (x_off, y_off, z_off):
                    axis = np.ndarray(
                        shape=(msg.height, msg.width),
                        dtype=np.float32,
                        buffer=raw,
                        strides=(msg.row_step, msg.point_step),
                        offset=offset,
                    ).copy()
                    axes.append(axis)
                points = np.stack(axes, axis=-1)

            self.latest_pointcloud = points
            self.latest_pc_stamp = msg.header.stamp
            self.latest_pc_frame = msg.header.frame_id

        except Exception as exc:  # noqa: BLE001 - ROS callback boundary
            self.get_logger().error(f"Point-cloud callback failed: {exc}")
            self.latest_pointcloud = None

    @staticmethod
    def _stamp_to_seconds(stamp) -> float:
        return float(stamp.sec) + 1.0e-9 * float(stamp.nanosec)

    def _pointcloud_is_usable(self, image_msg: Image) -> bool:
        if (
            self.latest_pointcloud is None
            or self.latest_pc_stamp is None
            or not self.latest_pc_frame
        ):
            self.skipped_missing_pc_count += 1
            return False

        age_sec = abs(
            self._stamp_to_seconds(image_msg.header.stamp)
            - self._stamp_to_seconds(self.latest_pc_stamp)
        )
        if age_sec > self.pointcloud_max_age_sec:
            self.skipped_stale_pc_count += 1
            self.get_logger().warn(
                f"Skipping semantic projection: RGB/point-cloud age={age_sec:.3f}s",
                throttle_duration_sec=2.0,
            )
            return False

        return True

    def _lookup_camera_to_body_transform(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.latest_pc_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=self.tf_timeout_sec),
            )
        except TransformException as exc:
            self.get_logger().warn(
                f"TF lookup failed ({self.latest_pc_frame} -> {self.target_frame}): {exc}",
                throttle_duration_sec=5.0,
            )
            return None

        trans = transform.transform.translation
        rot = transform.transform.rotation
        qx, qy, qz, qw = rot.x, rot.y, rot.z, rot.w

        rotation = np.array(
            [
                [
                    1.0 - 2.0 * (qy * qy + qz * qz),
                    2.0 * (qx * qy - qw * qz),
                    2.0 * (qx * qz + qw * qy),
                ],
                [
                    2.0 * (qx * qy + qw * qz),
                    1.0 - 2.0 * (qx * qx + qz * qz),
                    2.0 * (qy * qz - qw * qx),
                ],
                [
                    2.0 * (qx * qz - qw * qy),
                    2.0 * (qy * qz + qw * qx),
                    1.0 - 2.0 * (qx * qx + qy * qy),
                ],
            ],
            dtype=np.float32,
        )
        translation = np.array([trans.x, trans.y, trans.z], dtype=np.float32)
        return rotation, translation

    # ------------------------------------------------------------------
    # Inference and projection
    # ------------------------------------------------------------------

    def image_callback(self, msg: Image) -> None:
        if not self.should_process_image(msg):
            return

        total_start = time.perf_counter()
        try:
            cv_start = time.perf_counter()
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            cv_ms = (time.perf_counter() - cv_start) * 1000.0

            infer_start = time.perf_counter()
            results = self.model.predict(
                cv_image,
                conf=self.global_confidence_threshold,
                imgsz=self.image_size,
                verbose=False,
                show=False,
                device=self.device,
            )
            infer_ms = (time.perf_counter() - infer_start) * 1000.0

            class_masks, detections = self._extract_class_masks(
                results, cv_image.shape[:2]
            )

            encoded_mask = self._make_encoded_debug_mask(class_masks)
            mask_msg = self.bridge.cv2_to_imgmsg(encoded_mask, encoding="mono8")
            mask_msg.header = msg.header
            self.seg_mask_pub.publish(mask_msg)

            self._publish_human_centroid(class_masks[0], cv_image.shape[:2])

            projection_start = time.perf_counter()
            semantic_grids, visibility_grid = self._project_masks_to_grids(
                class_masks, msg
            )
            projection_ms = (time.perf_counter() - projection_start) * 1000.0

            for class_id, publisher in self.semantic_grid_publishers.items():
                publisher.publish(
                    self._make_occupancy_grid(semantic_grids[class_id], msg.header)
                )

            self.visibility_map_pub.publish(
                self._make_occupancy_grid(visibility_grid, msg.header)
            )

            if self.class_map_pub is not None:
                legacy_grid = self._compose_legacy_class_map(semantic_grids)
                self.class_map_pub.publish(
                    self._make_occupancy_grid(legacy_grid, msg.header)
                )

            if self.publish_annotated_image and self._logging_publish_due():
                annotated = self._draw_annotations(cv_image, detections)
                annotated_msg = self.bridge.cv2_to_imgmsg(
                    annotated, encoding="bgr8"
                )
                annotated_msg.header = msg.header
                self.annotated_image_pub.publish(annotated_msg)

            total_ms = (time.perf_counter() - total_start) * 1000.0
            occupied_counts = ", ".join(
                f"{SEMANTIC_CLASSES[class_id]}={int(np.count_nonzero(grid))}"
                for class_id, grid in semantic_grids.items()
            )
            self.get_logger().info(
                "YOLO timing | "
                f"total={total_ms:.1f} ms cv={cv_ms:.1f} ms "
                f"infer={infer_ms:.1f} ms projection={projection_ms:.1f} ms | "
                f"detections={len(detections)} cells[{occupied_counts}] | "
                f"received={self.received_image_count} inferred={self.inference_count} "
                f"disabled={self.skipped_disabled_count} stride={self.skipped_stride_count} "
                f"missing_pc={self.skipped_missing_pc_count} stale_pc={self.skipped_stale_pc_count}",
                throttle_duration_sec=1.0,
            )

        except Exception as exc:  # noqa: BLE001 - ROS callback boundary
            self.get_logger().error(f"Image callback failed: {exc}")

    def _extract_class_masks(
        self, results, image_shape: Tuple[int, int]
    ) -> Tuple[Dict[int, np.ndarray], List[Tuple[int, float, np.ndarray]]]:
        height, width = image_shape
        class_masks = {
            class_id: np.zeros((height, width), dtype=bool)
            for class_id in SEMANTIC_CLASSES
        }
        detections: List[Tuple[int, float, np.ndarray]] = []

        if not results:
            return class_masks, detections

        result = results[0]
        if result.masks is None or result.boxes is None:
            return class_masks, detections

        masks = result.masks.data
        boxes = result.boxes

        for mask, cls, conf, xyxy in zip(
            masks, boxes.cls, boxes.conf, boxes.xyxy
        ):
            class_id = int(cls.item())
            confidence = float(conf.item())
            if class_id not in SEMANTIC_CLASSES:
                continue
            if confidence < self.class_confidence_thresholds[class_id]:
                continue

            mask_np = mask.detach().cpu().numpy()
            mask_resized = cv2.resize(
                mask_np,
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )
            binary_mask = mask_resized > self.mask_threshold
            class_masks[class_id] |= binary_mask

            box = xyxy.detach().cpu().numpy().astype(int)
            detections.append((class_id, confidence, box))

        return class_masks, detections

    def _project_masks_to_grids(
        self,
        class_masks: Dict[int, np.ndarray],
        image_msg: Image,
    ) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
        semantic_grids = {
            class_id: np.zeros((self.grid_imax, self.grid_jmax), dtype=np.int8)
            for class_id in SEMANTIC_CLASSES
        }
        visibility_grid = np.zeros(
            (self.grid_imax, self.grid_jmax), dtype=np.int8
        )

        if not any(mask.any() for mask in class_masks.values()):
            return semantic_grids, visibility_grid
        if not self._pointcloud_is_usable(image_msg):
            return semantic_grids, visibility_grid

        transform = self._lookup_camera_to_body_transform()
        if transform is None:
            return semantic_grids, visibility_grid
        rotation, translation = transform

        xyz = self.latest_pointcloud
        assert xyz is not None
        pc_h, pc_w = xyz.shape[:2]

        for class_id, image_mask in class_masks.items():
            if not image_mask.any():
                continue

            if image_mask.shape != (pc_h, pc_w):
                mask_pc = cv2.resize(
                    image_mask.astype(np.uint8),
                    (pc_w, pc_h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            else:
                mask_pc = image_mask

            points_camera = xyz[mask_pc]
            if points_camera.size == 0:
                continue

            valid = np.isfinite(points_camera).all(axis=1)
            points_camera = points_camera[valid]
            if points_camera.size == 0:
                continue

            points_body = (rotation @ points_camera.T).T + translation

            # These are semantic points selected by the segmentation mask.
            # Keep floor-level detections such as tape and spills.
            valid_bounds = (
                (points_body[:, 2] > self.semantic_z_min)
                & (points_body[:, 2] < self.semantic_z_max)
                & (points_body[:, 0] > -self.grid_size / 2.0)
                & (points_body[:, 0] < self.grid_size / 2.0)
                & (points_body[:, 1] > -self.grid_size / 2.0)
                & (points_body[:, 1] < self.grid_size / 2.0)
            )
            points_body = points_body[valid_bounds]
            if points_body.size == 0:
                continue

            grid_rows = (
                self.grid_imax // 2
                + np.floor(points_body[:, 1] / self.grid_ds).astype(np.int32)
            )
            grid_cols = (
                self.grid_jmax // 2
                + np.floor(points_body[:, 0] / self.grid_ds).astype(np.int32)
            )

            in_bounds = (
                (grid_rows >= 0)
                & (grid_rows < self.grid_imax)
                & (grid_cols >= 0)
                & (grid_cols < self.grid_jmax)
            )
            grid_rows = grid_rows[in_bounds]
            grid_cols = grid_cols[in_bounds]
            if grid_rows.size == 0:
                continue

            semantic_grids[class_id][grid_rows, grid_cols] = 100
            visibility_grid[grid_rows, grid_cols] = 100

        return semantic_grids, visibility_grid

    # ------------------------------------------------------------------
    # Message construction and debug outputs
    # ------------------------------------------------------------------

    def _make_occupancy_grid(self, grid: np.ndarray, header) -> OccupancyGrid:
        msg = OccupancyGrid()
        msg.header = header
        msg.header.frame_id = self.target_frame
        msg.info.resolution = self.grid_ds
        msg.info.width = self.grid_jmax
        msg.info.height = self.grid_imax
        msg.info.origin.position.x = -0.5 * self.grid_jmax * self.grid_ds
        msg.info.origin.position.y = -0.5 * self.grid_imax * self.grid_ds
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.x = 0.0
        msg.info.origin.orientation.y = 0.0
        msg.info.origin.orientation.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = grid.astype(np.int8, copy=False).reshape(-1).tolist()
        return msg

    def _make_encoded_debug_mask(
        self, class_masks: Dict[int, np.ndarray]
    ) -> np.ndarray:
        shape = next(iter(class_masks.values())).shape
        encoded = np.zeros(shape, dtype=np.uint8)

        # Lower-priority classes are written first; human is written last.
        priority = [5, 3, 2, 4, 1, 0]
        for class_id in priority:
            encoded[class_masks[class_id]] = class_id + 1
        return encoded

    def _compose_legacy_class_map(
        self, semantic_grids: Dict[int, np.ndarray]
    ) -> np.ndarray:
        combined = np.zeros((self.grid_imax, self.grid_jmax), dtype=np.int8)
        priority = [5, 3, 2, 4, 1, 0]
        for class_id in priority:
            combined[semantic_grids[class_id] > 0] = class_id + 1
        return combined

    def _publish_human_centroid(
        self, human_mask: np.ndarray, image_shape: Tuple[int, int]
    ) -> None:
        height, width = image_shape
        centroid = Point()
        if human_mask.any():
            ys, xs = np.where(human_mask)
            centroid.x = float(np.mean(xs))
            centroid.y = float(np.mean(ys))
        else:
            centroid.x = 0.5 * float(width)
            centroid.y = 0.5 * float(height)
        centroid.z = float(width)
        self.human_centroid_pub.publish(centroid)

    def _draw_annotations(
        self,
        image: np.ndarray,
        detections: List[Tuple[int, float, np.ndarray]],
    ) -> np.ndarray:
        annotated = image.copy()
        for class_id, confidence, box in detections:
            x1, y1, x2, y2 = box.tolist()
            color = CLASS_COLORS[class_id]
            label = f"{SEMANTIC_CLASSES[class_id]}: {confidence:.2f}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            label_top = max(0, y1 - label_h - 6)
            cv2.rectangle(
                annotated,
                (x1, label_top),
                (x1 + label_w, y1),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1, max(label_h, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
        return annotated

    def _logging_publish_due(self) -> bool:
        if self.logging_publish_period <= 0.0:
            return True
        now = self.get_clock().now()
        elapsed = (now - self.last_logging_publish_time).nanoseconds * 1.0e-9
        if elapsed < self.logging_publish_period:
            return False
        self.last_logging_publish_time = now
        return True


def main(args=None) -> None:
    rclpy.init(args=args)
    node = YOLODetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
