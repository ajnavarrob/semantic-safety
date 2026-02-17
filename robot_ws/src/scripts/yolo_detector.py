#!/usr/bin/env python3
"""
YOLO Detection Node
Subscribes to ZED camera images, runs YOLO detection, and publishes segmentation masks.
"""

# IMPORTANT: Preload libgomp before importing torch/ultralytics to fix TLS error
import ctypes
ctypes.CDLL('/home/unitree/miniconda3/envs/semantic-safety/lib/libgomp.so.1.0.0', mode=ctypes.RTLD_GLOBAL)
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point, TransformStamped
# Removed: unitree_go.msg.SportModeState - now using TF for robot pose
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import tf2_ros
from tf2_ros import TransformException


class YOLODetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        
        # Parameters
        self.declare_parameter('model_path', 'yolo11n-seg.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('use_tensorrt', False)
        
        model_path = self.get_parameter('model_path').value
        self.conf_threshold = self.get_parameter('confidence_threshold').value
        use_tensorrt = self.get_parameter('use_tensorrt').value
        
        # Initialize YOLO model
        self.get_logger().info(f'Loading YOLO model: {model_path}, using TensorRT: {use_tensorrt}')
        
        # Export to TensorRT if requested (one-time)
        if use_tensorrt:
            # replace .pt with .engine
            engine_path = model_path.replace('.pt', '.engine')
            # check if file exists
            if os.path.exists(engine_path):
                self.get_logger().info(f'TensorRT engine already exists: {engine_path}')
                self.model = YOLO(engine_path)
                self.get_logger().info(f'TensorRT engine loaded: {engine_path}')
            else:
                self.get_logger().info('Exporting to TensorRT engine...')
                self.model.export(format='engine', device=0)
                self.model = YOLO(engine_path)
                self.get_logger().info(f'TensorRT engine loaded: {engine_path}')
        else:
            self.model = YOLO(model_path)
        
        # CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # TF2 for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Storage for latest pointcloud (synchronized with image)
        self.latest_pointcloud = None
        self.latest_pc_stamp = None
        self.latest_pc_frame = None
        
        # Robot position in odom frame (for robot-relative grid coordinates)
        self.robot_pos = np.array([0.0, 0.0, 0.0])
        self.robot_yaw = 0.0  # Robot yaw in odom frame
        
        # Point cloud filtering parameters (matching yolo_zed_ros.py)
        self.declare_parameter('floor_threshold', -1.0)
        self.declare_parameter('ceiling_threshold', 3.5)
        self.declare_parameter('height_min', 0.1)
        self.declare_parameter('height_max', 10.0)
        self.declare_parameter('forward_min', 0.1)
        self.declare_parameter('grid_size', 5.0)
        
        self.floor_threshold = self.get_parameter('floor_threshold').value
        self.ceiling_threshold = self.get_parameter('ceiling_threshold').value
        self.height_min = self.get_parameter('height_min').value
        self.height_max = self.get_parameter('height_max').value
        self.forward_min = self.get_parameter('forward_min').value
        self.grid_size = self.get_parameter('grid_size').value
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_rect_color',
            self.image_callback,
            10
        )
        
        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            '/camera/point_cloud/cloud_registered',
            self.pointcloud_callback,
            10
        )
        
        # Robot pose from TF (odom -> body_link), provided by odom_publisher.py
        # No longer subscribe to sportmodestate directly
        
        # Publishers
        self.seg_mask_pub = self.create_publisher(
            Image,
            '/yolo/segmentation_mask',
            10
        )
        
        # Publisher for human centroid (for gimbal tracking)
        self.human_centroid_pub = self.create_publisher(
            Point,
            '/human_tracking/centroid',
            10
        )
        
        # Publisher for class map (for semantic safety field)
        self.class_map_pub = self.create_publisher(
            OccupancyGrid,
            '/class_map',
            10
        )
        
        # Publisher for camera visibility mask (cells the camera can see)
        self.visibility_map_pub = self.create_publisher(
            OccupancyGrid,
            '/visibility_map',
            10
        )
        
        # Publisher for annotated RGB image with YOLO bounding boxes
        self.annotated_image_pub = self.create_publisher(
            Image,
            '/yolo/annotated_image',
            10
        )
        
        # Grid parameters (must match poisson.h)
        self.declare_parameter('grid_imax', 100)
        self.declare_parameter('grid_jmax', 100)
        self.declare_parameter('grid_ds', 0.05)
        self.grid_imax = self.get_parameter('grid_imax').value
        self.grid_jmax = self.get_parameter('grid_jmax').value
        self.grid_ds = self.get_parameter('grid_ds').value
        
        # Logging publish rate limiting
        self.declare_parameter('logging_publish_hz', 10.0)
        self.logging_publish_hz = self.get_parameter('logging_publish_hz').value
        self.logging_publish_period = 1.0 / self.logging_publish_hz if self.logging_publish_hz > 0 else 0.0
        self.last_logging_publish_time = self.get_clock().now()
        
        self.get_logger().info(f'YOLO Detector Node initialized (logging publish rate: {self.logging_publish_hz} Hz)')
    
    def pointcloud_callback(self, msg):
        """Store latest organized pointcloud for synchronization with image."""
        try:
            # Parse organized pointcloud (width × height structure)
            # RealSense publishes organized clouds where each pixel has corresponding 3D point
            if msg.height > 1:  # Organized pointcloud
                # Fast zero-copy numpy access from raw PointCloud2 buffer
                # Find xyz field offsets from message metadata
                field_offsets = {}
                for field in msg.fields:
                    if field.name in ('x', 'y', 'z'):
                        field_offsets[field.name] = field.offset
                
                data = np.frombuffer(msg.data, dtype=np.uint8)
                
                # Check if x,y,z are contiguous float32 fields (offset x, x+4, x+8)
                x_off = field_offsets['x']
                y_off = field_offsets['y']
                z_off = field_offsets['z']
                
                if y_off == x_off + 4 and z_off == x_off + 8:
                    # Contiguous xyz — use strided view (fastest path)
                    points = np.ndarray(
                        shape=(msg.height, msg.width, 3),
                        dtype=np.float32,
                        buffer=data,
                        strides=(msg.row_step, msg.point_step, 4),
                        offset=x_off
                    ).copy()  # copy to own memory so buffer can be freed
                else:
                    # Non-contiguous fields — extract each axis separately
                    x = np.ndarray(shape=(msg.height, msg.width), dtype=np.float32,
                                   buffer=data, strides=(msg.row_step, msg.point_step),
                                   offset=x_off).copy()
                    y = np.ndarray(shape=(msg.height, msg.width), dtype=np.float32,
                                   buffer=data, strides=(msg.row_step, msg.point_step),
                                   offset=y_off).copy()
                    z = np.ndarray(shape=(msg.height, msg.width), dtype=np.float32,
                                   buffer=data, strides=(msg.row_step, msg.point_step),
                                   offset=z_off).copy()
                    points = np.stack([x, y, z], axis=-1)
                
                self.latest_pointcloud = points
                self.latest_pc_stamp = msg.header.stamp
                self.latest_pc_frame = msg.header.frame_id
                self.get_logger().debug(
                    f'Received organized PC: {msg.height}×{msg.width} '
                    f'in frame {msg.header.frame_id}'
                )
            else:
                self.get_logger().warn('Received unorganized pointcloud!')
                self.latest_pointcloud = None
                
        except Exception as e:
            self.get_logger().error(f'Error in pointcloud callback: {e}')
    
    def update_pose_from_tf(self):
        """Get robot pose from TF (odom -> body_link) provided by odom_publisher.py"""
        try:
            if not self.tf_buffer.can_transform('odom', 'body_link', rclpy.time.Time()):
                return False
            
            transform = self.tf_buffer.lookup_transform(
                'odom', 'body_link', rclpy.time.Time()
            )
            
            # Extract position
            self.robot_pos = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])
            
            # Extract yaw from quaternion
            q = transform.transform.rotation
            sin_yaw = 2.0 * (q.w * q.z + q.x * q.y)
            cos_yaw = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self.robot_yaw = np.arctan2(sin_yaw, cos_yaw)
            
            return True
        except Exception:
            return False
    
    def image_callback(self, msg):
        try:
            # Update robot pose from TF
            self.update_pose_from_tf()  # Non-blocking, uses last known if fails
            
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Run YOLO detection
            results = self.model.predict(
                cv_image,
                conf=self.conf_threshold,
                verbose=False,
                device=0  # Use GPU
            )
            
            # Create segmentation mask
            seg_mask = np.zeros((cv_image.shape[0], cv_image.shape[1]), dtype=np.uint8)
            
            if results and results[0].masks is not None:
                for i, (mask, cls) in enumerate(zip(results[0].masks.data, results[0].boxes.cls)):
                    class_id = int(cls.item())
                    
                    # Resize mask to image size
                    mask_np = mask.cpu().numpy()
                    mask_resized = cv2.resize(
                        mask_np, 
                        (cv_image.shape[1], cv_image.shape[0]),
                        interpolation=cv2.INTER_LINEAR
                    )
                    
                    # Map COCO classes to our classes:
                    # 0 (person) -> 1 (human)
                    # Everything else -> 3 (object)
                    if class_id == 0:  # person
                        seg_mask[mask_resized > 0.5] = 1
                    else:
                        # Only set to 3 if not already human
                        seg_mask[(mask_resized > 0.5) & (seg_mask != 1)] = 3
            
            # Publish segmentation mask
            mask_msg = self.bridge.cv2_to_imgmsg(seg_mask, encoding='mono8')
            mask_msg.header = msg.header  # Preserve timestamp
            self.seg_mask_pub.publish(mask_msg)
            
            # Draw bounding boxes on the RGB image and publish annotated image
            annotated_image = cv_image.copy()
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                for i, box in enumerate(boxes):
                    # Get bounding box coordinates (xyxy format)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    class_id = int(box.cls.item())
                    confidence = float(box.conf.item())
                    
                    # Get class name from YOLO model
                    class_name = self.model.names[class_id] if hasattr(self.model, 'names') else f'class_{class_id}'
                    
                    # Color: red for person (human), blue for other objects
                    if class_id == 0:  # person
                        color = (0, 0, 255)  # Red (BGR)
                        label = f'HUMAN: {confidence:.2f}'
                    else:
                        color = (255, 0, 0)  # Blue (BGR)
                        label = f'{class_name}: {confidence:.2f}'
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label background
                    (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated_image, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
                    
                    # Draw label text
                    cv2.putText(annotated_image, label, (x1, y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Rate-limited publishing and display of annotated image
            current_time = self.get_clock().now()
            time_since_last = (current_time - self.last_logging_publish_time).nanoseconds / 1e9
            if time_since_last >= self.logging_publish_period:
                self.last_logging_publish_time = current_time
                
                # Publish annotated image
                annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
                annotated_msg.header = msg.header
                self.annotated_image_pub.publish(annotated_msg)
                
                # Display annotated RGB image with YOLO bounding boxes
                cv2.imshow('YOLO Bounding Boxes', annotated_image)
                cv2.waitKey(1)
            
            # Calculate and publish human centroid for gimbal tracking
            centroid_msg = Point()
            human_mask = (seg_mask == 1)
            
            if human_mask.any():
                # Calculate centroid of all detected humans
                y_coords, x_coords = np.where(human_mask)
                if len(x_coords) > 0:
                    centroid_x = float(np.mean(x_coords))
                    centroid_y = float(np.mean(y_coords))
                    
                    # Publish centroid (x, y in pixels, z = image width for reference)
                    centroid_msg.x = centroid_x
                    centroid_msg.y = centroid_y
                    centroid_msg.z = float(cv_image.shape[1])  # Image width
                    self.human_centroid_pub.publish(centroid_msg)
            else:
                # No human detected - send center position (straight forward)
                img_width = float(cv_image.shape[1])
                img_height = float(cv_image.shape[0])
                centroid_msg.x = img_width / 2.0  # Center X
                centroid_msg.y = img_height / 2.0  # Center Y
                centroid_msg.z = img_width  # Image width
                self.human_centroid_pub.publish(centroid_msg)
            
            # Project human detections to class_map grid using organized 3D pointcloud
            # Pipeline: Map detections to pointcloud in camera frame -> Transform to odom -> Create grid
            # This sparse map will be expanded by brushfire in C++
            class_map_grid = np.zeros((self.grid_imax, self.grid_jmax), dtype=np.int8)
            visibility_grid = np.zeros((self.grid_imax, self.grid_jmax), dtype=np.int8)  # Camera visibility
            
            # Check if we have synchronized organized pointcloud
            if (self.latest_pointcloud is not None and 
                len(self.latest_pointcloud.shape) == 3 and
                results and results[0].masks is not None):
                
                # Get organized pointcloud (H×W×3) - still in camera frame
                xyz = self.latest_pointcloud
                pc_h, pc_w = xyz.shape[:2]
                
                # Resize segmentation mask to match pointcloud dimensions if needed
                if seg_mask.shape != (pc_h, pc_w):
                    seg_mask_pc = cv2.resize(seg_mask, (pc_w, pc_h), 
                                             interpolation=cv2.INTER_NEAREST)
                else:
                    seg_mask_pc = seg_mask
                
                # Flatten both for vectorized processing
                xyz_flat = xyz.reshape(-1, 3)
                seg_mask_flat = seg_mask_pc.reshape(-1)
                
                # STEP 1: Get ALL valid points for visibility tracking
                valid_mask = np.isfinite(xyz_flat).all(axis=1)
                xyz_valid = xyz_flat[valid_mask]
                seg_mask_valid = seg_mask_flat[valid_mask]
                
                # For class_map, we only use labeled points; for visibility, we use all valid points
                
                # STEP 2: Transform ALL valid points from camera frame to odom frame
                try:
                    # Lookup transform from camera frame to body_link (robot-centered)
                    transform = self.tf_buffer.lookup_transform(
                        'body_link',
                        self.latest_pc_frame,
                        rclpy.time.Time(),
                        timeout=rclpy.duration.Duration(seconds=0.1)
                    )
                    
                    # Extract rotation and translation
                    trans = transform.transform.translation
                    rot = transform.transform.rotation
                    
                    # Convert quaternion to rotation matrix
                    qx, qy, qz, qw = rot.x, rot.y, rot.z, rot.w
                    R = np.array([
                        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
                        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
                        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
                    ])
                    
                    # Apply transformation: xyz_odom = R @ xyz_camera + t
                    xyz_odom = (R @ xyz_valid.T).T + np.array([trans.x, trans.y, trans.z])
                    
                except (TransformException, tf2_ros.LookupException, 
                        tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                    self.get_logger().warn(
                        f'TF lookup failed ({self.latest_pc_frame} -> body_link): {e}',
                        throttle_duration_sec=5.0
                    )
                    # Skip this frame if transform not available
                    xyz_odom = np.array([]).reshape(0, 3)
                    seg_mask_valid = np.array([])
                
                # Points are now in body_link frame (robot-centered)
                # X is forward, Y is left, Z is up
                if xyz_odom.shape[0] > 0:
                    xyz_body = xyz_odom  # Already in body frame
                    
                    # Coordinate system: Z is up, X is forward, Y is left (body frame)
                    # Filter out ground points using Z (height) thresholding
                    height_mask = ((xyz_body[:, 2] > self.floor_threshold) & 
                                  (xyz_body[:, 2] < self.ceiling_threshold))
                    xyz_body_filtered = xyz_body[height_mask]
                    seg_mask_filtered = seg_mask_valid[height_mask]
                    
                    # Filter by forward, lateral, and height bounds (in BODY frame)
                    if xyz_body_filtered.shape[0] > 0:
                        forward_mask = ((xyz_body_filtered[:, 0] > self.forward_min) & 
                                       (xyz_body_filtered[:, 0] < self.grid_size / 2.0))
                        lateral_mask = ((xyz_body_filtered[:, 1] > -self.grid_size / 2.0) & 
                                       (xyz_body_filtered[:, 1] < self.grid_size / 2.0))
                        height_mask2 = ((xyz_body_filtered[:, 2] > self.height_min) & 
                                       (xyz_body_filtered[:, 2] < self.height_max))
                        
                        combined_mask = forward_mask & lateral_mask & height_mask2
                        xyz_final = xyz_body_filtered[combined_mask]  # Use body coords for grid
                        seg_mask_final = seg_mask_filtered[combined_mask]
                        
                        # Convert 3D points to grid coordinates
                        # Match LiDAR convention (cloud_merger.h) in body_link frame:
                        # ic = pt.y / DS + IMAX/2  -> row from Y, add for positive
                        # jc = pt.x / DS + JMAX/2  -> col from X, add for positive
                        # Body frame: X is forward, Y is left
                        if xyz_final.shape[0] > 0:
                            grid_row = (self.grid_imax // 2 + 
                                       np.floor(xyz_final[:, 1] / self.grid_ds).astype(int))
                            grid_col = (self.grid_jmax // 2 + 
                                       np.floor(xyz_final[:, 0] / self.grid_ds).astype(int))
                            
                            # Filter points within grid bounds
                            in_bounds = ((grid_col >= 0) & (grid_col < self.grid_jmax) & 
                                        (grid_row >= 0) & (grid_row < self.grid_imax))
                            grid_col = grid_col[in_bounds]
                            grid_row = grid_row[in_bounds]
                            seg_mask_in_bounds = seg_mask_final[in_bounds]
                            
                            # Populate visibility_grid for ALL camera-visible cells
                            # and class_map for cells with YOLO detections
                            for i in range(len(grid_row)):
                                row = grid_row[i]
                                col = grid_col[i]
                                # Mark as visible (camera can see this cell)
                                visibility_grid[row, col] = 1
                                # Also populate class_map if labeled
                                class_idx = seg_mask_in_bounds[i]
                                if class_idx > 0:  # Human (1) or object (3)
                                    # Keep human class (1) as higher priority
                                    if class_map_grid[row, col] == 0 or class_idx == 1:
                                        class_map_grid[row, col] = class_idx
            
            # Publish class_map as OccupancyGrid
            class_map_msg = OccupancyGrid()
            class_map_msg.header = msg.header
            class_map_msg.header.frame_id = 'body_link'
            class_map_msg.info.resolution = self.grid_ds
            class_map_msg.info.width = self.grid_jmax
            class_map_msg.info.height = self.grid_imax
            class_map_msg.data = class_map_grid.flatten().tolist()
            self.class_map_pub.publish(class_map_msg)
            
            # Publish visibility_map as OccupancyGrid (1 = camera can see, 0 = not visible)
            visibility_map_msg = OccupancyGrid()
            visibility_map_msg.header = msg.header
            visibility_map_msg.header.frame_id = 'body_link'
            visibility_map_msg.info.resolution = self.grid_ds
            visibility_map_msg.info.width = self.grid_jmax
            visibility_map_msg.info.height = self.grid_imax
            visibility_map_msg.data = visibility_grid.flatten().tolist()
            self.visibility_map_pub.publish(visibility_map_msg)
            
            # Visualize class_map with OpenCV for debugging
            # Create color image: human=red, object=blue, empty=black
            class_map_vis = np.zeros((self.grid_imax, self.grid_jmax, 3), dtype=np.uint8)
            class_map_vis[class_map_grid == 1] = [0, 0, 255]  # Human = red (BGR)
            class_map_vis[class_map_grid == 3] = [255, 0, 0]  # Object = blue (BGR)
            
            # Mark robot position at center with green circle
            center = (self.grid_jmax // 2, self.grid_imax // 2)
            cv2.circle(class_map_vis, center, 3, (0, 255, 0), -1)
            
            # Upscale for visibility
            upscale = 6
            class_map_vis = cv2.resize(class_map_vis, None, fx=upscale, fy=upscale, 
                                        interpolation=cv2.INTER_NEAREST)
            
            # Flip vertically to match display convention
            class_map_vis = cv2.flip(class_map_vis, 0)
            
            cv2.imshow('YOLO Class Map', class_map_vis)
            cv2.waitKey(1)
            
            # Log detection statistics
            pc_status = "synced" if self.latest_pointcloud is not None else "waiting"
            self.get_logger().info(
                f'Detected: {int((seg_mask == 1).sum())} human pixels, '
                f'{int((seg_mask == 3).sum())} object pixels, '
                f'{int(np.sum(class_map_grid == 1))} grid cells, '
                f'PC: {pc_status}',
                throttle_duration_sec=2.0
            )
            
        except Exception as e:
            self.get_logger().error(f'Error in image callback: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = YOLODetectorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
