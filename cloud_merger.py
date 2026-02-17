#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid
import sensor_msgs_py.point_cloud2 as pc2
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_py as tf2
import numpy as np
from scipy.ndimage import convolve
import time
import math
import threading

# Grid Configuration (from poisson.h)
IMAX = 100
JMAX = 100
DS = 0.05  # Grid resolution in meters

# Point cloud filtering bounds
minX = 0.40  # Must be >= 0.370
maxX = (JMAX / 2) * DS
minY = 0.20  # Must be >= 0.185
maxY = (IMAX / 2) * DS
minZ = 0.05
maxZ = 0.80


class CloudMergerNode(Node):
    def __init__(self):
        super().__init__('cloud_merger')
        
        # Target frame for all point clouds (common reference frame)
        self.target_frame = self.declare_parameter('target_frame', 'livox_frame').value
        
        # TF2 setup for dynamic transforms
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Robot pose and orientation
        self.r = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # position [x, y, z]
        self.r_map = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # map origin
        self.rpy = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # roll, pitch, yaw
        
        # Time tracking
        self.last_time = time.time()
        self.dt = 1.0e10  # Large initial dt
        
        # Combined point cloud storage (as NumPy array for efficiency)
        self.combined_cloud = np.empty((0, 4), dtype=np.float32)
        
        # Confidence map (occupancy grid with temporal filtering)
        self.old_conf = np.zeros((IMAX, JMAX), dtype=np.int8)
        
        # Precompute polar coordinates for each grid cell
        self.polar_r2 = np.zeros((IMAX, JMAX), dtype=np.float32)
        self.polar_th = np.zeros((IMAX, JMAX), dtype=np.float32)
        
        for i in range(IMAX):
            for j in range(JMAX):
                x = (j - JMAX / 2) * DS
                y = (i - IMAX / 2) * DS
                self.polar_r2[i, j] = x * x + y * y
                self.polar_th[i, j] = math.atan2(y, x)
        
        # Create Gaussian kernel for convolution
        self.gauss_kernel = self.gaussian_kernel(9, 2.0)
        
        # Create subscribers and publishers
        self.livox_sub = self.create_subscription(
            PointCloud2,
            '/livox/lidar',
            self.lidar_callback,
            1
        )
        
        self.utlidar_sub = self.create_subscription(
            PointCloud2,
            '/zed/pointcloud',
            self.combined_callback,
            1
        )
        # Publishers
        self.cloud_pub = self.create_publisher(PointCloud2, 'merged_cloud', 10)
        self.map_pub = self.create_publisher(OccupancyGrid, 'occupancy_grid', 10)
        
        # Storage for latest data to publish (updated by callback, published by timer)
        self.latest_cloud_data = None
        self.latest_map_data = None
        self.data_lock = threading.Lock()
        
        # Create publication timer (20 Hz publishing rate)
        self.pub_timer = self.create_timer(0.05, self.publish_callback)  # 50ms = 20Hz
        
        self.get_logger().info(f'CloudMerger node initialized (target_frame: {self.target_frame})')
    
    def transform_pointcloud(self, points, source_frame, target_frame):
        """
        Transform point cloud from source_frame to target_frame using TF
        
        Args:
            points: numpy array of shape (N, 4) with [x, y, z, intensity]
            source_frame: source frame ID
            target_frame: target frame ID
            
        Returns:
            Transformed points as numpy array
        """
        if source_frame == target_frame:
            return points
        
        try:
            # Lookup transform with a timeout
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
            # Extract translation and rotation
            trans = transform.transform.translation
            rot = transform.transform.rotation
            
            # Convert quaternion to rotation matrix
            # Using tf2 convention: qx, qy, qz, qw
            qx, qy, qz, qw = rot.x, rot.y, rot.z, rot.w
            
            # Quaternion to rotation matrix
            R = np.array([
                [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
                [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
                [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
            ], dtype=np.float32)
            
            # Apply transform: p_target = R * p_source + t
            xyz_transformed = (R @ points[:, :3].T).T
            xyz_transformed[:, 0] += trans.x
            xyz_transformed[:, 1] += trans.y
            xyz_transformed[:, 2] += trans.z
            
            # Keep intensity unchanged
            transformed = np.column_stack([xyz_transformed, points[:, 3]])
            return transformed
            
        except TransformException as ex:
            self.get_logger().warn(
                f'Could not transform from {source_frame} to {target_frame}: {ex}'
            )
            return points  # Return untransformed if lookup fails
    
    def gaussian_kernel(self, kernel_size, sigma):
        """Create a Gaussian kernel for convolution"""
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        half = kernel_size // 2
        
        for i in range(-half, half + 1):
            for j in range(-half, half + 1):
                val = math.exp(-(i * i + j * j) / (2.0 * sigma * sigma))
                kernel[i + half, j + half] = val
        
        return kernel
    
    def ang_diff(self, a1, a2):
        """Compute difference between two angles wrapped between [-pi, pi] (VECTORIZED)"""
        a3 = a1 - a2
        # Vectorized angle wrapping using modulo
        a3 = (a3 + np.pi) % (2.0 * np.pi) - np.pi
        return a3
    
    def bilinear_interpolation_int8(self, grid, i, j):
        """Perform bilinear interpolation on a 2D int8 grid"""
        i1 = int(np.floor(i))
        j1 = int(np.floor(j))
        i2 = int(np.ceil(i))
        j2 = int(np.ceil(j))
        
        if i1 != i2 and j1 != j2:
            f1 = (i2 - i) * float(grid[i1, j1]) + (i - i1) * float(grid[i2, j1])
            f2 = (i2 - i) * float(grid[i1, j2]) + (i - i1) * float(grid[i2, j2])
            return int(round((j2 - j) * f1 + (j - j1) * f2))
        elif i1 != i2:
            return int(round((i2 - i) * float(grid[i1, j1]) + (i - i1) * float(grid[i2, j1])))
        elif j1 != j2:
            return int(round((j2 - j) * float(grid[i1, j1]) + (j - j1) * float(grid[i1, j2])))
        else:
            return grid[i1, j1]
    
    def lidar_callback(self, msg):
        """Process Livox LiDAR point cloud"""
        # Convert ROS PointCloud2 to numpy array
        points = []
        
        # Check if intensity field exists
        field_names = [field.name for field in msg.fields]
        has_intensity = 'intensity' in field_names
        
        if has_intensity:
            for p in pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True):
                points.append([p[0], p[1], p[2], p[3]])
        else:
            for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                points.append([p[0], p[1], p[2], 1.0])  # Default intensity = 1.0
        
        if len(points) == 0:
            return
        
        cloud = np.array(points, dtype=np.float32)
        
        # Transform to target frame using TF
        cloud_transformed = self.transform_pointcloud(cloud, msg.header.frame_id, self.target_frame)
        
        # Add to combined cloud (keep as NumPy array)
        if len(self.combined_cloud) == 0:
            self.combined_cloud = cloud_transformed
        else:
            self.combined_cloud = np.vstack([self.combined_cloud, cloud_transformed])
    
    def combined_callback(self, msg):
        """Process combined point cloud and generate occupancy grid"""
        start_time = time.time()
        
        # Convert ROS message to numpy array (ZED only has x, y, z)
        utlidar_points = []
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            utlidar_points.append([p[0], p[1], p[2], 1.0])  # Add full intensity
        
        if len(utlidar_points) > 0:
            utlidar_cloud = np.array(utlidar_points, dtype=np.float32)
            # Transform ZED points to target frame
            utlidar_cloud = self.transform_pointcloud(utlidar_cloud, msg.header.frame_id, self.target_frame)
        else:
            utlidar_cloud = np.empty((0, 4), dtype=np.float32)
        
        # Combine with stored Livox cloud (keep as NumPy arrays)
        if len(self.combined_cloud) > 0:
            odom_cloud = np.vstack([utlidar_cloud, self.combined_cloud])
        else:
            odom_cloud = utlidar_cloud
        
        self.combined_cloud = np.empty((0, 4), dtype=np.float32)  # Clear combined cloud
        
        if len(odom_cloud) == 0:
            return
        
        # Create raw occupancy map (VECTORIZED)
        raw_map = np.zeros((IMAX, JMAX), dtype=np.float32)
        
        # Vectorized height filtering
        z_valid = (odom_cloud[:, 2] > minZ) & (odom_cloud[:, 2] < maxZ)
        valid_cloud = odom_cloud[z_valid]
        
        if len(valid_cloud) > 0:
            # Vectorized grid coordinate conversion
            ic = (valid_cloud[:, 1] - self.r[1]) / DS + IMAX / 2
            jc = (valid_cloud[:, 0] - self.r[0]) / DS + JMAX / 2
            
            # Vectorized bounds checking
            in_bounds = (ic > 0) & (ic < IMAX - 1) & (jc > 0) & (jc < JMAX - 1)
            ic_valid = np.round(ic[in_bounds]).astype(int)
            jc_valid = np.round(jc[in_bounds]).astype(int)
            
            # Mark cells as occupied (multiple points can map to same cell)
            raw_map[ic_valid, jc_valid] = 1.0
        
        # Apply filtered occupancy convolution
        confidence_values = self.filtered_occupancy_convolution(raw_map, self.old_conf)
        self.old_conf = confidence_values.copy()
        
        # Prepare data for publishing (done asynchronously by timer)
        header = msg.header
        header.frame_id = self.target_frame
        header.stamp = self.get_clock().now().to_msg()
        
        # Store latest data with thread safety
        with self.data_lock:
            self.latest_cloud_data = (header, odom_cloud[:, :3].copy())
            self.latest_map_data = (header, confidence_values.copy())
        
        elapsed = (time.time() - start_time) * 1000
        self.get_logger().info(f'Occ Map Solve Time: {elapsed:.2f} ms')
    
    def publish_callback(self):
        """Timer callback to publish latest cloud and occupancy grid data"""
        with self.data_lock:
            if self.latest_cloud_data is None or self.latest_map_data is None:
                return
            
            # Get local copies to release lock quickly
            cloud_header, cloud_points = self.latest_cloud_data
            map_header, confidence_values = self.latest_map_data
        
        # Publish merged point cloud
        cloud_msg = pc2.create_cloud_xyz32(cloud_header, cloud_points)
        self.cloud_pub.publish(cloud_msg)
        
        # Publish occupancy grid
        map_msg = OccupancyGrid()
        map_msg.header = map_header
        map_msg.info.width = JMAX
        map_msg.info.height = IMAX
        map_msg.info.resolution = float(DS)
        map_msg.info.origin.position.x = float(-maxX + self.r[0])
        map_msg.info.origin.position.y = float(-maxY + self.r[1])
        map_msg.info.origin.position.z = 0.0
        map_msg.info.origin.orientation.w = 1.0
        map_msg.info.origin.orientation.x = 0.0
        map_msg.info.origin.orientation.y = 0.0
        map_msg.info.origin.orientation.z = 0.0
        
        # Flatten confidence values
        map_msg.data = confidence_values.flatten().tolist()
        
        self.map_pub.publish(map_msg)
    
    def filtered_occupancy_convolution(self, occupancy_data, old_conf_map):
        """Apply temporal filtering and Gaussian convolution to occupancy data (VECTORIZED)"""
        
        # Shift confidence values based on egomotion (VECTORIZED)
        drx = self.r[0] - self.r_map[0]
        dry = self.r[1] - self.r_map[1]
        self.r_map[0] = self.r[0]
        self.r_map[1] = self.r[1]
        
        # Vectorized egomotion compensation using scipy.ndimage shift
        from scipy.ndimage import shift
        shift_pixels = [dry / DS, drx / DS]
        confidence_values = shift(old_conf_map.astype(np.float32), shift_pixels, order=1, mode='constant', cval=0) / 127.0
        
        # Apply Gaussian convolution to binary occupancy data
        buffered_binary = convolve(occupancy_data, self.gauss_kernel, mode='constant', cval=0.0)
        
        # Vectorized temporal filtering
        thresh = 4.0
        
        # Determine if cells are in front of robot (VECTORIZED)
        range_flag = self.polar_r2 > 1.44
        angle_flag = np.abs(self.ang_diff(self.rpy[2], self.polar_th)) > 0.6
        front_flag = ~(range_flag | angle_flag)
        
        # Occupied vs free cell mask
        occupied_mask = buffered_binary > thresh
        
        # Compute beta_up and beta_dn for all cells at once
        beta_up = np.where(front_flag, 4.0, 1.0)  # Front: 4.0, Back: 1.0
        beta_dn = 4.0  # Same for all regions
        
        # Compute sig and C for occupied cells
        sig_occupied = 1.0 - np.exp(-beta_up * buffered_binary * self.dt)
        C_occupied = 1.0
        
        # Compute sig and C for free cells
        sig_free = 1.0 - np.exp(-beta_dn * self.dt)
        C_free = 0.0
        
        # Select appropriate values based on occupied_mask
        sig = np.where(occupied_mask, sig_occupied, sig_free)
        C = np.where(occupied_mask, C_occupied, C_free)
        
        # Update confidence with temporal filter (vectorized)
        confidence_values *= (1.0 - sig)
        confidence_values += sig * C
        
        # Convert back to int8
        confidence_values = np.clip(np.round(127.0 * confidence_values), 0, 127).astype(np.int8)
        
        return confidence_values


def main(args=None):
    rclpy.init(args=args)
    node = CloudMergerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
