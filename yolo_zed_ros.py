"""Command line script to test inference on one or more images."""

import argparse
import os.path as path
import sys

import numpy as np
import torch
import cv2
from PIL import Image
from ultralytics import YOLO
import scipy.ndimage
import matplotlib.pyplot as plt
import struct
import threading
from queue import Queue

# ROS2 imports (optional - will disable ROS2 publishing if not available)
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2, PointField
    from nav_msgs.msg import OccupancyGrid
    from std_msgs.msg import Header
    from geometry_msgs.msg import Point
    ROS2_AVAILABLE = True
    print("[INFO] ROS2 available - PointCloud2 publishing enabled")
except (ModuleNotFoundError, ImportError) as e:
    print(f"[WARNING] ROS2 not available: {e}")
    print("[WARNING] PointCloud2 publishing will be disabled")
    ROS2_AVAILABLE = False
    Node = object  # Dummy class for compatibility

# COCO dataset labels (subset relevant to navigation)
labels = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
    9: 'traffic light',
    11: 'stop sign',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    24: 'backpack',
    26: 'handbag',
    # Add more COCO classes as needed
}

# Import ZED SDK
import pyzed.sl as sl

# --- Poisson solver import ---
try:
    # Ensure local build directory precedes site-packages so newest .so is loaded
    build_path = path.abspath(path.join(path.dirname(__file__), 'PoissonZO', 'build'))
    print(build_path)
    if build_path not in sys.path:
        sys.path.insert(0, build_path)
    import importlib
    import poisson_solver
except ImportError as e:
    print(f"Error importing poisson_solver: {e}")
    print("Please ensure that 'poisson_solver.cpython-*.so' is in your Python path.")
    poisson_solver = None

# ============================================================================
# CONFIGURATION - All tuning parameters in one place
# ============================================================================

# --- YOLO Detection ---
YOLO_MODEL_PATH = 'yolo11n-seg.pt'
YOLO_ALLOWED_CLASS_IDS = [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck

# --- Occupancy Grid ---
GRID_SIZE = 5.0  # meters (width and height of grid)
GRID_RESOLUTION = 0.05  # meters per cell

# --- Occupancy Hysteresis Thresholds ---
OCCUPANCY_THRESHOLD_HIGH = 85  # Strong occupancy threshold (0-127)
OCCUPANCY_THRESHOLD_LOW = 64   # Weak occupancy threshold (0-127)

# --- Point Cloud Filtering ---
FLOOR_THRESHOLD = -1.0      # meters (Z-axis, reject points below this)
CEILING_THRESHOLD = 0.5     # meters (Z-axis, reject points above this)
HEIGHT_MIN = 0.1            # meters (minimum object height)
HEIGHT_MAX = 2.0            # meters (maximum object height)
FORWARD_MIN = 0.1           # meters (minimum forward distance)

# --- Brushfire Algorithm ---
BRUSHFIRE_CONNECTIVITY = 8  # 4 or 8 (8 includes diagonals)
BRUSHFIRE_PERSISTENCE_FRAMES = 10  # Number of frames to keep human detections
BRUSHFIRE_DECAY_RATE = 0.8  # Decay factor per frame (0.0 = instant decay, 1.0 = no decay)

# --- Class Map Dilation ---
DILATION_STRUCTURE_SIZE = 7  # Size of dilation kernel (7x7)

# --- Safety Field (Poisson Solver) ---
DH0_HUMAN = 6.0      # Boundary strength for humans (class 1)
DH0_OBJECT = 3.0     # Boundary strength for other objects (class 3)
DH0_FREE = 1.0       # Baseline boundary strength for free space
GUIDANCE_SMOOTH_SIZE = 3  # Median filter size for guidance field smoothing

# --- ZED Camera ---
ZED_RESOLUTION = sl.RESOLUTION.HD720
ZED_FPS = 30
ZED_DEPTH_MODE = sl.DEPTH_MODE.NEURAL
ZED_MIN_DISTANCE = 0.1  # meters
ZED_COORDINATE_SYSTEM = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD

# --- ROS2 Publishing ---
ROS2_DOWNSAMPLE_FACTOR = 20  # Skip every N points for pointcloud
ROS2_QUEUE_SIZE = 10         # Queue size for publishers/subscribers
ROS2_POINTCLOUD_QUEUE_SIZE = 2  # Max frames in pointcloud queue

# --- Visualization ---
CV2_WINDOW_SCALE = 0.4       # Scale factor for OpenCV window (1.0 = original size, 0.5 = half, 2.0 = double)
DISPLAY_SCALE = 1.2           # Scale factor for individual images before combining
QUIVER_STEP = 8               # Grid spacing for guidance arrows
ARROW_SCALE = 1.2             # Arrow length scale factor
ARROW_MIN_LENGTH = 3          # Minimum arrow length in pixels
FIGURE_SIZE = (15, 10)        # Matplotlib figure size
OVERLAY_ALPHA = 0.5           # Alpha blend for h-field overlay

# ============================================================================
# END CONFIGURATION
# ============================================================================


def run_poisson_solver(occupancy_grid, class_map=None, dh0_classes=None, ds=None):
    """
    occupancy_grid: 2D numpy array, 0 for free, 1 for occupied (grid_dim x grid_dim)
    class_map: 2D numpy array of class indices (same shape as occupancy_grid)
    dh0_classes: list or array of dh0 values, one per class index
    Returns: h_field (2D numpy array, grid_dim x grid_dim), guidance_x, guidance_y
    """
    if poisson_solver is None:
        raise RuntimeError("poisson_solver module not loaded.")
    IMAX_solver = occupancy_grid.shape[0]
    JMAX_solver = occupancy_grid.shape[1]
    # Align solver spatial step with the grid resolution if provided
    DS_solver = float(ds) if ds is not None else 0.1
    poisson_solver.set_IMAX(IMAX_solver)
    poisson_solver.set_JMAX(JMAX_solver)
    poisson_solver.set_DS(DS_solver)
    if dh0_classes is not None:
        poisson_solver.set_dh0_classes([float(x) for x in dh0_classes])
    # Use +1 free, -1 occupied format for solver
    occ_grid_solver_fmt_1d = np.ones(IMAX_solver * JMAX_solver, dtype=np.float32)
    flat_occupancy_grid = occupancy_grid.flatten()
    occupied_indices = flat_occupancy_grid == 1
    occ_grid_solver_fmt_1d[occupied_indices] = -1.0
    sample_yaw = 0.0
    class_map_1d = None
    if class_map is not None:
        class_map_1d = np.asarray(class_map, dtype=np.int32).flatten()
    
    # Run class map-based solver
    h_grid_np, guidance_x_np, guidance_y_np, _ = poisson_solver.solve_poisson_safety_function(
        occ_grid_solver_fmt_1d, sample_yaw, class_map_1d)
    
    h_field = h_grid_np.reshape((IMAX_solver, JMAX_solver))
    guidance_x = guidance_x_np.reshape((IMAX_solver, JMAX_solver))
    guidance_y = guidance_y_np.reshape((IMAX_solver, JMAX_solver))
    return h_field, guidance_x, guidance_y

class ZedPointCloudPublisher(Node):
    """ROS2 Node for publishing ZED pointcloud and subscribing to occupancy grid."""
    
    def __init__(self, downsample_factor=20):
        """
        Initialize publisher.
        
        Args:
            downsample_factor: Skip every N points (default 20 = ~45k points from ~900k)
                              Higher = fewer points, faster. MID360 ~= 10-50k points
        """
        if not ROS2_AVAILABLE:
            self.enabled = False
            return
        
        super().__init__('yolo_zed_pointcloud_publisher')
        self.enabled = True
        self.downsample_factor = downsample_factor
        
        # Create publisher for PointCloud2
        self.pointcloud_pub = self.create_publisher(
            PointCloud2,
            '/zed/pointcloud',
            10
        )
        
        # Subscribe to occupancy grid from cloud_merger
        self.occupancy_sub = self.create_subscription(
            OccupancyGrid,
            'occupancy_grid',
            self.occupancy_callback,
            10
        )
        
        # Create publisher for human centroid (for gimbal tracking)
        self.human_centroid_pub = self.create_publisher(
            Point,
            '/human_tracking/centroid',
            10
        )
        
        # Storage for latest occupancy grid (confidence values 0-127)
        self.latest_occupancy_conf = None
        self.occupancy_grid_shape = (100, 100)  # IMAX x JMAX from cloud_merger
        self.occupancy_old = None  # Previous binary occupancy for hysteresis
        
        # Brushfire temporal persistence - store history of human class maps
        self.human_class_map_history = []  # List of (class_map, strength) tuples
        self.brushfire_persistence_frames = BRUSHFIRE_PERSISTENCE_FRAMES
        self.brushfire_decay_rate = BRUSHFIRE_DECAY_RATE
        
        # Debug visualization for raw occupancy grid
        self.debug_viz = False  # Disabled - matplotlib not thread-safe with ROS callbacks
        self.frame_count = 0
        
        # Create queue for async publishing
        self.pointcloud_queue = Queue(maxsize=2)  # Keep only latest 2 frames
        self.publishing_thread = None
        self.stop_thread = False
        
        # Start background publishing thread
        self.publishing_thread = threading.Thread(target=self._publish_worker, daemon=True)
        self.publishing_thread.start()
        
        self.get_logger().info(f'ZED PointCloud Publisher initialized (downsample={downsample_factor}x, async)')
    
    def occupancy_callback(self, msg):
        """Receive occupancy grid from cloud_merger"""
        if not self.enabled:
            return
        
        self.frame_count += 1
        
        # Convert OccupancyGrid data (int8 list, values 0-127) to numpy array
        # cloud_merger: grid[i, j] where i=Y index, j=X index
        # We want display: grid[row, col] where row 0 = X+ (up/forward), col 0 = Y+ (left)
        # Steps: 1) Transpose to swap axes, 2) Flip both axes
        conf_data = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
        conf_data_transposed = conf_data.T  # Now [X, Y]
        self.latest_occupancy_conf = np.flipud(np.fliplr(conf_data_transposed))  # Flip both for correct orientation
        
        # Debug: Plot raw received data immediately
        if self.debug_viz and self.frame_count % 5 == 0:  # Update every 5 frames to avoid overhead
            if self.debug_im is None:
                self.debug_im = self.debug_ax.imshow(self.latest_occupancy_conf, cmap='gray', vmin=0, vmax=127, origin='upper')
                self.debug_ax.set_title('Raw Occupancy Grid (from cloud_merger)')
                self.debug_ax.axis('off')
                plt.colorbar(self.debug_im, ax=self.debug_ax, fraction=0.046)
            else:
                self.debug_im.set_data(self.latest_occupancy_conf)
            self.debug_fig.canvas.draw_idle()
            self.debug_fig.canvas.flush_events()
            self.get_logger().info(f'Occupancy grid callback #{self.frame_count}: shape={self.latest_occupancy_conf.shape}, min={self.latest_occupancy_conf.min()}, max={self.latest_occupancy_conf.max()}')
    
    def build_occ_map(self, conf):
        """
        Threshold occupancy map with hysteresis.
        Based on C++ logic from cloudmerger.cpp.
        
        Args:
            conf: Confidence map (IMAX x JMAX) with values 0-127
            
        Returns:
            Binary occupancy map (IMAX x JMAX) with values 0 (free) or 1 (occupied)
        """
        T_hi = OCCUPANCY_THRESHOLD_HIGH
        T_lo = OCCUPANCY_THRESHOLD_LOW
        
        IMAX, JMAX = conf.shape
        occ_map = np.ones((IMAX, JMAX), dtype=np.float32)  # Default to free (1.0)
        
        # Initialize old occupancy if needed
        if self.occupancy_old is None:
            self.occupancy_old = occ_map.copy()
        
        # Note: dx shift is not implemented here since we're not tracking robot motion
        # If needed, this would shift the old map based on robot displacement
        dx = np.array([0.0, 0.0])  # No shift for now
        
        for i in range(IMAX):
            for j in range(JMAX):
                # Calculate shifted indices for temporal consistency
                # (Not used if dx = [0, 0])
                i0 = int(round(i + dx[1] / GRID_RESOLUTION))
                j0 = int(round(j + dx[0] / GRID_RESOLUTION))
                in_grid = (0 <= i0 < IMAX) and (0 <= j0 < JMAX)
                
                strong = conf[i, j] >= T_hi
                weak = conf[i, j] >= T_lo
                
                if strong:
                    occ_map[i, j] = 0.0  # Occupied (use 0 instead of -1 for binary)
                elif weak and in_grid:
                    occ_map[i, j] = self.occupancy_old[i0, j0]  # Keep previous state
                else:
                    occ_map[i, j] = 1.0  # Free
        
        # Update old occupancy for next iteration
        self.occupancy_old = occ_map.copy()
        
        # Convert to binary 0-1 format (invert: 0=free, 1=occupied)
        binary_occ = (occ_map < 0.5).astype(np.int32)
        return binary_occ
    
    def expand_human_labels_to_clusters(self, occupancy_grid, class_map):
        """
        Expand human labels to cover entire connected clusters using brushfire.
        Uses efficient connected component labeling algorithm with temporal persistence.
        
        Human detections persist across frames with exponential decay, creating a
        "burning" effect where human labels gradually fade over time.
        
        Args:
            occupancy_grid: Binary grid (0=free, 1=occupied)
            class_map: Sparse class labels (0=free, 1=human, 3=object)
            
        Returns:
            Updated class_map with human labels expanded to entire clusters
        """
        from scipy.ndimage import label
        
        # Determine connectivity structure based on configuration
        if BRUSHFIRE_CONNECTIVITY == 8:
            structure = np.ones((3, 3), dtype=int)  # 8-connectivity (includes diagonals)
        else:
            structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)  # 4-connectivity
        
        # Find connected components in occupancy grid
        labeled_clusters, num_clusters = label(occupancy_grid, structure=structure)
        
        # For each cluster, check if it contains any human points in CURRENT frame
        cluster_is_human_now = np.zeros(num_clusters + 1, dtype=bool)
        
        for cluster_id in range(1, num_clusters + 1):
            cluster_mask = (labeled_clusters == cluster_id)
            # If ANY cell in this cluster is labeled as human, mark the whole cluster
            if np.any(class_map[cluster_mask] == 1):
                cluster_is_human_now[cluster_id] = True
        
        # Create a strength map that combines current and historical human detections
        # Higher strength = more recent/confident human detection
        human_strength_map = np.zeros_like(occupancy_grid, dtype=np.float32)
        
        # Add current frame detections at full strength (1.0)
        for cluster_id in range(1, num_clusters + 1):
            if cluster_is_human_now[cluster_id]:
                cluster_mask = (labeled_clusters == cluster_id)
                human_strength_map[cluster_mask] = 1.0
        
        # Decay and merge historical detections
        decayed_history = []
        for hist_map, hist_strength in self.human_class_map_history:
            # Apply exponential decay
            new_strength = hist_strength * self.brushfire_decay_rate
            
            # Only keep if strength is still significant (threshold at 0.01)
            if new_strength > 0.01:
                # Merge with current strength map (take maximum)
                # This ensures that re-detections refresh the strength
                human_strength_map = np.maximum(human_strength_map, hist_map * new_strength)
                decayed_history.append((hist_map, new_strength))
        
        # Update history: add current binary human map at strength 1.0
        current_human_map = (human_strength_map > 0).astype(np.float32)
        decayed_history.insert(0, (current_human_map, 1.0))
        
        # Keep only the most recent N frames
        self.human_class_map_history = decayed_history[:self.brushfire_persistence_frames]
        
        # Count stats for debugging
        num_human_clusters_now = np.sum(cluster_is_human_now[1:])  # Exclude background (0)
        human_cells_before = np.sum(class_map == 1)
        human_cells_after = np.sum(human_strength_map > 0)
        human_cells_strong = np.sum(human_strength_map > 0.5)  # Recently detected
        
        # Create expanded class map based on strength threshold
        class_map_expanded = class_map.copy()
        
        # Apply human labels where strength is above threshold (0.3 = ~3 frames with decay 0.8)
        human_threshold = 0.3
        for cluster_id in range(1, num_clusters + 1):
            cluster_mask = (labeled_clusters == cluster_id)
            
            # Check if this cluster has significant human strength
            cluster_human_strength = np.max(human_strength_map[cluster_mask])
            
            if cluster_human_strength > human_threshold:
                # Label entire cluster as human (class 1)
                class_map_expanded[cluster_mask] = 1
            elif not cluster_is_human_now[cluster_id]:
                # Keep as generic obstacle (class 3) for occupied cells without human detection
                # Only set to 3 if not already labeled as something else
                class_map_expanded[cluster_mask] = np.where(
                    class_map_expanded[cluster_mask] == 0, 
                    3, 
                    class_map_expanded[cluster_mask]
                )
        
        # Log statistics (only when human detection changes occur or periodically)
        if human_cells_before > 0 or human_cells_after > 0:
            history_len = len(self.human_class_map_history)
            self.get_logger().info(
                f"[BRUSHFIRE] Clusters: {num_clusters}, Current humans: {num_human_clusters_now}, "
                f"Human cells: now={human_cells_before}, persist={human_cells_after} (strong={human_cells_strong}), "
                f"history={history_len}/{self.brushfire_persistence_frames}"
            )
        
        return class_map_expanded
    
    def get_thresholded_occupancy(self):
        """
        Get the latest occupancy grid with hysteresis thresholding applied.
        
        Returns:
            Binary occupancy grid (0=free, 1=occupied) or None if no data received yet
        """
        if self.latest_occupancy_conf is None:
            return None
        
        return self.build_occ_map(self.latest_occupancy_conf)
    
    def publish_pointcloud(self, pointcloud_zed):
        """Queue pointcloud for async publishing in background thread."""
        if not self.enabled:
            return
        
        # Non-blocking: if queue is full, drop the oldest frame
        if self.pointcloud_queue.full():
            try:
                self.pointcloud_queue.get_nowait()  # Remove oldest
            except:
                pass
        
        # Add downsampled pointcloud to queue - only copy XYZ data
        try:
            pc_full = pointcloud_zed.get_data()
            # Downsample by skipping pixels (much faster than voxel grid)
            # [::N, ::N] samples every Nth row and column
            pc_xyz_downsampled = pc_full[::self.downsample_factor, ::self.downsample_factor, :3].copy()
            self.pointcloud_queue.put_nowait(pc_xyz_downsampled)
        except:
            pass  # Queue full, skip this frame
    
    def _publish_worker(self):
        """Background worker thread that processes and publishes pointclouds."""
        while not self.stop_thread:
            try:
                # Get pointcloud from queue (blocking with timeout)
                xyz = self.pointcloud_queue.get(timeout=0.1)  # Already XYZ-only from queue
            except:
                continue  # Timeout, check stop_thread flag
            
            try:
                # Process pointcloud - XYZ already extracted, just flatten and filter
                height, width = xyz.shape[:2]
                
                # Flatten
                xyz_flat = xyz.reshape(-1, 3)
                
                # Filter out invalid points (NaN or Inf)
                valid_mask = np.isfinite(xyz_flat).all(axis=1)
                xyz_valid = xyz_flat[valid_mask]
                
                # Create PointCloud2 message
                header = Header()
                header.stamp = self.get_clock().now().to_msg()
                header.frame_id = 'zed_frame'
                
                # Define PointCloud2 fields (XYZ only)
                fields = [
                    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                ]
                
                # Create structured array with XYZ data
                num_points = xyz_valid.shape[0]
                cloud_data = np.zeros(num_points, dtype=[
                    ('x', np.float32),
                    ('y', np.float32),
                    ('z', np.float32)
                ])
                
                cloud_data['x'] = xyz_valid[:, 0]
                cloud_data['y'] = xyz_valid[:, 1]
                cloud_data['z'] = xyz_valid[:, 2]
                
                # Create PointCloud2 message
                msg = PointCloud2()
                msg.header = header
                msg.height = 1  # Unorganized point cloud
                msg.width = num_points
                msg.fields = fields
                msg.is_bigendian = False
                msg.point_step = 12  # 4 bytes * 3 fields (x, y, z)
                msg.row_step = msg.point_step * num_points
                msg.is_dense = True  # All points are valid (we filtered out invalid ones)
                msg.data = cloud_data.tobytes()
                
                # Publish
                self.pointcloud_pub.publish(msg)
                
            except Exception as e:
                # Silently handle errors (likely shutdown-related)
                if not self.stop_thread:
                    self.get_logger().error(f"Worker error: {e}")
    
    def shutdown(self):
        """Stop the publishing thread and cleanup."""
        if self.enabled:
            self.stop_thread = True
            if self.publishing_thread is not None:
                self.publishing_thread.join(timeout=1.0)

def main():
    # Initialize ROS2
    rclpy.init()
    
    # Create ROS2 publisher node
    ros_node = ZedPointCloudPublisher(downsample_factor=ROS2_DOWNSAMPLE_FACTOR)
    
    # Run ROS2 spinning in a separate thread so callbacks process independently
    import threading
    ros_thread = threading.Thread(target=lambda: rclpy.spin(ros_node), daemon=True)
    ros_thread.start()
    
    print('===> Initializing ZED camera')
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = ZED_RESOLUTION
    init_params.camera_fps = ZED_FPS
    init_params.depth_mode = ZED_DEPTH_MODE
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = ZED_COORDINATE_SYSTEM
    init_params.depth_minimum_distance = ZED_MIN_DISTANCE
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera")
        rclpy.shutdown()
        exit(1)

    print('==> Loading YOLO11 segmentation model')
    model = YOLO(YOLO_MODEL_PATH)
    
    # Explicitly use GPU for inference with half-precision for speed
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f'==> YOLO using device: {device}')
    if device == 'cuda':
        print(f'==> GPU: {torch.cuda.get_device_name(0)}')

    runtime_parameters = sl.RuntimeParameters()
    image_zed = sl.Mat()
    pointcloud_zed = sl.Mat()

    allowed_ids = YOLO_ALLOWED_CLASS_IDS
    
    # Create matplotlib figure once for reuse
    plt.ion()  # Interactive mode
    fig, axes = plt.subplots(2, 3, figsize=FIGURE_SIZE)
    fig.suptitle('Real-time Class Map Safety Field Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Initialize plot objects (will be updated each frame)
    im_objs = [None] * 5
    quiver_obj = None
    quiver_cbar = None  # Track colorbar for quiver plot to avoid duplication
    
    print('==> Press ESC in the window to exit')
    while True:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            zed.retrieve_measure(pointcloud_zed, sl.MEASURE.XYZRGBA)
            
            # Publish pointcloud to ROS2 (if available)
            if ROS2_AVAILABLE and ros_node is not None:
                ros_node.publish_pointcloud(pointcloud_zed)
            
            img_rgba = image_zed.get_data()
            img_rgb = img_rgba[:, :, :3]  # Drop alpha channel
            
            # Run YOLO11 segmentation with explicit GPU device and half precision
            results = model(img_rgb, verbose=False, device=device, half=(device=='cuda'))[0]
            
            # Create visualization with bounding boxes and masks
            cv2_img = results.plot()
            
            # Extract segmentation masks
            if results.masks is not None:
                # Get the masks data (H x W) for each detection
                masks = results.masks.data.cpu().numpy()  # Shape: (N, H, W)
                classes = results.boxes.cls.cpu().numpy().astype(int)  # Class IDs
                confidences = results.boxes.conf.cpu().numpy().reshape(-1)  # Confidence per detection
                
                # Create a combined segmentation mask with class mapping
                # Map COCO class 0 (person/human) to our class index 6
                # Map other allowed objects to class index 3
                seg_mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.int32)
                person_mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.bool_)

                for i, (mask, cls, conf) in enumerate(zip(masks, classes, confidences)):
                    # Resize mask to original image size if needed
                    if mask.shape != seg_mask.shape:
                        mask = cv2.resize(mask, (seg_mask.shape[1], seg_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                    mask_bool = (mask > 0.5)
                    
                    # Only process allowed classes
                    if cls in allowed_ids:
                        # Map human (COCO class 0) to our internal class 1 (to avoid conflict with zeros)
                        # Map other allowed objects to class index 3
                        # This way dh0_classes[1] = 6.0 controls human safety
                        mapped_class = 1 if cls == 0 else 3
                        seg_mask[mask_bool] = mapped_class
                        
                        # Track person detections separately for visualization
                        if cls == 0:
                            person_mask[mask_bool] = True
            else:
                seg_mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.int32)
                person_mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.bool_)
            
            # Calculate and publish human centroid for gimbal tracking
            if ROS2_AVAILABLE and ros_node is not None:
                centroid_msg = Point()
                
                if person_mask.any():
                    # Calculate centroid of all detected humans
                    y_coords, x_coords = np.where(person_mask)
                    if len(x_coords) > 0:
                        centroid_x = float(np.mean(x_coords))
                        centroid_y = float(np.mean(y_coords))
                        
                        # Publish centroid (x, y in pixels, z = image width for reference)
                        centroid_msg.x = centroid_x
                        centroid_msg.y = centroid_y
                        centroid_msg.z = float(img_rgb.shape[1])  # Image width
                        ros_node.human_centroid_pub.publish(centroid_msg)
                else:
                    # No human detected - send center position (straight forward)
                    img_width = float(img_rgb.shape[1])
                    img_height = float(img_rgb.shape[0])
                    centroid_msg.x = img_width / 2.0  # Center X
                    centroid_msg.y = img_height / 2.0  # Center Y
                    centroid_msg.z = img_width  # Image width
                    ros_node.human_centroid_pub.publish(centroid_msg)
            
            # Get occupancy grid from cloud_merger (merged Livox + ZED with temporal filtering)
            # This replaces the custom point cloud processing
            grid = ros_node.get_thresholded_occupancy()
            
            # If no occupancy data yet, skip this frame
            if grid is None:
                print("[WARNING] No occupancy grid data from cloud_merger yet, skipping frame")
                continue
            
            # Get grid dimensions from received occupancy grid
            grid_dim = grid.shape[0]  # Should be 100x100 from cloud_merger
            robot_x, robot_y = grid_dim // 2, grid_dim // 2
            
            # Create class_map from YOLO segmentation
            # We still need to project YOLO detections onto the grid for class labeling
            class_map = np.zeros((grid_dim, grid_dim), dtype=np.int32)
            
            # Get point cloud data for class mapping only
            pc_np = pointcloud_zed.get_data()
            xyz = pc_np[:, :, :3]
            
            # Flatten both xyz and seg_mask to match points with their segmentation class
            xyz_flat = xyz.reshape(-1, 3)
            seg_mask_flat = seg_mask.reshape(-1)
            
            # Filter only valid points (finite values)
            valid_mask = np.isfinite(xyz_flat).all(axis=1)
            xyz_valid = xyz_flat[valid_mask]
            seg_mask_valid = seg_mask_flat[valid_mask]
            
            # Coordinate system: Z is up, X is forward, Y is left
            # Filter out ground points using Z (height) thresholding
            height_mask = (xyz_valid[:, 2] > FLOOR_THRESHOLD) & (xyz_valid[:, 2] < CEILING_THRESHOLD)
            xyz_no_ground = xyz_valid[height_mask]
            seg_mask_no_ground = seg_mask_valid[height_mask]
            
            # Filter by forward, lateral, and height to match cloud_merger bounds
            forward_mask = (xyz_no_ground[:, 0] > FORWARD_MIN) & (xyz_no_ground[:, 0] < GRID_SIZE / 2.0)
            lateral_mask = (xyz_no_ground[:, 1] > -GRID_SIZE / 2.0) & (xyz_no_ground[:, 1] < GRID_SIZE / 2.0)
            height_mask = (xyz_no_ground[:, 2] > HEIGHT_MIN) & (xyz_no_ground[:, 2] < HEIGHT_MAX)
            
            combined_mask = forward_mask & lateral_mask & height_mask
            xyz_filtered = xyz_no_ground[combined_mask]
            seg_mask_filtered = seg_mask_no_ground[combined_mask]
            
            # Convert 3D points to grid coordinates for class mapping
            # DENSELY populate class_map first, regardless of occupancy
            if xyz_filtered.shape[0] > 0:
                grid_row = grid_dim // 2 - np.floor(xyz_filtered[:, 0] / GRID_RESOLUTION).astype(int)
                grid_col = grid_dim // 2 - np.floor(xyz_filtered[:, 1] / GRID_RESOLUTION).astype(int)
                
                # Filter points within grid bounds
                in_bounds = (grid_col >= 0) & (grid_col < grid_dim) & (grid_row >= 0) & (grid_row < grid_dim)
                grid_col = grid_col[in_bounds]
                grid_row = grid_row[in_bounds]
                seg_mask_in_bounds = seg_mask_filtered[in_bounds]
                
                # Densely populate class_map from YOLO detections
                # Use maximum class value when multiple points map to same cell
                for i in range(len(grid_row)):
                    row = grid_row[i]
                    col = grid_col[i]
                    class_idx = seg_mask_in_bounds[i]
                    if class_idx > 0:  # YOLO detected human (1) or other object (3)
                        # Keep maximum class (1 = human is higher priority than background)
                        if class_map[row, col] == 0 or class_idx == 1:
                            class_map[row, col] = class_idx
            
            # Apply brushfire algorithm to expand human labels to entire clusters
            class_map = ros_node.expand_human_labels_to_clusters(grid, class_map)
            
            # Now apply occupancy mask: only keep classes where grid is occupied
            # For occupied cells without YOLO detection, default to class 3 (generic obstacle)
            class_map_masked = np.where(grid == 1, 
                                       np.where(class_map > 0, class_map, 3),  # If occupied: use class_map or default to 3
                                       0)  # If free: class 0
            
            # Create occupancy grid visualization with color coding
            occupancy_grid_img = np.zeros((grid_dim, grid_dim, 3), dtype=np.uint8)
            occupancy_grid_img[grid == 0] = [255, 255, 255]  # Free space is white
            occupancy_grid_img[grid > 0] = [0, 0, 0]  # Occupied space is black (default)
            occupancy_grid_img[class_map_masked == 1] = [255, 0, 0]  # Human cells (class 1) are RED
            
            # Draw robot at center
            cv2.circle(occupancy_grid_img, (robot_x, robot_y), 3, (0, 165, 255), -1)
            
            # Run Poisson solver to compute safety field
            try:
                # Use binary grid for Poisson solver
                grid_binary = (grid > 0).astype(np.uint8)
                
                # CRITICAL FIX: Dilate class_map so boundary cells get obstacle class values!
                # The C++ code applies dh0 at FREE SPACE boundary cells, not obstacle cells.
                # It also INFLATES obstacles by robot size, so boundaries are ~3-5 cells away!
                # Dilate class 1 (human) and class 3 (objects) separately, with class 1 taking priority
                from scipy.ndimage import binary_dilation
                class_map_boundary = np.zeros_like(class_map)
                class_map_boundary = np.zeros_like(class_map_masked)
                
                # Use larger dilation to match C++ robot inflation radius
                dilation_struct = np.ones((DILATION_STRUCTURE_SIZE, DILATION_STRUCTURE_SIZE))
                
                # First dilate class 3 (objects) into adjacent free cells
                object_mask = (class_map_masked == 3)
                object_dilated = binary_dilation(object_mask, structure=dilation_struct)
                class_map_boundary[object_dilated & (grid == 0)] = 3
                
                # Then dilate class 1 (human) - this overwrites class 3 where they overlap!
                # Humans take priority for safety
                human_mask = (class_map_masked == 1)
                human_dilated = binary_dilation(human_mask, structure=dilation_struct)
                class_map_boundary[human_dilated & (grid == 0)] = 1
                
                # Also keep the original occupied cell values
                class_map_boundary[grid > 0] = class_map[grid > 0]
                
                # Set up dh0_classes array to map class indices to boundary strength values
                dh0_classes = [DH0_FREE, DH0_HUMAN, DH0_FREE, DH0_OBJECT]  # [class0=free, class1=human, class2=unused, class3=objects]
                
                # Debug: Verify dh0_classes is set correctly
                poisson_solver.set_dh0_classes([float(x) for x in dh0_classes])
                readback = poisson_solver.get_dh0_classes()
                # print(f"[DH0_CLASSES] Set: {dh0_classes}, Readback: {readback}")
                
                # Debug: Check class_map_boundary distribution
                unique_boundary, counts_boundary = np.unique(class_map_boundary, return_counts=True)
                boundary_stats = dict(zip(unique_boundary, counts_boundary))
                # print(f"[CLASS_MAP_BOUNDARY] After dilation: {boundary_stats}")

                # Run solver with DILATED class map so boundaries get correct class values
                h_field, guidance_x, guidance_y = run_poisson_solver(
                    grid_binary, class_map=class_map_boundary, dh0_classes=dh0_classes, ds=GRID_RESOLUTION
                )
                
                # Debug: Print class map statistics
                unique, counts = np.unique(class_map, return_counts=True)
                class_stats = dict(zip(unique, counts))
                # print(f"[CLASS_MAP] {class_stats} (0=free, 1=human, 3=object)")
                
                # Debug: Check guidance magnitude before smoothing
                mag_before = np.sqrt(guidance_x**2 + guidance_y**2)
                # print(f"[GUIDANCE] Before smoothing - max: {mag_before.max():.3f}, mean: {mag_before[mag_before>0].mean():.3f}")
                
                # Very light post-smoothing of guidance fields only for visualization
                # Use median filter to preserve edges better than Gaussian
                from scipy.ndimage import median_filter
                guidance_x = median_filter(guidance_x, size=GUIDANCE_SMOOTH_SIZE)
                guidance_y = median_filter(guidance_y, size=GUIDANCE_SMOOTH_SIZE)
                
                # Debug: Check guidance magnitude after smoothing
                mag_after = np.sqrt(guidance_x**2 + guidance_y**2)
                # print(f"[GUIDANCE] After smoothing - max: {mag_after.max():.3f}, mean: {mag_after[mag_after>0].mean():.3f}")

                # Normalize and colorize h_field
                h_field_norm = cv2.normalize(h_field, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                h_field_color = cv2.applyColorMap(h_field_norm, cv2.COLORMAP_VIRIDIS)
                
                # Draw guidance vectors as quivers
                occupancy_grid_quiver = occupancy_grid_img.copy()
                # Denser arrows around obstacles and edges
                quiver_step = QUIVER_STEP
                grid_h, grid_w = guidance_x.shape
                arrow_color = (0, 0, 255)  # Red arrows
                arrow_thickness = 1
                # Arrow length controls
                arrow_min_len = ARROW_MIN_LENGTH
                arrow_scale = ARROW_SCALE  # scales relative to quiver_step
                
                # Compute magnitude for arrow scaling
                mag = np.sqrt(guidance_x**2 + guidance_y**2)
                mag_max = float(mag[mag > 0].max()) if np.any(mag > 0) else 1.0
                
                for y in range(0, grid_h, quiver_step):
                    for x in range(0, grid_w, quiver_step):
                        gx = guidance_x[y, x]  # Fixed: use row,col indexing
                        gy = guidance_y[y, x]  # Fixed: use row,col indexing
                        norm = float(np.sqrt(gx**2 + gy**2))
                        if norm > 1e-6:  # threshold to avoid noise
                            start_pt = (x, y)
                            # Compute arrow pixel length proportional to magnitude, clamped
                            # Base length relative to quiver_step
                            target_len = int(max(arrow_min_len, min(quiver_step * arrow_scale, (norm / mag_max) * quiver_step * arrow_scale)))
                            # Note: guidance_x is in grid i-direction (rows, Y in image)
                            # guidance_y is in grid j-direction (cols, X in image)
                            # Swap to image coordinates
                            dx = int((gy / norm) * target_len)  # horizontal
                            dy = int((gx / norm) * target_len)  # vertical
                            end_pt = (start_pt[0] + dx, start_pt[1] + dy)
                            cv2.arrowedLine(occupancy_grid_quiver, start_pt, end_pt, arrow_color, arrow_thickness, tipLength=0.3)
                
                # Blend h_field with occupancy grid
                blended = cv2.addWeighted(occupancy_grid_quiver, 1 - OVERLAY_ALPHA, h_field_color, OVERLAY_ALPHA, 0)
            except Exception as e:
                print(f"Error in Poisson solver: {e}")
                import traceback
                traceback.print_exc()
                blended = occupancy_grid_img
            
            # Resize images to larger size for better visibility and display side by side
            display_scale = DISPLAY_SCALE  # Scale factor for display
            h1, w1 = blended.shape[:2]
            h2, w2 = cv2_img.shape[:2]
            target_height = int(max(h1, h2) * display_scale)
            occupancy_resized = cv2.resize(blended, (int(w1 * target_height / h1), target_height))
            cv2_img_resized = cv2.resize(cv2_img, (int(w2 * target_height / h2), target_height))
            side_by_side = np.hstack([occupancy_resized, cv2_img_resized])
            
            # Apply final window scale for OpenCV display
            if CV2_WINDOW_SCALE != 1.0:
                final_h, final_w = side_by_side.shape[:2]
                side_by_side = cv2.resize(side_by_side, 
                                         (int(final_w * CV2_WINDOW_SCALE), int(final_h * CV2_WINDOW_SCALE)))
            
            cv2.imshow('Occupancy Grid | YOLO Segmentation', side_by_side)
            
            # Update matplotlib visualization (reuse figure)
            guidance_mag = np.sqrt(guidance_x**2 + guidance_y**2)
            
            # Update or create images in each subplot
            # Row 0, Col 0: Occupancy Grid (display flipped vertically, origin='lower')
            Hh, Ww = occupancy_grid_img.shape[:2]
            occ_disp = np.flipud(occupancy_grid_img)
            if im_objs[0] is None:
                axes[0, 0].clear()
                im_objs[0] = axes[0, 0].imshow(
                    occ_disp, origin='lower', extent=[0, Ww, 0, Hh]
                )
                axes[0, 0].set_title('Occupancy Grid')
                axes[0, 0].axis('off')
            else:
                im_objs[0].set_data(occ_disp)
                im_objs[0].set_extent([0, Ww, 0, Hh])
            
            # Row 0, Col 1: Class Map
            if im_objs[1] is None:
                axes[0, 1].clear()
                im_objs[1] = axes[0, 1].imshow(class_map, cmap='hot', origin='upper', vmin=0, vmax=3)
                axes[0, 1].set_title('Class Map (0=free, 1=human, 3=object)')
                axes[0, 1].axis('off')
                plt.colorbar(im_objs[1], ax=axes[0, 1], fraction=0.046, ticks=[0, 1, 3])
            else:
                im_objs[1].set_data(class_map)
                im_objs[1].set_clim(vmin=0, vmax=3)
            
            # Row 0, Col 2: H-Field
            if im_objs[2] is None:
                axes[0, 2].clear()
                im_objs[2] = axes[0, 2].imshow(h_field, cmap='coolwarm', origin='upper')
                axes[0, 2].set_title('H-Field (Safety Function)')
                axes[0, 2].axis('off')
                plt.colorbar(im_objs[2], ax=axes[0, 2], fraction=0.046)
            else:
                im_objs[2].set_data(h_field)
                im_objs[2].set_clim(vmin=h_field.min(), vmax=h_field.max())
            
            # Row 1, Col 0: Guidance X
            gx_max = max(abs(guidance_x.min()), abs(guidance_x.max()), 0.1)
            if im_objs[3] is None:
                axes[1, 0].clear()
                im_objs[3] = axes[1, 0].imshow(guidance_x, cmap='RdBu', origin='upper', vmin=-gx_max, vmax=gx_max)
                axes[1, 0].set_title('Guidance X')
                axes[1, 0].axis('off')
                plt.colorbar(im_objs[3], ax=axes[1, 0], fraction=0.046)
            else:
                im_objs[3].set_data(guidance_x)
                im_objs[3].set_clim(vmin=-gx_max, vmax=gx_max)
            
            # Row 1, Col 1: Guidance Y
            gy_max = max(abs(guidance_y.min()), abs(guidance_y.max()), 0.1)
            if im_objs[4] is None:
                axes[1, 1].clear()
                im_objs[4] = axes[1, 1].imshow(guidance_y, cmap='RdBu', origin='upper', vmin=-gy_max, vmax=gy_max)
                axes[1, 1].set_title('Guidance Y')
                axes[1, 1].axis('off')
                plt.colorbar(im_objs[4], ax=axes[1, 1], fraction=0.046)
            else:
                im_objs[4].set_data(guidance_y)
                im_objs[4].set_clim(vmin=-gy_max, vmax=gy_max)
            
            # Row 1, Col 2: Guidance Field Quiver (redraw each time)
            axes[1, 2].clear()
            axes[1, 2].imshow(occupancy_grid_img, origin='upper', alpha=0.3)
            step = 8
            Y_grid, X_grid = np.mgrid[0:grid_h:step, 0:grid_w:step]
            U_grid = guidance_y[::step, ::step]
            V_grid = guidance_x[::step, ::step]
            
            # Compute magnitude for adaptive scaling and coloring
            mag_grid = np.sqrt(U_grid**2 + V_grid**2)
            max_mag = mag_grid.max() if mag_grid.max() > 0 else 1.0
            
            # Remove old colorbar if it exists (only on error/special cases)
            # Note: we reuse the colorbar instead of removing it to avoid plot shrinking
            
            # Plot with magnitude-based arrow lengths and colors
            if max_mag > 0.01:
                # scale controls arrow length, smaller scale = longer arrows
                # Scale reduced from 30 to 15 to make arrows 2x longer
                q = axes[1, 2].quiver(X_grid, Y_grid, U_grid, -V_grid, mag_grid, 
                                     cmap='plasma', scale=max_mag*15, scale_units='width', 
                                     width=0.004, alpha=0.8)
                # Create or update colorbar for magnitude
                if quiver_cbar is None:
                    # Create colorbar on first frame
                    quiver_cbar = plt.colorbar(q, ax=axes[1, 2], fraction=0.046)
                    quiver_cbar.set_label('Magnitude', fontsize=8)
                else:
                    # Update existing colorbar with new data
                    quiver_cbar.update_normal(q)
            else:
                axes[1, 2].text(grid_w//2, grid_h//2, 'No guidance vectors (magnitude too small)', 
                               ha='center', va='center', fontsize=10, color='red')
            axes[1, 2].set_title(f'Guidance Field Vectors (max: {max_mag:.3f})')
            axes[1, 2].axis('off')
            
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            
            key = cv2.waitKey(1)
            if key == 27:
                break       
    cv2.destroyAllWindows()
    # Cleanup ROS2 (if available)
    if ROS2_AVAILABLE and ros_node is not None:
        ros_node.shutdown()  # Stop background thread first
        ros_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()