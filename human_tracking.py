#!/usr/bin/env python3
"""
Human tracking with gimbal control.
Subscribes to human centroid from yolo_zed_ros and controls gimbal yaw.
All configuration parameters are exposed as ROS parameters.
"""

import numpy as np
import time

# ============================================================================
# DEFAULT CONFIGURATION (overridden by ROS parameters)
# ============================================================================

DEFAULT_MAX_ROTATION_DEGREES = 45.0
DEFAULT_DEADZONE_PIXELS = 50
DEFAULT_P_GAIN = 0.05
DEFAULT_MAX_STEP_TICKS = 100
DEFAULT_YAW_SERVO_ID = 1
DEFAULT_PITCH_SERVO_ID = 2
DEFAULT_YAW_ZERO_POSITION = 2125
DEFAULT_PITCH_ZERO_POSITION = 1702
DEFAULT_PITCH_UP_DEGREES = 30.0
DEFAULT_DEVICE_PORT = '/dev/ttyUSB0'
DEFAULT_BAUDRATE = 57600

# ============================================================================


# ROS2 imports
try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Point, TransformStamped
    from tf2_ros import TransformBroadcaster
    from unitree_go.msg import SportModeState
    import tf_transformations
    ROS2_AVAILABLE = True
except (ModuleNotFoundError, ImportError) as e:
    print(f"[ERROR] ROS2 not available: {e}")
    print("This script requires ROS2")
    exit(1)

# Gimbal control imports
try:
    from dynamixel_sdk import *
    GIMBAL_AVAILABLE = True
except (ModuleNotFoundError, ImportError) as e:
    print(f"[ERROR] Dynamixel SDK not available: {e}")
    print("Install with: pip install dynamixel-sdk")
    exit(1)


class GimbalController:
    """Controls Dynamixel gimbal servos"""
    
    def __init__(self, device, baudrate, yaw_servo_id, pitch_servo_id,
                 yaw_zero_position, pitch_zero_position, pitch_up_degrees,
                 max_rotation_degrees):
        # Servo configuration
        self.PROTOCOL_VERSION = 2.0
        self.YAW_ID = yaw_servo_id
        self.PITCH_ID = pitch_servo_id
        
        # Control table addresses
        self.ADDR_TORQUE_ENABLE = 64
        self.ADDR_GOAL_POSITION = 116
        self.ADDR_PRESENT_POSITION = 132
        
        self.TORQUE_ENABLE = 1
        self.TORQUE_DISABLE = 0
        
        # Zero position calibration
        self.YAW_ZERO = yaw_zero_position
        self.PITCH_ZERO = pitch_zero_position
        self.PITCH_UP_DEG = pitch_up_degrees
        # Calculate pitch-up offset in ticks (4096 ticks = 360 degrees)
        self.PITCH_UP_OFFSET = int(pitch_up_degrees * 4096.0 / 360.0)
        
        # Calculate position limits from max_rotation_degrees
        # 4096 ticks = 360 degrees, so ticks = degrees * 4096 / 360
        max_ticks = int(max_rotation_degrees * 4096.0 / 360.0)
        self.MAX_POS_OFFSET = max_ticks
        self.MIN_POS_OFFSET = -max_ticks
        
        print(f"[Gimbal] Rotation limit: ±{max_rotation_degrees}° (±{max_ticks} ticks)")
        
        # Initialize communication
        self.port_handler = PortHandler(device)
        self.packet_handler = PacketHandler(self.PROTOCOL_VERSION)
        
        # Open port
        if not self.port_handler.openPort():
            raise RuntimeError("Failed to open port")
        
        if not self.port_handler.setBaudRate(baudrate):
            raise RuntimeError("Failed to set baudrate")
        
        print(f"[Gimbal] Port opened successfully")
        
        # Enable torque on both servos
        for servo_id, name in [(self.YAW_ID, "yaw"), (self.PITCH_ID, "pitch")]:
            result, error = self.packet_handler.write1ByteTxRx(
                self.port_handler, servo_id, 
                self.ADDR_TORQUE_ENABLE, self.TORQUE_ENABLE
            )
            
            if result != COMM_SUCCESS or error != 0:
                raise RuntimeError(f"Failed to enable torque on {name} servo (ID {servo_id})")
            
            print(f"[Gimbal] Torque enabled on {name} servo (ID {servo_id})")
        
        # Move both servos to zero position
        self.move_to_zero()
        self.current_yaw_position = self.YAW_ZERO
        self.current_pitch_position = self.PITCH_ZERO
    
    def move_to_zero(self):
        """Move both servos to zero position"""
        # Move yaw to zero
        self.set_yaw_position(0)
        # Move pitch to angled-up position
        self.set_pitch_up()
        time.sleep(0.5)  # Wait for movement
        print("[Gimbal] Moved to zero position (yaw=0, pitch=level)")
    
    def set_yaw_position(self, offset_ticks):
        """
        Set yaw gimbal position relative to zero.
        
        Args:
            offset_ticks: Position offset in ticks (positive = left/CCW)
        """
        # Clamp to limits
        offset_ticks = max(self.MIN_POS_OFFSET, min(self.MAX_POS_OFFSET, offset_ticks))
        
        # Calculate absolute position
        target_position = self.YAW_ZERO + offset_ticks
        target_position = max(0, min(4095, target_position))
        
        # Send command
        result, error = self.packet_handler.write4ByteTxRx(
            self.port_handler, self.YAW_ID,
            self.ADDR_GOAL_POSITION, target_position
        )
        
        if result == COMM_SUCCESS and error == 0:
            self.current_yaw_position = target_position
            return True
        return False
    
    def set_pitch_up(self):
        """
        Hold pitch servo at angled-up position (PITCH_UP_DEGREES from level).
        """
        # Calculate target position with pitch-up offset (subtract to pitch up)
        target_position = self.PITCH_ZERO - self.PITCH_UP_OFFSET
        target_position = max(0, min(4095, target_position))
        
        result, error = self.packet_handler.write4ByteTxRx(
            self.port_handler, self.PITCH_ID,
            self.ADDR_GOAL_POSITION, target_position
        )
        
        if result == COMM_SUCCESS and error == 0:
            self.current_pitch_position = target_position
            return True
        return False
    
    def get_pitch_angle_degrees(self):
        """Get current pitch angle in degrees (0 at level, positive = up)"""
        offset = self.PITCH_ZERO - self.current_pitch_position  # Negated for correct direction
        return offset * 360.0 / 4096.0
    
    def get_yaw_position_offset(self):
        """Get current yaw position offset from zero in ticks"""
        return self.current_yaw_position - self.YAW_ZERO
    
    def get_yaw_angle_degrees(self):
        """Get current yaw angle in degrees (0 at zero position)"""
        offset = self.get_yaw_position_offset()
        # Convert ticks to degrees (4096 ticks = 360 degrees)
        return -offset * 360.0 / 4096.0
    
    def cleanup(self):
        """Disable torque and close port"""
        try:
            for servo_id, name in [(self.YAW_ID, "yaw"), (self.PITCH_ID, "pitch")]:
                self.packet_handler.write1ByteTxRx(
                    self.port_handler, servo_id,
                    self.ADDR_TORQUE_ENABLE, self.TORQUE_DISABLE
                )
                print(f"[Gimbal] Torque disabled on {name} servo")
        except:
            pass
        
        try:
            self.port_handler.closePort()
            print(f"[Gimbal] Port closed")
        except:
            pass


class HumanTrackingNode(Node):
    """ROS2 node that tracks humans and controls gimbal"""
    
    def __init__(self):
        super().__init__('human_tracking_node')
        
        # Declare all ROS parameters with defaults
        self.declare_parameter('max_rotation_degrees', DEFAULT_MAX_ROTATION_DEGREES)
        self.declare_parameter('deadzone_pixels', DEFAULT_DEADZONE_PIXELS)
        self.declare_parameter('p_gain', DEFAULT_P_GAIN)
        self.declare_parameter('max_step_ticks', DEFAULT_MAX_STEP_TICKS)
        self.declare_parameter('yaw_servo_id', DEFAULT_YAW_SERVO_ID)
        self.declare_parameter('pitch_servo_id', DEFAULT_PITCH_SERVO_ID)
        self.declare_parameter('yaw_zero_position', DEFAULT_YAW_ZERO_POSITION)
        self.declare_parameter('pitch_zero_position', DEFAULT_PITCH_ZERO_POSITION)
        self.declare_parameter('pitch_up_degrees', DEFAULT_PITCH_UP_DEGREES)
        self.declare_parameter('device_port', DEFAULT_DEVICE_PORT)
        self.declare_parameter('baudrate', DEFAULT_BAUDRATE)
        
        # Get parameter values
        max_rotation_degrees = self.get_parameter('max_rotation_degrees').value
        self.DEADZONE = self.get_parameter('deadzone_pixels').value
        self.P_GAIN = self.get_parameter('p_gain').value
        self.MAX_STEP = self.get_parameter('max_step_ticks').value
        yaw_servo_id = self.get_parameter('yaw_servo_id').value
        pitch_servo_id = self.get_parameter('pitch_servo_id').value
        yaw_zero_position = self.get_parameter('yaw_zero_position').value
        pitch_zero_position = self.get_parameter('pitch_zero_position').value
        pitch_up_degrees = self.get_parameter('pitch_up_degrees').value
        device_port = self.get_parameter('device_port').value
        baudrate = self.get_parameter('baudrate').value
        
        # Initialize gimbal with parameters
        self.get_logger().info('Initializing gimbal controller...')
        try:
            self.gimbal = GimbalController(
                device=device_port,
                baudrate=baudrate,
                yaw_servo_id=yaw_servo_id,
                pitch_servo_id=pitch_servo_id,
                yaw_zero_position=yaw_zero_position,
                pitch_zero_position=pitch_zero_position,
                pitch_up_degrees=pitch_up_degrees,
                max_rotation_degrees=max_rotation_degrees
            )
        except Exception as e:
            self.get_logger().error(f'Failed to initialize gimbal: {e}')
            raise
        
        self.last_centroid = None
        self.image_width = None
        
        # Robot pose in odom frame (from sportmodestate)
        self.robot_position = [0.0, 0.0, 0.0]  # x, y, z
        self.robot_rpy = [0.0, 0.0, 0.0]  # roll, pitch, yaw
        
        # Subscribe to human centroid from yolo_zed_ros
        self.centroid_sub = self.create_subscription(
            Point,
            '/human_tracking/centroid',
            self.centroid_callback,
            10
        )
        
        # Subscribe to robot pose for odom frame transform
        self.pose_sub = self.create_subscription(
            SportModeState,
            'sportmodestate',
            self.pose_callback,
            10
        )
        
        # Create transform broadcaster for gimbal rotation
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Camera offset from robot body center (in body frame)
        # Testing: X is forward, Y is left, Z is up
        self.camera_offset_x = 0.14  # 14 cm forward (X-axis)
        self.camera_offset_y = 0.0
        self.camera_offset_z = 0.2  # 20 cm above
        
        # Create 100 Hz control timer for smooth gimbal control
        control_rate = 100  # Hz
        self.control_timer = self.create_timer(1.0 / control_rate, self.control_loop)
        
        self.get_logger().info('Human tracking node initialized')
        self.get_logger().info(f'Deadzone: {self.DEADZONE}px, P_gain: {self.P_GAIN}, Max step: {self.MAX_STEP} ticks')
        self.get_logger().info(f'Pitch up: {pitch_up_degrees}°, Max rotation: ±{max_rotation_degrees}°')
        self.get_logger().info(f'Control loop rate: {control_rate} Hz')
    
    def centroid_callback(self, msg):
        """Receive human centroid from yolo_zed_ros - just store the latest data"""
        # Extract data: x, y are pixel coordinates, z is image width
        self.last_centroid = (msg.x, msg.y)
        self.image_width = int(msg.z)
    
    def pose_callback(self, msg):
        """Receive robot pose from sportmodestate"""
        self.robot_position[0] = msg.position[0]
        self.robot_position[1] = msg.position[1]
        self.robot_position[2] = msg.position[2]
        self.robot_rpy[0] = msg.imu_state.rpy[0]
        self.robot_rpy[1] = msg.imu_state.rpy[1]
        self.robot_rpy[2] = msg.imu_state.rpy[2]
    
    def control_loop(self):
        """
        100 Hz control loop - executes gimbal control based on latest centroid data.
        Runs continuously regardless of centroid update rate for smooth motion.
        """
        # Always publish transform at control rate
        self.publish_transform()
        
        # Check if we have centroid data
        if self.last_centroid is None or self.image_width is None:
            return
        
        human_x, human_y = self.last_centroid
        
        # Calculate control
        image_center_x = self.image_width // 2
        error_x = human_x - image_center_x
        
        # Apply deadzone
        if abs(error_x) < self.DEADZONE:
            # Still hold pitch up even when not moving yaw
            self.gimbal.set_pitch_up()
            return
        
        # Calculate control output
        # Positive error = human is to the right, should rotate right
        # Negative error = human is to the left, should rotate left
        control = error_x * self.P_GAIN
        
        # Clamp to max step
        control = max(-self.MAX_STEP, min(self.MAX_STEP, control))
        
        # Update yaw gimbal position
        current_offset = self.gimbal.get_yaw_position_offset()
        new_offset = int(current_offset + control)
        
        success = self.gimbal.set_yaw_position(new_offset)
        
        # Always hold pitch up
        self.gimbal.set_pitch_up()
        
        if success:
            angle = self.gimbal.get_yaw_angle_degrees()
            # Only log occasionally to avoid spam (every 50 iterations = 0.5 sec)
            if hasattr(self, '_log_counter'):
                self._log_counter += 1
            else:
                self._log_counter = 0
                
            if self._log_counter % 50 == 0:
                self.get_logger().info(
                    f'Tracking: error={error_x:+.0f}px, control={control:+.1f}, '
                    f'angle={angle:+.1f}°'
                )
    
    def publish_transform(self):
        """Publish transform from odom to camera_link (includes robot pose + gimbal rotation)"""
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = 'odom'
        transform.child_frame_id = 'camera_link'
        
        # Get gimbal angles
        gimbal_yaw_deg = self.gimbal.get_yaw_angle_degrees()
        gimbal_pitch_deg = self.gimbal.get_pitch_angle_degrees()
        gimbal_yaw_rad = np.radians(gimbal_yaw_deg)
        gimbal_pitch_rad = np.radians(gimbal_pitch_deg)
        
        # Robot body rotation quaternion (from IMU)
        body_quat = tf_transformations.quaternion_from_euler(
            self.robot_rpy[0], self.robot_rpy[1], self.robot_rpy[2]
        )
        
        # Gimbal rotation quaternion (relative to body)
        # Note: pitch is negated to match camera orientation convention
        gimbal_quat = tf_transformations.quaternion_from_euler(0, -gimbal_pitch_rad, gimbal_yaw_rad)
        
        # Combined rotation: body * gimbal (order matters!)
        combined_quat = tf_transformations.quaternion_multiply(body_quat, gimbal_quat)
        
        # Camera offset in body frame, rotated to odom frame using body rotation
        # Build rotation matrix from body quaternion
        body_rot_matrix = tf_transformations.quaternion_matrix(body_quat)[:3, :3]
        offset_body = np.array([self.camera_offset_x, self.camera_offset_y, self.camera_offset_z])
        offset_odom = body_rot_matrix @ offset_body
        
        # Final position: robot position + rotated camera offset
        transform.transform.translation.x = self.robot_position[0] + offset_odom[0]
        transform.transform.translation.y = self.robot_position[1] + offset_odom[1]
        transform.transform.translation.z = self.robot_position[2] + offset_odom[2]
        
        transform.transform.rotation.x = combined_quat[0]
        transform.transform.rotation.y = combined_quat[1]
        transform.transform.rotation.z = combined_quat[2]
        transform.transform.rotation.w = combined_quat[3]
        
        self.tf_broadcaster.sendTransform(transform)
        
        # Also publish odom -> body_link transform (for livox_frame chain)
        body_transform = TransformStamped()
        body_transform.header.stamp = self.get_clock().now().to_msg()
        body_transform.header.frame_id = 'odom'
        body_transform.child_frame_id = 'body_link'
        
        body_transform.transform.translation.x = float(self.robot_position[0])
        body_transform.transform.translation.y = float(self.robot_position[1])
        body_transform.transform.translation.z = float(self.robot_position[2])
        
        body_transform.transform.rotation.x = float(body_quat[0])
        body_transform.transform.rotation.y = float(body_quat[1])
        body_transform.transform.rotation.z = float(body_quat[2])
        body_transform.transform.rotation.w = float(body_quat[3])
        
        self.tf_broadcaster.sendTransform(body_transform)


def main(args=None):
    # Initialize ROS2
    rclpy.init(args=args)
    
    # Create ROS2 node (gimbal is initialized inside with ROS parameters)
    print("[Init] Creating ROS2 node...")
    try:
        node = HumanTrackingNode()
    except Exception as e:
        print(f"[ERROR] Failed to initialize: {e}")
        rclpy.shutdown()
        return
    
    print("\n=== Human Tracking Active ===")
    print("Subscribing to /human_tracking/centroid from yolo_zed_ros")
    print("Press Ctrl+C to exit\n")
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[Exit] Interrupted by user")
    finally:
        # Cleanup
        print("[Cleanup] Shutting down...")
        if hasattr(node, 'gimbal'):
            node.gimbal.cleanup()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except:
            pass  # Already shutdown
        print("[Exit] Shutdown complete")


if __name__ == "__main__":
    main()
