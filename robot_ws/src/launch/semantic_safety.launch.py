#!/usr/bin/env python3
"""
Launch file for semantic safety field pipeline with YOLO detection
Supports both RealSense and ZED cameras via camera_type argument
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

# Detect if running on Jetson (NVIDIA Tegra platform)
IS_JETSON = os.path.exists('/etc/nv_tegra_release')


def generate_launch_description():
    # Declare arguments
    camera_type_arg = DeclareLaunchArgument(
        'camera_type',
        default_value='realsense',
        description='Camera type: realsense or zed'
    )
    
    camera_model_arg = DeclareLaunchArgument(
        'camera_model',
        default_value='zed2i',
        description='ZED camera model (only used if camera_type=zed)'
    )
    
    camera_fps_arg = DeclareLaunchArgument(
        'camera_fps',
        default_value='15',
        description='Camera FPS (RealSense: 6, 15, 30, or 60)'
    )
    
    yolo_model_arg = DeclareLaunchArgument(
        'yolo_model',
        default_value='/home/unitree/semantic-safety-master/yolo11n-seg.engine',
        description='Path to YOLO model file'
    )
    
    use_tensorrt_arg = DeclareLaunchArgument(
        'use_tensorrt',
        default_value='true' if IS_JETSON else 'false',
        description='Use TensorRT for YOLO inference (auto-enabled on Jetson)'
    )
    
    point_cloud_density_arg = DeclareLaunchArgument(
        'point_cloud_density',
        default_value='0.1',
        description='Point cloud density for camera'
    )
    
    dh0_human_arg = DeclareLaunchArgument(
        'dh0_human',
        default_value='1.7', # '0.5',
        description='Safety field gradient magnitude for humans (stronger repulsion)'
    )
    
    dh0_obstacle_arg = DeclareLaunchArgument(
        'dh0_obstacle',
        default_value='0.5',
        description='Safety field gradient magnitude for generic obstacles'
    )
    
    min_z_arg = DeclareLaunchArgument(
        'min_z',
        default_value='0.1',
        description='Minimum height (Z) for point cloud filtering in occupancy grid'
    )
    
    max_z_arg = DeclareLaunchArgument(
        'max_z',
        default_value='0.5',
        description='Maximum height (Z) for point cloud filtering in occupancy grid'
    )
    
    # Tight-area wall softening parameters
    tight_area_human_threshold_arg = DeclareLaunchArgument(
        'tight_area_human_threshold',
        default_value='1.0',
        description='Distance threshold (m) to nearest human for tight-area detection'
    )
    
    tight_area_h_threshold_arg = DeclareLaunchArgument(
        'tight_area_h_threshold',
        default_value='0.3',
        description='h-value threshold for tight-area detection (lower = closer to obstacles)'
    )
    
    tight_area_wall_slack_arg = DeclareLaunchArgument(
        'tight_area_wall_slack',
        default_value='0.0',
        description='Slack added to wall boundaries in tight areas (negative allows wall approach)'
    )
    
    # Human tracker lifecycle parameters
    human_track_timeout_arg = DeclareLaunchArgument(
        'human_track_timeout_sec',
        default_value='10.0',
        description='How long (sec) to keep track after last observation before deletion'
    )
    
    human_track_gate_radius_arg = DeclareLaunchArgument(
        'human_track_gate_radius',
        default_value='0.8',
        description='Maximum distance (m) to associate LiDAR cluster with existing track'
    )
    
    human_track_velocity_decay_tau_arg = DeclareLaunchArgument(
        'human_track_velocity_decay_tau',
        default_value='1.0',
        description='Time constant (sec) for velocity decay when human stops'
    )
    
    velocity_threshold_arg = DeclareLaunchArgument(
        'human_track_velocity_threshold',
        default_value='0.2',
        description='Velocity threshold (m/s) to consider cluster as moving human'
    )
    
    min_yolo_cells_arg = DeclareLaunchArgument(
        'min_yolo_cells',
        default_value='20',
        description='Minimum YOLO cells required to confirm cluster as human'
    )
    
    enable_human_tracker_dilation_arg = DeclareLaunchArgument(
        'enable_human_tracker_dilation',
        default_value='true',
        description='Enable brushfire dilation of human labels (set to false to disable)'
    )
    
    decay_in_fov_arg = DeclareLaunchArgument(
        'decay_in_fov',
        default_value='0.85',
        description='Decay rate when cluster is in camera FOV but no YOLO detection (~0.5 sec)'
    )
    
    decay_stationary_arg = DeclareLaunchArgument(
        'decay_stationary',
        default_value='0.99',
        description='Decay rate when cluster is stationary outside camera FOV (~10 sec)'
    )
    
    decay_unconfirmed_arg = DeclareLaunchArgument(
        'decay_unconfirmed',
        default_value='0.95',
        description='Decay rate for clusters never confirmed by YOLO (~1.5 sec)'
    )
    
    no_retrack_on_move_arg = DeclareLaunchArgument(
        'no_retrack_on_move',
        default_value='true',
        description='If true, lost human tracks will not be re-tracked when cluster moves again (requires fresh YOLO confirmation)'
    )
    
    social_tangent_bias_arg = DeclareLaunchArgument(
        'social_tangent_bias',
        default_value='1.2',
        description='Tangential bias strength when social navigation is enabled. 0=pure repulsion, 1=strong leftward bias'
    )

    social_tangent_layers_arg = DeclareLaunchArgument(
        'social_tangent_layers',
        default_value='8',
        description='Number of layers outward from human boundary with tangential bias (0=disable, preserves CBF at boundary)'
    )
    
    social_layer_thickness_arg = DeclareLaunchArgument(
        'social_layer_thickness',
        default_value='1',
        description='Grid cells per layer (circular dilation radius)'
    )
    
    human_direction_threshold_arg = DeclareLaunchArgument(
        'human_direction_threshold',
        default_value='0.3',
        description='Dot product threshold for switching tangent direction. Higher = more stable, needs clearer walk direction'
    )
    
    robot_mos_human_arg = DeclareLaunchArgument(
        'robot_mos_human',
        default_value='1.0',
        description='Margin of Safety multiplier for human boundary inflation. Larger = more buffer around humans'
    )
    
    robot_mos_obstacle_arg = DeclareLaunchArgument(
        'robot_mos_obstacle',
        default_value='1.0',
        description='Margin of Safety multiplier for obstacle boundary inflation. Smaller = closer to walls'
    )
    
    enable_social_navigation_arg = DeclareLaunchArgument(
        'enable_social_navigation',
        default_value='true',
        description='Enable social navigation - robot passes on human right (goes left)'
    )
      
    cbf_sigma_epsilon_arg = DeclareLaunchArgument(
        'cbf_sigma_epsilon',
        default_value='0.1',
        description='CBF sigma function saturation value (prevents division by zero near boundary)'
    )
    
    cbf_sigma_kappa_arg = DeclareLaunchArgument(
        'cbf_sigma_kappa',
        default_value='5.0',
        description='CBF sigma function transition rate'
    )
    
    # Velocity bound parameters
    vel_max_x_fwd_arg = DeclareLaunchArgument(
        'vel_max_x_fwd',
        default_value='1.1', # '0.75',
        description='Maximum forward velocity (m/s)'
    )
    
    vel_max_x_bwd_arg = DeclareLaunchArgument(
        'vel_max_x_bwd',
        default_value='0.75',
        description='Maximum backward velocity (m/s)'
    )
    
    vel_max_y_arg = DeclareLaunchArgument(
        'vel_max_y',
        default_value='0.75',
        description='Maximum lateral velocity (m/s)'
    )
    
    vel_max_yaw_arg = DeclareLaunchArgument(
        'vel_max_yaw',
        default_value='0.75',
        description='Maximum yaw rate (rad/s)'
    )
    
    logging_publish_hz_arg = DeclareLaunchArgument(
        'logging_publish_hz',
        default_value='10.0',
        description='Frequency (Hz) at which to publish visualization images and logging data'
    )
    
    enable_data_logging_to_file_arg = DeclareLaunchArgument(
        'enable_data_logging_to_file',
        default_value='false',
        description='Enable CSV and BIN data logging for experiments'
    )
    
    # RealSense camera launch
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('unitree_ros2_poisson_simple'),
                'launch',
                'camera_realsense.launch.py'
            ])
        ]),
        launch_arguments={
            'fps': LaunchConfiguration('camera_fps'),
            'point_cloud_density': LaunchConfiguration('point_cloud_density')
        }.items(),
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration('camera_type'), "' == 'realsense'"])
        )
    )
    
    # ZED camera launch
    zed_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('unitree_ros2_poisson_simple'),
                'launch',
                'camera_zed.launch.py'
            ])
        ]),
        launch_arguments={'camera_model': LaunchConfiguration('camera_model')}.items(),
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration('camera_type'), "' == 'zed'"])
        )
    )
    
    # Cloud Merger Node
    cloud_merger_node = Node(
        package='unitree_ros2_poisson_simple',
        executable='cloudMerger',
        name='cloud_merger',
        output='screen',
    )
    
    # YOLO Detector Node
    yolo_detector_node = Node(
        package='unitree_ros2_poisson_simple',
        executable='yolo_detector.py',
        name='yolo_detector',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('yolo_model'),
            'confidence_threshold': 0.5,
            'use_tensorrt': LaunchConfiguration('use_tensorrt'),
            'grid_imax': 100,  # Must match poisson.h IMAX
            'grid_jmax': 100,  # Must match poisson.h JMAX
            'grid_ds': 0.05,   # Must match poisson.h DS
            'logging_publish_hz': LaunchConfiguration('logging_publish_hz'),
        }],
    )
    
    # Semantic Poisson Node (includes CloudMergerNode internally)
    semantic_poisson_node = Node(
        package='unitree_ros2_poisson_simple',
        executable='semantic_poisson',
        name='semantic_poisson',
        output='screen',
        parameters=[{
            'dh0_human': LaunchConfiguration('dh0_human'),
            'dh0_obstacle': LaunchConfiguration('dh0_obstacle'),
            'enable_display': False,  # Show OpenCV visualization
            'min_z': LaunchConfiguration('min_z'),  # Forwarded to CloudMergerNode
            'max_z': LaunchConfiguration('max_z'),  # Forwarded to CloudMergerNode
            # Tight-area wall softening
            'tight_area_human_threshold': LaunchConfiguration('tight_area_human_threshold'),
            'tight_area_h_threshold': LaunchConfiguration('tight_area_h_threshold'),
            'tight_area_wall_slack': LaunchConfiguration('tight_area_wall_slack'),
            # Human tracker lifecycle
            'human_track_timeout_sec': LaunchConfiguration('human_track_timeout_sec'),
            'human_track_gate_radius': LaunchConfiguration('human_track_gate_radius'),
            'human_track_velocity_decay_tau': LaunchConfiguration('human_track_velocity_decay_tau'),
            'human_track_velocity_threshold': LaunchConfiguration('human_track_velocity_threshold'),
            'min_yolo_cells': LaunchConfiguration('min_yolo_cells'),
            'enable_human_tracker_dilation': LaunchConfiguration('enable_human_tracker_dilation'),
            'decay_in_fov': LaunchConfiguration('decay_in_fov'),
            'decay_stationary': LaunchConfiguration('decay_stationary'),
            'decay_unconfirmed': LaunchConfiguration('decay_unconfirmed'),
            'no_retrack_on_move': LaunchConfiguration('no_retrack_on_move'),
            'enable_social_navigation': LaunchConfiguration('enable_social_navigation'),
            'social_tangent_bias': LaunchConfiguration('social_tangent_bias'),
            'social_tangent_layers': LaunchConfiguration('social_tangent_layers'),
            'social_layer_thickness': LaunchConfiguration('social_layer_thickness'),
            'human_direction_threshold': LaunchConfiguration('human_direction_threshold'),
            'robot_mos_human': LaunchConfiguration('robot_mos_human'),
            'robot_mos_obstacle': LaunchConfiguration('robot_mos_obstacle'),
            'cbf_sigma_epsilon': LaunchConfiguration('cbf_sigma_epsilon'),
            'cbf_sigma_kappa': LaunchConfiguration('cbf_sigma_kappa'),
            # Velocity bounds
            'vel_max_x_fwd': LaunchConfiguration('vel_max_x_fwd'),
            'vel_max_x_bwd': LaunchConfiguration('vel_max_x_bwd'),
            'vel_max_y': LaunchConfiguration('vel_max_y'),
            'vel_max_yaw': LaunchConfiguration('vel_max_yaw'),
            'logging_publish_hz': LaunchConfiguration('logging_publish_hz'),
            'enable_data_logging_to_file': LaunchConfiguration('enable_data_logging_to_file'),
        }],
    )
    
    # Human Tracking Node (Gimbal Control)
    human_tracking_node = Node(
        package='unitree_ros2_poisson_simple',
        executable='human_tracking.py',
        name='human_tracking',
        output='screen',
        parameters=[{
            # Gimbal rotation limits
            'max_rotation_degrees': 80.0,
            # Control parameters
            'deadzone_pixels': 25,
            'p_gain': 0.05,
            'max_step_ticks': 200,
            # Servo configuration
            'yaw_servo_id': 1,
            'pitch_servo_id': 2,
            'yaw_zero_position': 1617,
            'pitch_zero_position': 1757,
            'pitch_up_degrees': 25.0,
            # Hardware
            'device_port': '/dev/ttyUSB0',
            'baudrate': 57600,
        }],
    )
    
    # Livox Mid360 LiDAR (matches official msg_MID360_launch.py format)
    livox_lidar_node = Node(
        package='livox_ros_driver2',
        executable='livox_ros_driver2_node',
        name='livox_lidar_publisher',
        output='screen',
        parameters=[
            {'xfer_format': 0},  # 0-Pointcloud2(PointXYZRTL), 1-customized pointcloud format
            {'multi_topic': 0},  # 0-All LiDARs share the same topic, 1-One LiDAR one topic
            {'data_src': 0},     # 0-lidar, others-Invalid data src
            {'publish_freq': 15.0},  # frequency of publish, 5.0, 10.0, 20.0, 50.0, etc.
            {'output_data_type': 0},
            {'frame_id': 'livox_frame'},
            {'lvx_file_path': '/home/livox/livox_test.lvx'},
            {'user_config_path': PathJoinSubstitution([
                FindPackageShare('unitree_ros2_poisson_simple'),
                'config',
                'MID360_config.json'
            ])},
            {'cmdline_input_bd_code': 'livox0000000001'}
        ]
    )
    
    # Static transform: livox_frame to body_link
    # Matches the transform in cloud_merger.h lidar_callback:
    # Translation: (0.05, 0.0, -0.18) - livox is 5cm forward, 18cm below body center
    # Rotation: 180° flip (R_flip matrix) - livox is mounted upside down
    # The R_flip matrix flips X and Z axes = 180° roll rotation
    # Quaternion for 180° roll: [1, 0, 0, 0] (x, y, z, w) but tf2 uses (w, x, y, z) order
    # Static TF: odom -> camera_init (identity - same world frame)
    # FAST_LIO uses 'camera_init' as world frame, we call it 'odom'
    odom_to_camera_init_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='odom_to_camera_init_tf',
        arguments=[
            '0', '0', '0',  # identity translation
            '0', '0', '0',  # identity rotation
            'odom', 'camera_init'
        ],
    )
    
    # Static TF: body -> livox_frame (identity - same lidar frame)
    # FAST_LIO uses 'body' as lidar frame, we call it 'livox_frame'
    body_to_livox_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='body_to_livox_tf',
        arguments=[
            '0', '0', '0',  # identity translation
            '0', '0', '0',  # identity rotation
            'body', 'livox_frame'
        ],
    )
    
    # Static TF: livox_frame -> body_link
    # From cloud_merger.h: lidar is at (0.05, 0, -0.18) from body, with 180° flip
    # So body is at (-0.05, 0, 0.18) from lidar in lidar coords, then apply inverse rotation
    # R_flip = diag(-1, 1, -1) means 180° rotation around Y axis
    livox_to_body_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='livox_to_body_tf',
        arguments=[
             '-0.05', '0.0', '0.18',  # x, y, z translation (body relative to livox)
             '0', '3.14159', '0',     # roll, pitch, yaw (180° pitch for Y-axis flip)
             'livox_frame', 'body_link'  # parent_frame, child_frame
        ],

        # arguments=[
        #     '-0.05', '0.0', '0.18',  # x, y, z translation (body relative to livox)
        #     '0', '3.14159', '0',     # roll, pitch, yaw (180° pitch for Y-axis flip)
        #     'livox_frame', 'body_link'  # parent_frame, child_frame
        # ],
    )
    
    # Static TF: body_link -> utlidar_link (front UTLidar on Go2)
    # The Go2 front lidar is approximately 37cm forward from body center
    # UTLidar publishes deskewed clouds already in odom frame, but we need the frame for reference
    body_to_utlidar_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='body_to_utlidar_tf',
        arguments=[
            '0.37', '0.0', '0.05',  # x, y, z (front lidar position relative to body)
            '0', '2.9', '0',    # roll, pitch, yaw (150 degrees pitch)
            'body_link', 'utlidar_lidar'
        ],
    )
    
    # FAST_LIO Mapping Node (LiDAR-Inertial Odometry)
    # We only enable odometry output, disabling rviz, map, and path publishing
    
    fast_lio_node = Node(
        package='fast_lio',
        executable='fastlio_mapping',
        name='fastlio_mapping',
        output='screen',
        parameters=[{
            # Core config file parameters (mid360.yaml)
            'feature_extract_enable': False,
            'point_filter_num': 3,
            'max_iteration': 3,
            'filter_size_surf': 0.5,
            'filter_size_map': 0.5,
            'cube_side_length': 1000.0,
            'runtime_pos_log_enable': False,
            # Common
            'common.lid_topic': '/livox/lidar',
            'common.imu_topic': '/livox/imu',
            'common.time_sync_en': False,
            'common.time_offset_lidar_to_imu': 0.0,
            # Preprocess - lidar_type 0 for standard PointCloud2 (xfer_format=0)
            # FAST_LIO enum: AVIA=1, VELO16=2, OUST64=3, MID360=4
            # Use 0 (or >4) to trigger default_handler for generic PointXYZI
            'preprocess.lidar_type': 0,
            'preprocess.scan_line': 4,
            'preprocess.blind': 0.5,
            'preprocess.timestamp_unit': 3,
            'preprocess.scan_rate': 10,
            # Mapping
            'mapping.acc_cov': 0.1,
            'mapping.gyr_cov': 0.1,
            'mapping.b_acc_cov': 0.0001,
            'mapping.b_gyr_cov': 0.0001,
            'mapping.fov_degree': 360.0,
            'mapping.det_range': 100.0,
            'mapping.extrinsic_est_en': True,
            # Use identity extrinsics - we handle lidar->body transform via static TF
            'mapping.extrinsic_T': [0.0, 0.0, 0.0],
            'mapping.extrinsic_R': [1., 0., 0., 0., 1., 0., 0., 0., 1.],
            # Publish - disable everything except odometry
            'publish.path_en': False,
            'publish.effect_map_en': False,
            'publish.map_en': False,
            'publish.scan_publish_en': False,  # Disable scan output
            'publish.dense_publish_en': False,
            'publish.scan_bodyframe_pub_en': False,
            # PCD save - disable
            'pcd_save.pcd_save_en': False,
        }],
    )
    
    # Odometry publisher - subscribes to FAST_LIO /Odometry, publishes /odom and TF (odom -> body_link)
    # Uses FAST_LIO LiDAR-Inertial odometry instead of Unitree sportmodestate or T265
    odom_publisher_node = Node(
        package='unitree_ros2_poisson_simple',
        executable='odom_publisher_fastlio',
        name='odom_publisher',
        output='screen',
    )

    
    return LaunchDescription([
        camera_type_arg,
        camera_model_arg,
        camera_fps_arg,
        yolo_model_arg,
        use_tensorrt_arg,
        point_cloud_density_arg,
        dh0_human_arg,
        dh0_obstacle_arg,
        min_z_arg,
        max_z_arg,
        tight_area_human_threshold_arg,
        tight_area_h_threshold_arg,
        tight_area_wall_slack_arg,
        human_track_timeout_arg,
        human_track_gate_radius_arg,
        human_track_velocity_decay_tau_arg,
        velocity_threshold_arg,
        min_yolo_cells_arg,
        enable_human_tracker_dilation_arg,
        decay_in_fov_arg,
        decay_stationary_arg,
        decay_unconfirmed_arg,
        no_retrack_on_move_arg,
        enable_social_navigation_arg,
        social_tangent_layers_arg,
        social_tangent_bias_arg,
        social_layer_thickness_arg,
        human_direction_threshold_arg,
        robot_mos_human_arg,
        robot_mos_obstacle_arg,
        cbf_sigma_epsilon_arg,
        cbf_sigma_kappa_arg,
        vel_max_x_fwd_arg,
        vel_max_x_bwd_arg,
        vel_max_y_arg,
        vel_max_yaw_arg,
        logging_publish_hz_arg,
        enable_data_logging_to_file_arg,
        realsense_launch,
        # zed_launch,
        # cloud_merger_node,
        livox_lidar_node,
        odom_to_camera_init_tf,   # Connect odom to FAST_LIO's camera_init
        body_to_livox_tf,         # Connect FAST_LIO's body to livox_frame
        livox_to_body_tf,
        body_to_utlidar_tf,
        fast_lio_node,  # FAST_LIO provides /Odometry from LiDAR-inertial fusion
        odom_publisher_node,  # Subscribes to FAST_LIO /Odometry, publishes /odom and TF
        yolo_detector_node,
        human_tracking_node,  # Provides odom -> camera_link TF
        semantic_poisson_node,  # Start last - needs TF from odom_publisher and human_tracking
    ])
