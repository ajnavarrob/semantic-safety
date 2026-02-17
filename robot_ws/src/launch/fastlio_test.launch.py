#!/usr/bin/env python3
"""
Minimal launch file for testing FAST_LIO with Livox Mid360
"""

from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Livox Mid360 LiDAR driver
    livox_lidar_node = Node(
        package='livox_ros_driver2',
        executable='livox_ros_driver2_node',
        name='livox_lidar_publisher',
        output='screen',
        parameters=[
            {'xfer_format': 0},
            {'multi_topic': 0},
            {'data_src': 0},
            {'publish_freq': 10.0},
            {'output_data_type': 0},
            {'frame_id': 'livox_frame'},
            {'user_config_path': PathJoinSubstitution([
                FindPackageShare('unitree_ros2_poisson_simple'),
                'config',
                'MID360_config.json'
            ])},
        ]
    )
    
    # FAST_LIO node
    fast_lio_node = Node(
        package='fast_lio',
        executable='fastlio_mapping',
        name='fastlio_mapping',
        output='screen',
        parameters=[{
            'feature_extract_enable': False,
            'point_filter_num': 3,
            'max_iteration': 3,
            'filter_size_surf': 0.5,
            'filter_size_map': 0.5,
            'cube_side_length': 1000.0,
            'runtime_pos_log_enable': False,
            # Topics
            'common.lid_topic': '/livox/lidar',
            'common.imu_topic': '/livox/imu',
            'common.time_sync_en': False,
            # Preprocess for Mid360 with standard PointCloud2 (xfer_format=0)
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
            # Enable outputs for debugging
            'publish.path_en': True,
            'publish.scan_publish_en': True,
            'publish.map_en': False,
            'pcd_save.pcd_save_en': False,
        }],
    )
    
    # Static TF: livox_frame -> body_link
    # From cloud_merger.h: lidar is at (0.05, 0, -0.18) from body, 180° flip
    # So body is at (-0.05, 0, 0.18) from lidar, with inverse rotation
    livox_to_body_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='livox_to_body_tf',
        arguments=[
            '-0.05', '0.0', '0.18',  # x, y, z translation
            '0', '3.14159', '0',     # roll, pitch, yaw (180° pitch)
            'livox_frame', 'body_link'
        ],
    )
    
    # Odom republisher
    odom_publisher_node = Node(
        package='unitree_ros2_poisson_simple',
        executable='odom_publisher_fastlio',
        name='odom_publisher',
        output='screen',
    )
    
    return LaunchDescription([
        livox_lidar_node,
        livox_to_body_tf,
        fast_lio_node,
        odom_publisher_node,
    ])

