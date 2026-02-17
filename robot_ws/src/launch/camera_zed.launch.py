#!/usr/bin/env python3
"""
ZED Camera launch file
Launches ZED camera with frame transform to camera_link
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    camera_model_arg = DeclareLaunchArgument(
        'camera_model',
        default_value='zed2i',
        description='ZED camera model (zed, zed2, zed2i, zedm, zedx, zedxm)'
    )
    
    # ZED Camera launch
    zed_wrapper_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('zed_wrapper'),
                'launch',
                'zed_camera.launch.py'
            ])
        ]),
        launch_arguments={
            'camera_model': LaunchConfiguration('camera_model'),
            'point_cloud_freq': '15.0',
            'depth_confidence': '50',
            'depth_texture_conf': '100',
            'point_cloud_density': '0.1',  # Reduce density for performance
        }.items(),
    )
    
    # Static TF: camera_link to zed_left_camera_optical_frame
    # ZED publishes in zed_left_camera_optical_frame by default
    # We map this to camera_link for consistency with RealSense
    zed_optical_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='zed_camera_optical_frame_publisher',
        # ZED Mini has 63mm baseline. Left eye is offset by half baseline (31.5mm) from center.
        # Arguments: x y z yaw pitch roll frame_id child_frame_id
        arguments=['0', '0.0315', '0', '0', '0', '0', 
                  'camera_link', 'zed_left_camera_optical_frame']
    )
    
    return LaunchDescription([
        camera_model_arg,
        zed_wrapper_launch,
        zed_optical_tf_node,
    ])
