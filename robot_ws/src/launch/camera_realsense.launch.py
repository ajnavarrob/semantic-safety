#!/usr/bin/env python3
"""
RealSense D435 Camera launch file
Launch RealSense D435 with topic remapping and frame transforms
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    fps_arg = DeclareLaunchArgument(
        'fps',
        default_value='30',
        description='Camera FPS (6, 15, 30, or 60)'
    )
    
    point_cloud_density_arg = DeclareLaunchArgument(
        'point_cloud_density',
        default_value='1.0',
        description='Pointcloud density ratio (1.0 = full organized cloud for YOLO)'
    )
    
    # RealSense D435 Node
    realsense_node = Node(
        package='realsense_ros2',
        executable='rs_d435_node',
        name='rs_d435',
        output='screen',
        parameters=[{
            'publish_depth': False,
            'publish_pointcloud': True,
            'is_color': True,
            'publish_image_raw_': True,
            'fps': LaunchConfiguration('fps'),
            'point_cloud_density': LaunchConfiguration('point_cloud_density'),
        }],
        remappings=[
            # Remap RealSense topics to generic camera topics
            ('rs_d435/image_raw', '/camera/image_rect_color'),
            ('rs_d435/point_cloud', '/camera/point_cloud/cloud_registered'),
        ]
    )

    # Static TF: camera_link_d435 to camera_link_d435_pcl
    # This aligns the pointcloud frame with the image frame
    depth_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_depth_optical_frame_publisher',
        arguments=['0', '0', '0', '-1.5707963', '0', '-1.5707963', 
                'camera_link', 'camera_link_d435_pcl']
    )
    # Static transform for color optical frame (RGB image)
    # 15mm offset along native Y-axis (becomes -Z in rotated frame)
    color_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_color_optical_frame_publisher',
        arguments=['0', '-0.015', '0', '0.0', '0.0', '0.0',
                'camera_link', 'camera_link_d435']
    )
    
    return LaunchDescription([
        fps_arg,
        point_cloud_density_arg,
        realsense_node,
        depth_tf_node,
        color_tf_node,
    ])
