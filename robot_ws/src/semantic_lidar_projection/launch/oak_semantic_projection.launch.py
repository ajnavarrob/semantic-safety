import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory("semantic_lidar_projection")
    config_file = os.path.join(pkg_share, "config", "oak_projection.yaml")

    static_tf_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, "launch", "static_oak_tfs.launch.py")
        )
    )

    projector_node = Node(
        package="semantic_lidar_projection",
        executable="project_lidar_debug_node",
        name="project_lidar_debug_node",
        output="screen",
        parameters=[config_file],
    )

    return LaunchDescription([
        static_tf_launch,
        projector_node,
    ])