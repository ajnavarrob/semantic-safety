from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg = get_package_share_directory('oak_semantic_volume_poc')
    params = os.path.join(pkg, 'config', 'poc.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=params),
        Node(package='oak_semantic_volume_poc', executable='semantic_volume_node',
             name='semantic_volume_node', output='screen',
             parameters=[LaunchConfiguration('params_file')]),
    ])
