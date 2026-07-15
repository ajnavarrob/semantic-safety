from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    depthai_launch = PathJoinSubstitution([
        FindPackageShare("depthai_ros_driver"),
        "launch",
        "camera.launch.py",
    ])

    config_file = PathJoinSubstitution([
        FindPackageShare("oak_semantic_volume_poc"),
        "config",
        "oak1_segmentation.yaml",
    ])

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(depthai_launch),
            launch_arguments={
                "name": "oak_left",
                "parent_frame": "body_link",

                # Temporary hardcoded OAK-1 mount transform.
                "cam_pos_x": "-0.05",
                "cam_pos_y": "0.1",
                "cam_pos_z": "0.25",

                "cam_roll": "0.0",
                "cam_pitch": "0.0",
                "cam_yaw": "1.5708",  # 90 degrees in radians

                "params_file": config_file,
                "use_rviz": "false",
            }.items(),
        )
    ])