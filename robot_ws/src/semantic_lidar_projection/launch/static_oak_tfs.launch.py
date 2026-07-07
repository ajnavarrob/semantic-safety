from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="oak_front_static_tf",
            arguments=[
                "0.30", "0.12", "0.25",
                "0.0", "0.0", "0.0",
                "body_link",
                "oak_front_base_frame",
            ],
        ),

        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="oak_rear_static_tf",
            arguments=[
                "-0.25", "-0.12", "0.25",
                "3.14159", "0.0", "3.14159",
                "body_link",
                "oak_rear_base_frame",
            ],
        ),
    ])