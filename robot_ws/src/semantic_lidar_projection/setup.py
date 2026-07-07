import os
from glob import glob
from setuptools import setup

package_name = "semantic_lidar_projection"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="unitree",
    maintainer_email="unitree@example.com",
    description="Project LiDAR points into OAK camera images and transfer semantic labels.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "project_lidar_debug_node = semantic_lidar_projection.project_lidar_debug_node:main",
        ],
    },
)
