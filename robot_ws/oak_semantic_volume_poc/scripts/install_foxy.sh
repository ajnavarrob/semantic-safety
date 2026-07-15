#!/usr/bin/env bash
set -euo pipefail
source /opt/ros/foxy/setup.bash
sudo apt update
sudo apt install -y python3-pip python3-opencv ros-foxy-pcl-conversions ros-foxy-pcl-ros ros-foxy-tf2-geometry-msgs
python3 -m pip install --user 'depthai>=2.22,<3'
WS=${WS:-$HOME/oak_semantic_volume_ws}
mkdir -p "$WS/src"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PKG_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
rm -rf "$WS/src/oak_semantic_volume_poc"
cp -a "$PKG_DIR" "$WS/src/oak_semantic_volume_poc"
cd "$WS"
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
printf '\nsource %s/install/setup.bash\n' "$WS"
