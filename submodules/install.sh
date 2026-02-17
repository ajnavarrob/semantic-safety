#!/bin/bash
# install_submodules.sh - Install osqp, osqp-eigen, unitree_ros2 from submodules

# Don't use set -e - allow script to continue even if some steps fail
# This is important for Jetson where some dependencies might not be available

echo "Connect to the internet and run this script to install osqp, osqp-eigen, unitree_ros2 from submodules"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMODULES_DIR="$SCRIPT_DIR"

# Detect if running on Jetson
IS_JETSON=false
if [ -f /etc/nv_tegra_release ]; then
    IS_JETSON=true
    echo "🤖 Detected Jetson platform - skipping OSQP, OsqpEigen, and unitree_ros2 builds"
fi

echo "Installing submodules from $SUBMODULES_DIR..."

# Update submodules (skip on Jetson since we don't need osqp/unitree_ros2)
if [ "$IS_JETSON" = false ]; then
    echo "Updating submodules..."
    git submodule update --init --recursive osqp osqp-eigen unitree_ros2 || true
else
    echo "Skipping osqp/unitree_ros2 submodule update on Jetson"
fi

# 1. Install OSQP (dependency for OsqpEigen) - Skip on Jetson
if [ "$IS_JETSON" = false ]; then
    echo "=== 1/3 Installing OSQP ==="
    cd osqp
    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    sudo make install
    sudo ldconfig
    echo "✓ OSQP installed"
    cd "$SUBMODULES_DIR"
else
    echo "=== 1/3 Skipping OSQP (Jetson) ==="
fi

# 2. Install OsqpEigen (depends on OSQP) - Skip on Jetson
if [ "$IS_JETSON" = false ]; then
    echo "=== 2/3 Installing OsqpEigen ==="
    cd "$SUBMODULES_DIR/osqp-eigen"
    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    sudo make install
    sudo ldconfig
    echo "✓ OsqpEigen installed"
    cd "$SUBMODULES_DIR"
else
    echo "=== 2/4 Skipping OsqpEigen (Jetson) ==="
fi

# 3. Install unitree_ros2 (ROS 2 package) - Skip on Jetson
if [ "$IS_JETSON" = false ]; then
    echo "=== 3/4 Installing unitree_ros2 ==="
    cd "$SUBMODULES_DIR/unitree_ros2"
    # Install system dependencies via rosdep
    rosdep install --from-paths . --ignore-src -r -y

    # Build the ROS 2 workspace
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
    echo "✓ unitree_ros2 built"
    cd "$SUBMODULES_DIR"
else
    echo "=== 3/4 Skipping unitree_ros2 (Jetson) ==="
fi

# 4. Install realsense_ros2 (ROS 2 package)
echo "=== 4/5 Installing realsense_ros2 ==="
cd "$SUBMODULES_DIR/realsense_ros2"
# Build the ROS 2 workspace
colcon build --symlink-install --cmake-args=-DCMAKE_BUILD_TYPE=Release
cd "$SUBMODULES_DIR"/..
echo "✓ realsense_ros2 built"

# 5. Install livox_ros_driver2 (ROS 2 package for Livox LiDAR)
echo "=== 5/5 Installing livox_ros_driver2 ==="
cd "$SUBMODULES_DIR/lidar_ws/src/livox_ros_driver2"
# Build the ROS 2 workspace
./build.sh ROS2
cd "$SUBMODULES_DIR"/..
echo "✓ livox_ros_driver2 built"
# 6. Install FAST_LIO (ROS 2 package for LiDAR odometry)
echo "=== 6/6 Installing FAST_LIO ==="
cd "$SUBMODULES_DIR/odom_ws/"
# Build only the FAST_LIO package
rosdep install --from-paths . --ignore-src -r -y
source "$SUBMODULES_DIR/lidar_ws/install/setup.bash"
colcon build --symlink-install
cd "$SUBMODULES_DIR"/..
echo "✓ FAST_LIO built"

echo "=== ✅ Installation complete! ==="
echo "source $SUBMODULES_DIR/source_all.sh"
