#!/bin/bash
# source_all.sh - Source all installed ROS2 workspaces
# Automatically detects shell type (bash/zsh) and sources appropriate files

# Detect shell type and get script directory
if [ -n "$ZSH_VERSION" ]; then
    SCRIPT_DIR="${0:A:h}"
    SETUP_EXT="setup.zsh"
    echo "Detected zsh shell"
elif [ -n "$BASH_VERSION" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SETUP_EXT="setup.bash"
    echo "Detected bash shell"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SETUP_EXT="setup.bash"
    echo "Unknown shell, defaulting to bash"
fi

# Detect if running on Jetson
IS_JETSON=false
if [ -f /etc/nv_tegra_release ]; then
    IS_JETSON=true
    echo "🤖 Detected Jetson platform - skipping ROS2 base and unitree_ros2"
fi

echo "Sourcing ROS2 workspaces from $SCRIPT_DIR..."

# 1. Source ROS2 base installation - Skip on Jetson
if [ "$IS_JETSON" = false ]; then
    if [ -f "/opt/ros/humble/$SETUP_EXT" ]; then
        echo "Sourcing ROS2 Humble ($SETUP_EXT)..."
        source "/opt/ros/humble/$SETUP_EXT"
    else
        echo "⚠️  ROS2 Humble not found at /opt/ros/humble/$SETUP_EXT"
    fi
else
    echo "Skipping ROS2 base (Jetson)"
fi

# 2. Source unitree_ros2 workspace - Skip on Jetson
if [ "$IS_JETSON" = false ]; then
    UNITREE_SETUP="$SCRIPT_DIR/unitree_ros2/install/$SETUP_EXT"
    if [ -f "$UNITREE_SETUP" ]; then
        echo "Sourcing unitree_ros2 ($SETUP_EXT)..."
        source "$UNITREE_SETUP"
    else
        echo "⚠️  unitree_ros2 not built yet. Run install.sh first."
    fi
else
    echo "Skipping unitree_ros2 (Jetson)"
fi

# 3. Source RealSense workspace (always needed)
REALSENSE_SETUP="$SCRIPT_DIR/realsense_ros2/install/$SETUP_EXT"
if [ -f "$REALSENSE_SETUP" ]; then
    echo "Sourcing realsense_ros2 ($SETUP_EXT)..."
    source "$REALSENSE_SETUP"
else
    echo "⚠️  realsense_ros2 not built yet. Run install.sh first."
fi

# 4. Source Livox LiDAR workspace (always needed)
LIVOX_SETUP="$SCRIPT_DIR/lidar_ws/install/$SETUP_EXT"
if [ -f "$LIVOX_SETUP" ]; then
    echo "Sourcing livox_ros_driver2 ($SETUP_EXT)..."
    source "$LIVOX_SETUP"
else
    echo "⚠️  livox_ros_driver2 not built yet. Run install.sh first."
fi
# 5. Source FAST_LIO workspace (always needed)
FAST_LIO_SETUP="$SCRIPT_DIR/odom_ws/install/$SETUP_EXT"
if [ -f "$FAST_LIO_SETUP" ]; then
    echo "Sourcing FAST_LIO ($SETUP_EXT)..."
    source "$FAST_LIO_SETUP"
else
    echo "⚠️  FAST_LIO not built yet. Run install.sh first."
fi
echo "✅ Done sourcing workspaces"
