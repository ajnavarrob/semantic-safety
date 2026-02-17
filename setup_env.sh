#!/bin/bash
# setup_env.sh - Setup complete development environment
# Automatically detects shell type and sources appropriate files

# Detect shell type and get script directory
# Must handle being sourced from any directory
# Use SETUP_ENV_DIR instead of SCRIPT_DIR to avoid collision with source_all.sh
if [ -n "$ZSH_VERSION" ]; then
    # For zsh: ${(%):-%x} gives the path of the sourced script
    SETUP_ENV_DIR="${${(%):-%x}:A:h}"
    SETUP_EXT="setup.zsh"
    echo "Detected zsh shell"
elif [ -n "$BASH_VERSION" ]; then
    # For bash: BASH_SOURCE[0] gives the path of the sourced script
    SETUP_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SETUP_EXT="setup.bash"
    echo "Detected bash shell"
else
    # Fallback: try BASH_SOURCE or $0
    if [ -n "${BASH_SOURCE[0]}" ]; then
        SETUP_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    else
        SETUP_ENV_DIR="$(cd "$(dirname "$0")" && pwd)"
    fi
    SETUP_EXT="setup.bash"
    echo "Unknown shell, defaulting to bash"
fi

echo "Script directory: $SETUP_ENV_DIR"

# Activate conda environment
conda activate semantic-safety

# Source submodules (ROS2 base, unitree, realsense, livox)
source "$SETUP_ENV_DIR/submodules/source_all.sh"

# Source robot workspace
ROBOT_SETUP="$SETUP_ENV_DIR/robot_ws/install/$SETUP_EXT"
if [ -f "$ROBOT_SETUP" ]; then
    echo "Sourcing robot_ws ($SETUP_EXT)..."
    source "$ROBOT_SETUP"
else
    echo "⚠️  robot_ws not built yet. Run colcon build first."
fi

echo "✅ Environment setup complete"