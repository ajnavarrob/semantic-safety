#!/usr/bin/env bash
set -euo pipefail
source /opt/ros/foxy/setup.bash
WS=${WS:-$HOME/oak_semantic_volume_ws}
cd "$WS"
colcon test --packages-select oak_semantic_volume_poc --event-handlers console_direct+
colcon test-result --verbose
python3 src/oak_semantic_volume_poc/scripts/benchmark_geometry.py --rays 2500 --runs 30
