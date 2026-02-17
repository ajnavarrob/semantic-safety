# Semantic Safety Pipeline - Quick Start

## Quick Launch
```bash
cd /home/yangl/semantic-safety/robot_ws
source install/setup.zsh
ros2 launch unitree_ros2_poisson_simple semantic_safety.launch.py
```

---

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `camera_fps` | 15 | RealSense FPS (6, 15, 30, 60) |
| `dh0_human` | 3.0 | Safety gradient for humans |
| `dh0_obstacle` | 1.0 | Safety gradient for obstacles |
| `yolo_model` | `yolo11n-seg.pt` | YOLO model path |
| `use_tensorrt` | false | Enable TensorRT (auto on Jetson) |

**Example:**
```bash
ros2 launch unitree_ros2_poisson_simple semantic_safety.launch.py \
  dh0_human:=5.0 dh0_obstacle:=2.0 camera_fps:=30
```

---

## Grid Settings ([poisson.h](file:///home/yangl/semantic-safety/robot_ws/src/include/poisson/poisson.h))

| Parameter | Value | Description |
|-----------|-------|-------------|
| `IMAX` | 100 | Grid height (cells) |
| `JMAX` | 100 | Grid width (cells) |
| `DS` | 0.05 | Cell size (meters) |
| **Coverage** | 5m × 5m | Total area |

---

## Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/class_map` | OccupancyGrid | Human labels from YOLO (1=human) |
| `/occupancy_grid` | OccupancyGrid | Merged occupancy |
| `/human_tracking/centroid` | Point | Human position for gimbal |

---

## Build

```bash
# Submodules (first time only)
cd submodules && ./install.sh && source source_all.sh

# Main build
cd robot_ws && colcon build --symlink-install
source install/setup.zsh
```

---

## Verification

```bash
# Check nodes
ros2 node list
# Expected: /semantic_poisson, /yolo_detector, /human_tracking, /rs_d435

# Check class_map
ros2 topic hz /class_map

# Enable visualization: press SPACE 3 times
# Red overlay = detected humans
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Segfault on start | Rebuild: `colcon build --symlink-install` |
| No human detection | Lower `confidence_threshold` to 0.3 |
| TF lookup failed | Ensure `human_tracking.py` is running |
| Slow performance | Use TensorRT: `use_tensorrt:=true` |

---

## Performance (Jetson Orin)

- **Grid Loop**: ~50-70ms @ 15Hz
- **Poisson Solve**: ~25-30ms  
- **YOLO (TensorRT)**: ~15-25ms
