# OAK semantic viewing-volume POC — ROS 2 Foxy

This standalone package runs semantic segmentation inference on an OAK-1 Myriad X and intersects the segmentation-derived viewing volume with a LiDAR-derived voxel occupancy map. It does **not** project LiDAR points into the image.

## Architecture

1. OAK-1 captures RGB and runs the segmentation `.blob` on-device.
2. The host decodes the OAK output into a `mono16` class-ID image.
3. A C++ node voxelizes the latest LiDAR cloud once.
4. It erodes and coarsely samples the segmentation mask.
5. Each active sample becomes a camera ray in 3D.
6. 3D DDA traversal stops at the first occupied voxel.
7. The voxel receives the segmentation class and is published as semantic occupancy.

The geometry workload is bounded by `maximum_rays_per_frame`; it uses no KD-tree, clustering, image projection of LiDAR points, or region growing.

## Selected initial model

The default model contract is **DeepLabV3-MobileNetV2 trained on ADE20K**. ADE20K provides indoor scene classes, and MobileNetV2 is a realistic lightweight backbone for RVC2/Myriad X. The blob is not redistributed. The model/OAK Viewer choice can be changed later without touching `semantic_volume_node`.

## Build on Ubuntu 20.04 / ROS 2 Foxy

```bash
unzip oak_semantic_volume_poc.zip
cd oak_semantic_volume_poc
./scripts/install_foxy.sh
source ~/oak_semantic_volume_ws/install/setup.bash
```

Put a compatible `.blob` at the configured path or edit `config/poc.yaml`.

## Required TF

Provide transforms connecting:

```text
<LiDAR frame> -> target_frame
camera optical frame -> target_frame
```

The camera frame must use optical coordinates: +x right, +y down, +z forward.

## Run

```bash
ros2 launch oak_semantic_volume_poc poc.launch.py
```

Useful topics:

```text
/oak_front/segmentation/class_map     mono16 semantic classes
/oak_front/segmentation/overlay       visual validation
/semantic_volume/occupied_voxels      x,y,z,class_id first-hit voxels
/semantic_volume/first_hits           duplicate debug output
```

## Standalone geometry test

The geometry node can be tested with any publisher that provides `CameraInfo`, a `mono16` class map, and a cloud:

```bash
ros2 launch oak_semantic_volume_poc geometry_only.launch.py
```

## Latency controls

Start with:

```yaml
voxel_size_m: 0.10
ray_step_pixels: 6
mask_erosion_pixels: 2
maximum_rays_per_frame: 2500
max_range_m: 8.0
```

Reduce latency by increasing `ray_step_pixels`, increasing voxel size, decreasing maximum range, or reducing the ray cap. `semantic_volume_node` logs geometry time and ray/hit counts.

## Tests

```bash
cd ~/oak_semantic_volume_ws
./src/oak_semantic_volume_poc/scripts/run_tests.sh
```

## Important model-output note

Inference runs on the OAK. With raw logits, only `argmax` decoding runs on the Go2. For the final low-bandwidth configuration, export/compile the network with a final ArgMax class-map output and set `output_is_class_map: true`.
