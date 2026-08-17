#!/usr/bin/env python3

"""
Rule-aware semantic map fuser — Commit 4 - Added per class persistence

    1. Subscribe to front and rear per-class semantic observation grids.
    2. Maintain short-lived persistence on per-class semantic observations.
    3. Load the runtime constraints JSON used by semantic_poisson.
    4. Compile enforced exclusion/avoidance rules by semantic class.
    5. Expand only the semantic class affected by each rule.
    6. Publish raw and immediately expanded per-class layers.

This commit intentionally does NOT perform slow insertion/removal.
"""

import copy
import json
import math
import os
import re
from functools import partial
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

import rclpy
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import UInt64
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy


class SemanticMapFuser(Node):
    SEMANTIC_CLASSES: Tuple[str, ...] = (
        'human',
        'traffic_cone',
        'caution_tape',
        'floor_danger_tape',
        'wet_floor_sign',
        'spill',
    )

    TARGET_ALIASES: Dict[str, str] = {
        'human': 'human', 'humans': 'human', 'person': 'human',
        'persons': 'human', 'people': 'human',
        'pedestrian': 'human', 'pedestrians': 'human',
        'traffic_cone': 'traffic_cone', 'traffic_cones': 'traffic_cone',
        'cone': 'traffic_cone', 'cones': 'traffic_cone',
        'caution_tape': 'caution_tape', 'caution_tapes': 'caution_tape',
        'tape': 'caution_tape',
        'floor_danger_tape': 'floor_danger_tape',
        'floor_danger_tapes': 'floor_danger_tape',
        'danger_tape': 'floor_danger_tape',
        'danger_tape_on_floor': 'floor_danger_tape',
        'tape_on_floor': 'floor_danger_tape',
        'floor_tape': 'floor_danger_tape',
        'wet_floor_sign': 'wet_floor_sign',
        'wet_floor_signs': 'wet_floor_sign',
        'wet_sign': 'wet_floor_sign',
        'spill': 'spill', 'spills': 'spill',
        'liquid_spill': 'spill', 'liquid_spills': 'spill',
    }

    SUPPORTED_RULE_TYPES: Set[str] = {'exclusion', 'avoidance'}

    LEGACY_CLASS_VALUES: Dict[str, int] = {
        'human': 1,
        'traffic_cone': 2,
        'caution_tape': 3,
        'floor_danger_tape': 4,
        'wet_floor_sign': 5,
        'spill': 6,
    }

    LEGACY_CLASS_PRIORITY: Tuple[str, ...] = (
        'spill',
        'floor_danger_tape',
        'caution_tape',
        'traffic_cone',
        'wet_floor_sign',
        'human',
    )

    def __init__(self) -> None:
        super().__init__('semantic_map_fuser')

        self._declare_parameters()
        self._load_parameters()

        self.grid_info = None
        self.grid_cell_count = 0

        self.last_seen_ns: Dict[str, Optional[np.ndarray]] = {
            class_name: None for class_name in self.SEMANTIC_CLASSES
        }

        self.front_visibility_map: Optional[OccupancyGrid] = None
        self.rear_visibility_map: Optional[OccupancyGrid] = None

        self.class_buffer_m: Dict[str, float] = {
            class_name: 0.0 for class_name in self.SEMANTIC_CLASSES
        }
        self.class_rule_ids: Dict[str, List[str]] = {
            class_name: [] for class_name in self.SEMANTIC_CLASSES
        }

        self.constraints_signature: Optional[str] = None
        self.constraints_revision: int = 0
        self.constraints_last_mtime_ns: Optional[int] = None
        self.warned_keys: Set[str] = set()
        self.last_log_time_ns = 0

        self._create_subscriptions()
        self._create_publishers()

        self.publish_timer = self.create_timer(
            1.0 / self.publish_rate_hz,
            self.publish_outputs
        )

        if self.constraints_path:
            self.constraints_timer = self.create_timer(
                1.0 / self.constraints_reload_hz,
                self.reload_constraints_if_changed
            )
            self.reload_constraints_if_changed(force=True)
        else:
            self.constraints_timer = None
            self.get_logger().warn(
                'constraints_path is empty; semantic safety targets will remain empty'
            )

        self._log_configuration()

    def _declare_parameters(self) -> None:
        self.declare_parameter('semantic_observation_prefix', '/semantic_observations')
        self.declare_parameter('semantic_layer_prefix', '/semantic_layers')
        self.declare_parameter('semantic_safety_target_prefix', '/semantic_safety_targets')
        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('log_rate_hz', 1.0)
        self.declare_parameter('occupied_threshold', 50)
        self.declare_parameter('output_frame', 'body_link')

        self.declare_parameter('human_timeout_sec', 0.0)
        self.declare_parameter('traffic_cone_timeout_sec', 3.0)
        self.declare_parameter('caution_tape_timeout_sec', 5.0)
        self.declare_parameter('floor_danger_tape_timeout_sec', 5.0)
        self.declare_parameter('wet_floor_sign_timeout_sec', 3.0)
        self.declare_parameter('spill_timeout_sec', 5.0)

        self.declare_parameter('constraints_path', '')
        self.declare_parameter('constraints_reload_hz', 0.1)
        self.declare_parameter('max_buffer_distance_m', 5.0)
        self.declare_parameter('empty_target_without_rule', True)

        self.declare_parameter('front_visibility_topic', '/visibility_map_front')
        self.declare_parameter('rear_visibility_topic', '/visibility_map_rear')
        self.declare_parameter('fused_visibility_topic', '/visibility_map')

        self.declare_parameter('publish_legacy_outputs', True)
        self.declare_parameter('legacy_class_map_topic', '/class_map')

        self.declare_parameter('publish_combined_safety_target', True)
        self.declare_parameter('combined_safety_target_topic', '/semantic_safety_target')
        self.declare_parameter('combined_safety_target_revision_topic', '/semantic_safety_target_revision')

    def _load_parameters(self) -> None:
        self.semantic_observation_prefix = self._normalize_prefix(
            self.get_parameter('semantic_observation_prefix').value
        )
        self.semantic_layer_prefix = self._normalize_prefix(
            self.get_parameter('semantic_layer_prefix').value
        )
        self.semantic_safety_target_prefix = self._normalize_prefix(
            self.get_parameter('semantic_safety_target_prefix').value
        )

        self.publish_rate_hz = max(0.1, float(self.get_parameter('publish_rate_hz').value))
        self.log_rate_hz = max(0.0, float(self.get_parameter('log_rate_hz').value))
        self.occupied_threshold = int(np.clip(
            int(self.get_parameter('occupied_threshold').value), 1, 100
        ))
        self.output_frame = str(self.get_parameter('output_frame').value)

        # These durations control per-cell persistence of semantic observations.
        # A currently observed cell refreshes its timestamp; an absent detection
        # does not immediately clear it. Instead, the cell remains active until
        # its class-specific persistence duration expires.
        self.class_persistence_sec: Dict[str, float] = {}
        self.class_persistence_ns: Dict[str, int] = {}
        for class_name in self.SEMANTIC_CLASSES:
            persistence_sec = max(0.0, float(
                self.get_parameter(f'{class_name}_timeout_sec').value
            ))
            self.class_persistence_sec[class_name] = persistence_sec
            self.class_persistence_ns[class_name] = int(persistence_sec * 1e9)

        self.constraints_path = os.path.expanduser(
            str(self.get_parameter('constraints_path').value).strip()
        )
        self.constraints_reload_hz = max(
            0.1,
            float(self.get_parameter('constraints_reload_hz').value)
        )
        self.max_buffer_distance_m = max(
            0.0,
            float(self.get_parameter('max_buffer_distance_m').value)
        )
        self.empty_target_without_rule = bool(
            self.get_parameter('empty_target_without_rule').value
        )

        self.front_visibility_topic = str(
            self.get_parameter('front_visibility_topic').value
        )
        self.rear_visibility_topic = str(
            self.get_parameter('rear_visibility_topic').value
        )
        self.fused_visibility_topic = str(
            self.get_parameter('fused_visibility_topic').value
        )

        self.publish_legacy_outputs = bool(
            self.get_parameter('publish_legacy_outputs').value
        )
        self.legacy_class_map_topic = str(
            self.get_parameter('legacy_class_map_topic').value
        )
        self.publish_combined_safety_target = bool(
            self.get_parameter('publish_combined_safety_target').value
        )
        self.combined_safety_target_topic = str(
            self.get_parameter('combined_safety_target_topic').value
        )
        self.combined_safety_target_revision_topic = str(
            self.get_parameter('combined_safety_target_revision_topic').value
        )

    @staticmethod
    def _normalize_prefix(prefix: str) -> str:
        prefix = str(prefix).strip()
        if not prefix:
            return ''
        if not prefix.startswith('/'):
            prefix = '/' + prefix
        return prefix.rstrip('/')

    def _create_subscriptions(self) -> None:
        self.semantic_subscriptions = []
        for camera_name in ('front', 'rear'):
            for class_name in self.SEMANTIC_CLASSES:
                topic = f'{self.semantic_observation_prefix}/{camera_name}/{class_name}'
                subscription = self.create_subscription(
                    OccupancyGrid,
                    topic,
                    partial(self.semantic_observation_callback, camera_name, class_name),
                    10
                )
                self.semantic_subscriptions.append(subscription)

        self.front_visibility_subscription = self.create_subscription(
            OccupancyGrid,
            self.front_visibility_topic,
            self.front_visibility_callback,
            10
        )
        self.rear_visibility_subscription = self.create_subscription(
            OccupancyGrid,
            self.rear_visibility_topic,
            self.rear_visibility_callback,
            10
        )

    def _create_publishers(self) -> None:
        self.semantic_layer_publishers = {}
        self.semantic_target_publishers = {}

        for class_name in self.SEMANTIC_CLASSES:
            self.semantic_layer_publishers[class_name] = self.create_publisher(
                OccupancyGrid,
                f'{self.semantic_layer_prefix}/{class_name}',
                10
            )
            self.semantic_target_publishers[class_name] = self.create_publisher(
                OccupancyGrid,
                f'{self.semantic_safety_target_prefix}/{class_name}',
                10
            )

        self.visibility_map_publisher = self.create_publisher(
            OccupancyGrid,
            self.fused_visibility_topic,
            10
        )

        self.legacy_class_map_publisher = None
        if self.publish_legacy_outputs:
            self.legacy_class_map_publisher = self.create_publisher(
                OccupancyGrid,
                self.legacy_class_map_topic,
                10
            )

        self.combined_target_publisher = None
        self.combined_target_revision_publisher = None

        if self.publish_combined_safety_target:
            self.combined_target_publisher = self.create_publisher(
                OccupancyGrid,
                self.combined_safety_target_topic,
                10
            )

            revision_qos = QoSProfile(depth=1)
            revision_qos.reliability = ReliabilityPolicy.RELIABLE
            revision_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

            self.combined_target_revision_publisher = self.create_publisher(
                UInt64,
                self.combined_safety_target_revision_topic,
                revision_qos
            )

    def semantic_observation_callback(
        self,
        camera_name: str,
        class_name: str,
        msg: OccupancyGrid
    ) -> None:
        source_name = f'{camera_name} {class_name}'

        if not self._validate_message_data(msg, source_name):
            return

        if self.grid_info is None:
            self._initialize_grid(msg)
        elif not self._grid_matches_reference(msg):
            self._warn_once(
                f'geometry:{source_name}',
                f'Ignoring {source_name}: grid geometry does not match the reference grid'
            )
            return

        observed_mask = (
            np.asarray(msg.data, dtype=np.int16)
            >= self.occupied_threshold
        )

        # Persistence is attached to the semantic observation itself:
        # only currently observed cells refresh their timestamp. Empty cells do
        # not immediately erase prior detections; they expire in
        # _active_raw_masks() according to the class-specific duration.
        if not np.any(observed_mask):
            return

        now_ns = self.get_clock().now().nanoseconds
        self.last_seen_ns[class_name][observed_mask] = now_ns

    def _initialize_grid(self, msg: OccupancyGrid) -> None:
        self.grid_info = copy.deepcopy(msg.info)
        self.grid_cell_count = int(msg.info.width) * int(msg.info.height)

        for class_name in self.SEMANTIC_CLASSES:
            self.last_seen_ns[class_name] = np.full(
                self.grid_cell_count,
                -1,
                dtype=np.int64
            )

        self.get_logger().info(
            f'Initialized semantic grid: {msg.info.width}x{msg.info.height}, '
            f'resolution={msg.info.resolution:.4f} m, '
            f'frame={msg.header.frame_id or self.output_frame}'
        )

    def _active_raw_masks(self, now_ns: int) -> Dict[str, np.ndarray]:
        masks: Dict[str, np.ndarray] = {}
        for class_name in self.SEMANTIC_CLASSES:
            last_seen = self.last_seen_ns[class_name]
            persistence_ns = self.class_persistence_ns[class_name]

            if last_seen is None:
                masks[class_name] = np.zeros(self.grid_cell_count, dtype=bool)
                continue

            # Zero persistence means "current-frame only" behavior. This is
            # intentionally used for humans because human persistence/tracking
            # is handled separately in semantic_poisson.
            max_age_ns = (
                persistence_ns
                if persistence_ns > 0
                else int(1e9 / self.publish_rate_hz)
            )

            masks[class_name] = (
                (last_seen >= 0)
                & ((now_ns - last_seen) <= max_age_ns)
            )

        return masks

    @staticmethod
    def _constraint_revision_from_text(text: str) -> int:
        # 64-bit FNV-1a; semantic_poisson.cpp uses the same function.
        value = 1469598103934665603
        for byte in text.encode('utf-8'):
            value ^= byte
            value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        return value

    def reload_constraints_if_changed(self, force: bool = False) -> None:
        if not self.constraints_path:
            return

        try:
            mtime_ns = os.stat(self.constraints_path).st_mtime_ns
        except FileNotFoundError:
            self._warn_once(
                'constraints_missing',
                f'Constraints file does not exist: {self.constraints_path}'
            )
            return
        except OSError as exc:
            self.get_logger().warn(f'Could not stat constraints file: {exc}')
            return

        if not force and self.constraints_last_mtime_ns == mtime_ns:
            return

        try:
            with open(self.constraints_path, 'r', encoding='utf-8') as handle:
                raw_constraints_text = handle.read()
            document = json.loads(raw_constraints_text)
        except (OSError, json.JSONDecodeError) as exc:
            self.get_logger().warn(f'Failed to load constraints JSON: {exc}')
            return

        loaded_revision = self._constraint_revision_from_text(
            raw_constraints_text
        )

        compiled_buffers, compiled_rule_ids = self._compile_constraint_document(document)
        signature = json.dumps(
            {'buffers': compiled_buffers, 'rule_ids': compiled_rule_ids},
            sort_keys=True
        )

        self.constraints_last_mtime_ns = mtime_ns

        revision_changed = loaded_revision != self.constraints_revision
        self.constraints_revision = loaded_revision

        if signature == self.constraints_signature:
            if revision_changed:
                self.get_logger().info(
                    f'Constraint revision acknowledged: {self.constraints_revision}'
                )
            return

        old_buffers = dict(self.class_buffer_m)
        self.class_buffer_m = compiled_buffers
        self.class_rule_ids = compiled_rule_ids
        self.constraints_signature = signature

        changed_classes = [
            class_name
            for class_name in self.SEMANTIC_CLASSES
            if not math.isclose(
                old_buffers[class_name],
                self.class_buffer_m[class_name],
                abs_tol=1e-6
            )
        ]

        if changed_classes:
            for class_name in changed_classes:
                self.get_logger().info(
                    f'Semantic rule update: class={class_name}, '
                    f'buffer={old_buffers[class_name]:.3f} -> '
                    f'{self.class_buffer_m[class_name]:.3f} m, '
                    f'rules={self.class_rule_ids[class_name]}'
                )
        else:
            self.get_logger().info(
                'Constraints reloaded; effective semantic buffers unchanged'
            )

    def _compile_constraint_document(
        self,
        document: dict
    ) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
        buffers = {class_name: 0.0 for class_name in self.SEMANTIC_CLASSES}
        rule_ids = {class_name: [] for class_name in self.SEMANTIC_CLASSES}

        constraints = document.get('constraints', [])
        if not isinstance(constraints, list):
            self.get_logger().warn("Constraints JSON field 'constraints' is not an array")
            return buffers, rule_ids

        parsed_count = 0
        supported_count = 0

        for raw_constraint in constraints:
            parsed_count += 1
            if not isinstance(raw_constraint, dict):
                continue

            enabled = bool(raw_constraint.get('enabled', True))
            enforce = bool(raw_constraint.get('enforce', False))
            if not enabled or not enforce:
                continue

            raw_type = str(raw_constraint.get('type', '')).strip().lower()
            if raw_type not in self.SUPPORTED_RULE_TYPES:
                continue

            buffer_distance_m = self._extract_buffer_distance(raw_constraint)
            if buffer_distance_m <= 0.0:
                continue

            buffer_distance_m = min(buffer_distance_m, self.max_buffer_distance_m)
            targets = self._extract_target_classes(raw_constraint)

            if not targets:
                self.get_logger().warn(
                    f"Constraint '{raw_constraint.get('id', 'unnamed')}' "
                    'has no semantic target'
                )
                continue

            constraint_id = str(raw_constraint.get('id', 'unnamed_constraint'))
            applied = False

            for raw_target in targets:
                canonical_class = self._canonical_semantic_class(raw_target)
                if canonical_class is None:
                    self._warn_once(
                        f'unsupported_target:{raw_target}',
                        f"Ignoring unsupported semantic target '{raw_target}'"
                    )
                    continue

                previous_buffer = buffers[canonical_class]
                if buffer_distance_m > previous_buffer:
                    buffers[canonical_class] = buffer_distance_m
                    rule_ids[canonical_class] = [constraint_id]
                elif math.isclose(buffer_distance_m, previous_buffer, abs_tol=1e-6):
                    if constraint_id not in rule_ids[canonical_class]:
                        rule_ids[canonical_class].append(constraint_id)

                applied = True

            if applied:
                supported_count += 1

        self.get_logger().info(
            f'Compiled semantic rules: {supported_count} supported enforced rules '
            f'out of {parsed_count} parsed constraints'
        )

        return buffers, rule_ids

    @staticmethod
    def _extract_buffer_distance(constraint: dict) -> float:
        spatial_parameters = constraint.get('spatial_parameters', {})
        if not isinstance(spatial_parameters, dict):
            return -1.0

        # For exclusion rules, buffer_distance_m is authoritative. The other
        # fields are accepted as compatibility fallbacks.
        for key in ('buffer_distance_m', 'min_distance_m', 'radius_m'):
            value = spatial_parameters.get(key, -1.0)
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue

            if math.isfinite(numeric_value) and numeric_value > 0.0:
                return numeric_value

        return -1.0

    @staticmethod
    def _extract_target_classes(constraint: dict) -> List[str]:
        targets: List[str] = []

        target_object = constraint.get('target')
        if isinstance(target_object, dict):
            semantic_class = target_object.get('semantic_class')
            if isinstance(semantic_class, str):
                targets.append(semantic_class)
            elif isinstance(semantic_class, list):
                targets.extend(item for item in semantic_class if isinstance(item, str))

        objects = constraint.get('objects')
        if isinstance(objects, dict):
            legacy_target = objects.get('target')
            if isinstance(legacy_target, str):
                targets.append(legacy_target)
            elif isinstance(legacy_target, list):
                targets.extend(item for item in legacy_target if isinstance(item, str))

        return targets

    @classmethod
    def _canonical_semantic_class(cls, raw_value: str) -> Optional[str]:
        normalized = str(raw_value).strip().lower().replace('-', '_')
        normalized = re.sub(r'\s+', '_', normalized)
        normalized = re.sub(r'_+', '_', normalized)
        return cls.TARGET_ALIASES.get(normalized)

    def _build_target_masks(
        self,
        raw_masks: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        target_masks: Dict[str, np.ndarray] = {}
        height = int(self.grid_info.height)
        width = int(self.grid_info.width)
        resolution = float(self.grid_info.resolution)

        for class_name in self.SEMANTIC_CLASSES:
            raw_mask = raw_masks[class_name]
            buffer_m = self.class_buffer_m[class_name]

            if buffer_m <= 0.0:
                target_masks[class_name] = (
                    np.zeros_like(raw_mask, dtype=bool)
                    if self.empty_target_without_rule
                    else raw_mask.copy()
                )
                continue

            radius_cells = int(math.ceil(buffer_m / resolution))
            if radius_cells <= 0:
                target_masks[class_name] = raw_mask.copy()
                continue

            image = raw_mask.reshape(height, width).astype(np.uint8) * 255
            kernel_size = 2 * radius_cells + 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (kernel_size, kernel_size)
            )
            dilated = cv2.dilate(image, kernel, iterations=1)
            target_masks[class_name] = dilated.reshape(-1) > 0

        return target_masks

    def front_visibility_callback(self, msg: OccupancyGrid) -> None:
        if self._validate_message_data(msg, 'front visibility'):
            self.front_visibility_map = msg

    def rear_visibility_callback(self, msg: OccupancyGrid) -> None:
        if self._validate_message_data(msg, 'rear visibility'):
            self.rear_visibility_map = msg

    def _fuse_visibility_maps(self) -> Optional[OccupancyGrid]:
        front = self.front_visibility_map
        rear = self.rear_visibility_map

        if front is None and rear is None:
            return None
        if front is None:
            return self._copy_grid(rear)
        if rear is None:
            return self._copy_grid(front)

        if not self._grids_match_each_other(front, rear):
            self._warn_once(
                'visibility_geometry_mismatch',
                'Front and rear visibility grids do not match; using front visibility only'
            )
            return self._copy_grid(front)

        front_data = np.asarray(front.data, dtype=np.int16)
        rear_data = np.asarray(rear.data, dtype=np.int16)
        front_known = np.where(front_data < 0, 0, front_data)
        rear_known = np.where(rear_data < 0, 0, rear_data)

        fused = OccupancyGrid()
        fused.header = copy.deepcopy(front.header)
        fused.info = copy.deepcopy(front.info)
        fused.data = np.maximum(front_known, rear_known).astype(np.int8).tolist()
        return fused

    def publish_outputs(self) -> None:
        now = self.get_clock().now()
        stamp = now.to_msg()
        self._publish_visibility(stamp)

        if self.grid_info is None:
            return

        raw_masks = self._active_raw_masks(now.nanoseconds)
        target_masks = self._build_target_masks(raw_masks)

        for class_name in self.SEMANTIC_CLASSES:
            self.semantic_layer_publishers[class_name].publish(
                self._make_binary_grid(raw_masks[class_name], stamp)
            )
            self.semantic_target_publishers[class_name].publish(
                self._make_binary_grid(target_masks[class_name], stamp)
            )

        if self.publish_legacy_outputs:
            # Keep /class_map raw. semantic_poisson still performs its old
            # human-specific expansion, so publishing expanded cells here
            # would double-inflate them before Commit 5.
            self.legacy_class_map_publisher.publish(
                self._make_legacy_class_grid(raw_masks, stamp)
            )

        if self.publish_combined_safety_target:
            combined_mask = np.zeros(self.grid_cell_count, dtype=bool)
            for class_name in self.SEMANTIC_CLASSES:
                combined_mask |= target_masks[class_name]

            # Publish the revision first. semantic_poisson ignores target maps
            # until this exact revision has been acknowledged.
            if self.combined_target_revision_publisher is not None:
                revision_msg = UInt64()
                revision_msg.data = int(self.constraints_revision)
                self.combined_target_revision_publisher.publish(revision_msg)

            self.combined_target_publisher.publish(
                self._make_binary_grid(combined_mask, stamp)
            )

        self._maybe_log_counts(raw_masks, target_masks, now.nanoseconds)

    def _make_binary_grid(self, mask: np.ndarray, stamp) -> OccupancyGrid:
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = self.output_frame
        msg.info = copy.deepcopy(self.grid_info)

        data = np.zeros(self.grid_cell_count, dtype=np.int8)
        data[mask] = 100
        msg.data = data.tolist()
        return msg

    def _make_legacy_class_grid(
        self,
        raw_masks: Dict[str, np.ndarray],
        stamp
    ) -> OccupancyGrid:
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = self.output_frame
        msg.info = copy.deepcopy(self.grid_info)

        class_data = np.zeros(self.grid_cell_count, dtype=np.int8)
        for class_name in self.LEGACY_CLASS_PRIORITY:
            class_data[raw_masks[class_name]] = self.LEGACY_CLASS_VALUES[class_name]

        msg.data = class_data.tolist()
        return msg

    def _publish_visibility(self, stamp) -> None:
        fused_visibility = self._fuse_visibility_maps()
        if fused_visibility is None:
            return

        fused_visibility.header.stamp = stamp
        fused_visibility.header.frame_id = self.output_frame
        self.visibility_map_publisher.publish(fused_visibility)

    def _validate_message_data(self, msg: OccupancyGrid, source_name: str) -> bool:
        expected_cells = int(msg.info.width) * int(msg.info.height)
        if expected_cells <= 0:
            self._warn_once(
                f'invalid_dimensions:{source_name}',
                f'{source_name} grid has invalid dimensions'
            )
            return False

        if len(msg.data) != expected_cells:
            self._warn_once(
                f'invalid_data_length:{source_name}',
                f'{source_name} data length is {len(msg.data)}, expected {expected_cells}'
            )
            return False

        return True

    def _grid_matches_reference(self, msg: OccupancyGrid) -> bool:
        if self.grid_info is None:
            return True
        return self._grid_info_matches(self.grid_info, msg.info)

    @classmethod
    def _grid_info_matches(cls, info_a, info_b) -> bool:
        if info_a.width != info_b.width or info_a.height != info_b.height:
            return False
        if not cls._close(info_a.resolution, info_b.resolution):
            return False

        origin_a = info_a.origin
        origin_b = info_b.origin
        values_a = (
            origin_a.position.x, origin_a.position.y, origin_a.position.z,
            origin_a.orientation.x, origin_a.orientation.y,
            origin_a.orientation.z, origin_a.orientation.w,
        )
        values_b = (
            origin_b.position.x, origin_b.position.y, origin_b.position.z,
            origin_b.orientation.x, origin_b.orientation.y,
            origin_b.orientation.z, origin_b.orientation.w,
        )
        return all(cls._close(a, b) for a, b in zip(values_a, values_b))

    @classmethod
    def _grids_match_each_other(
        cls,
        map_a: OccupancyGrid,
        map_b: OccupancyGrid
    ) -> bool:
        return (
            len(map_a.data) == len(map_b.data)
            and cls._grid_info_matches(map_a.info, map_b.info)
        )

    @staticmethod
    def _close(value_a: float, value_b: float, tolerance: float = 1e-6) -> bool:
        return abs(float(value_a) - float(value_b)) <= tolerance

    @staticmethod
    def _copy_grid(msg: OccupancyGrid) -> OccupancyGrid:
        copied = OccupancyGrid()
        copied.header = copy.deepcopy(msg.header)
        copied.info = copy.deepcopy(msg.info)
        copied.data = list(msg.data)
        return copied

    def _warn_once(self, key: str, message: str) -> None:
        if key in self.warned_keys:
            return
        self.warned_keys.add(key)
        self.get_logger().warn(message)

    def _maybe_log_counts(
        self,
        raw_masks: Dict[str, np.ndarray],
        target_masks: Dict[str, np.ndarray],
        now_ns: int
    ) -> None:
        if self.log_rate_hz <= 0.0:
            return

        period_ns = int(1e9 / self.log_rate_hz)
        if now_ns - self.last_log_time_ns < period_ns:
            return

        self.last_log_time_ns = now_ns
        fields = []
        for class_name in self.SEMANTIC_CLASSES:
            raw_count = int(np.count_nonzero(raw_masks[class_name]))
            target_count = int(np.count_nonzero(target_masks[class_name]))
            buffer_m = self.class_buffer_m[class_name]
            persistence_sec = self.class_persistence_sec[class_name]
            fields.append(
                f'{class_name}={raw_count}->{target_count}'
                f'@{buffer_m:.2f}m/persist={persistence_sec:.2f}s'
            )

        self.get_logger().info('Semantic layers | ' + ', '.join(fields))

    def _log_configuration(self) -> None:
        lines = [
            'Rule-aware semantic map fuser initialized',
            f'Observation prefix: {self.semantic_observation_prefix}',
            f'Raw layer prefix: {self.semantic_layer_prefix}',
            f'Safety target prefix: {self.semantic_safety_target_prefix}',
            f'Constraints path: {self.constraints_path or "<empty>"}',
            f'Constraints reload rate: {self.constraints_reload_hz:.2f} Hz',
            f'Publish rate: {self.publish_rate_hz:.2f} Hz',
            f'Output frame: {self.output_frame}',
            'Semantic observation persistence: '
            + ', '.join(
                f'{class_name}={self.class_persistence_sec[class_name]:.2f}s'
                for class_name in self.SEMANTIC_CLASSES
            ),
            f'Empty target without enforced rule: {self.empty_target_without_rule}',
        ]
        self.get_logger().info('\n'.join(lines))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SemanticMapFuser()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
