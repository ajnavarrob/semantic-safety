#!/usr/bin/env python3
from __future__ import annotations
"""
Live diagnostics window for /poisson/profiling_data.

Subscribes to std_msgs/msg/Float32MultiArray published by semantic_poisson and
shows rolling timing traces, current/median/max values, and deadline warnings.

ROS 2 Foxy-compatible.

Example:
    source /opt/ros/foxy/setup.bash
    python3 poisson_live_diagnostics.py

Optional:
    python3 poisson_live_diagnostics.py --ros-args \
        -p history_seconds:=30.0 \
        -p refresh_hz:=5.0 \
        -p grid_deadline_ms:=100.0 \
        -p field_age_limit_ms:=200.0
"""


import math
import threading
import time
from collections import deque
from typing import Deque, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


TIMING_NAMES: List[str] = [
    "occupancy_preprocess",
    "semantic_fusion",
    "geometry_shaping",
    "guidance_boundary_setup",
    "guidance_social_expansion",
    "guidance_laplace",
    "guidance_copyout",
    "safety_field_solve",
    "dhdt_update",
    "predictive_control",
    "realtime_filter",
    "command_dispatch",
    "field_data_age",
    "end_to_end_grid",
]

DISPLAY_LABELS: Dict[str, str] = {
    "occupancy_preprocess": "Occupancy preprocess",
    "semantic_fusion": "Semantic fusion",
    "geometry_shaping": "Geometry shaping",
    "guidance_boundary_setup": "Guidance boundary",
    "guidance_social_expansion": "Social expansion",
    "guidance_laplace": "Guidance Laplace",
    "guidance_copyout": "Guidance copyout",
    "safety_field_solve": "Safety-field solve",
    "dhdt_update": "dh/dt update",
    "predictive_control": "Predictive control",
    "realtime_filter": "Realtime filter",
    "command_dispatch": "Command dispatch",
    "field_data_age": "Field age",
    "end_to_end_grid": "End-to-end grid",
}


class PoissonLiveDiagnostics(Node):
    def __init__(self) -> None:
        super().__init__("poisson_live_diagnostics")

        self.declare_parameter("topic", "/poisson/profiling_data")
        self.declare_parameter("history_seconds", 30.0)
        self.declare_parameter("expected_publish_hz", 10.0)
        self.declare_parameter("refresh_hz", 5.0)
        self.declare_parameter("grid_deadline_ms", 100.0)
        self.declare_parameter("field_age_limit_ms", 200.0)
        self.declare_parameter("realtime_filter_limit_ms", 10.0)
        self.declare_parameter("mpc_limit_ms", 50.0)
        self.declare_parameter("stale_timeout_s", 1.0)

        topic = str(self.get_parameter("topic").value)
        history_seconds = float(self.get_parameter("history_seconds").value)
        expected_publish_hz = float(
            self.get_parameter("expected_publish_hz").value
        )

        capacity = max(100, int(math.ceil(
            history_seconds * max(expected_publish_hz, 1.0) * 2.0
        )))

        self.history_seconds = history_seconds
        self.refresh_hz = max(
            0.5, float(self.get_parameter("refresh_hz").value)
        )
        self.stale_timeout_s = max(
            0.1, float(self.get_parameter("stale_timeout_s").value)
        )

        self.thresholds = {
            "end_to_end_grid": float(
                self.get_parameter("grid_deadline_ms").value
            ),
            "field_data_age": float(
                self.get_parameter("field_age_limit_ms").value
            ),
            "realtime_filter": float(
                self.get_parameter("realtime_filter_limit_ms").value
            ),
            "predictive_control": float(
                self.get_parameter("mpc_limit_ms").value
            ),
        }

        self.lock = threading.Lock()
        self.times: Deque[float] = deque(maxlen=capacity)
        self.values: Dict[str, Deque[float]] = {
            name: deque(maxlen=capacity) for name in TIMING_NAMES
        }

        self.start_monotonic = time.monotonic()
        self.last_message_monotonic: float | None = None
        self.bad_message_count = 0

        self.subscription = self.create_subscription(
            Float32MultiArray,
            topic,
            self.profiling_callback,
            10,
        )

        self.get_logger().info(
            f"Listening to {topic}; retaining {history_seconds:.1f} s"
        )

    def profiling_callback(self, msg: Float32MultiArray) -> None:
        if len(msg.data) != len(TIMING_NAMES):
            self.bad_message_count += 1
            self.get_logger().warning(
                f"Expected {len(TIMING_NAMES)} profiling values, "
                f"received {len(msg.data)}"
            )
            return

        now = time.monotonic()
        relative_time = now - self.start_monotonic

        with self.lock:
            self.times.append(relative_time)
            for name, value in zip(TIMING_NAMES, msg.data):
                self.values[name].append(float(value))
            self.last_message_monotonic = now

            cutoff = relative_time - self.history_seconds
            while self.times and self.times[0] < cutoff:
                self.times.popleft()
                for series in self.values.values():
                    if series:
                        series.popleft()

    def snapshot(self) -> tuple[np.ndarray, Dict[str, np.ndarray], float | None]:
        with self.lock:
            t = np.asarray(self.times, dtype=float)
            values = {
                name: np.asarray(series, dtype=float)
                for name, series in self.values.items()
            }
            last_message = self.last_message_monotonic
        return t, values, last_message


class DiagnosticsWindow:
    def __init__(self, node: PoissonLiveDiagnostics) -> None:
        self.node = node

        plt.ion()
        self.figure = plt.figure(figsize=(15, 9))
        grid = self.figure.add_gridspec(
            2, 2, width_ratios=[2.2, 1.0], height_ratios=[1.0, 1.0]
        )

        self.ax_pipeline = self.figure.add_subplot(grid[0, 0])
        self.ax_control = self.figure.add_subplot(grid[1, 0])
        self.ax_summary = self.figure.add_subplot(grid[:, 1])
        self.ax_summary.axis("off")

        self.pipeline_names = [
            "occupancy_preprocess",
            "semantic_fusion",
            "geometry_shaping",
            "guidance_boundary_setup",
            "guidance_social_expansion",
            "guidance_laplace",
            "guidance_copyout",
            "safety_field_solve",
            "dhdt_update",
            "end_to_end_grid",
        ]
        self.control_names = [
            "predictive_control",
            "realtime_filter",
            "command_dispatch",
            "field_data_age",
        ]

        self.pipeline_lines = {
            name: self.ax_pipeline.plot([], [], label=DISPLAY_LABELS[name])[0]
            for name in self.pipeline_names
        }
        self.control_lines = {
            name: self.ax_control.plot([], [], label=DISPLAY_LABELS[name])[0]
            for name in self.control_names
        }

        self.ax_pipeline.set_title("Grid and field pipeline")
        self.ax_pipeline.set_ylabel("Time (ms)")
        self.ax_pipeline.grid(True, alpha=0.3)
        self.ax_pipeline.legend(
            loc="upper left", fontsize=8, ncol=2
        )

        self.ax_control.set_title("Control and field freshness")
        self.ax_control.set_xlabel("Elapsed time (s)")
        self.ax_control.set_ylabel("Time / age (ms)")
        self.ax_control.grid(True, alpha=0.3)
        self.ax_control.legend(
            loc="upper left", fontsize=8, ncol=2
        )

        for name, threshold in self.node.thresholds.items():
            axis = (
                self.ax_pipeline
                if name == "end_to_end_grid"
                else self.ax_control
            )
            axis.axhline(
                threshold,
                linestyle="--",
                linewidth=1.0,
                label=f"{DISPLAY_LABELS[name]} limit",
            )

        self.summary_text = self.ax_summary.text(
            0.0,
            1.0,
            "Waiting for /poisson/profiling_data...",
            va="top",
            ha="left",
            family="monospace",
            fontsize=9,
            transform=self.ax_summary.transAxes,
        )

        self.figure.canvas.manager.set_window_title(
            "Poisson Controller Diagnostics"
        )
        self.figure.tight_layout()

        self.last_draw = 0.0

    @staticmethod
    def finite_stats(values: np.ndarray) -> tuple[float, float, float]:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return math.nan, math.nan, math.nan
        return (
            float(finite[-1]),
            float(np.median(finite)),
            float(np.max(finite)),
        )

    def build_summary(
        self,
        values: Dict[str, np.ndarray],
        last_message: float | None,
    ) -> str:
        now = time.monotonic()
        stale = (
            last_message is None
            or (now - last_message) > self.node.stale_timeout_s
        )

        status = "STALE / NO DATA" if stale else "LIVE"
        rows = [
            f"STATUS: {status}",
            "",
            f"{'Metric':25s} {'Now':>8s} {'Median':>8s} {'Max':>8s}",
            "-" * 55,
        ]

        summary_order = [
            "end_to_end_grid",
            "field_data_age",
            "safety_field_solve",
            "guidance_laplace",
            "predictive_control",
            "realtime_filter",
            "command_dispatch",
        ]

        alarms: List[str] = []
        for name in summary_order:
            current, median, maximum = self.finite_stats(values[name])
            rows.append(
                f"{DISPLAY_LABELS[name]:25.25s} "
                f"{current:8.2f} {median:8.2f} {maximum:8.2f}"
            )

            threshold = self.node.thresholds.get(name)
            if (
                threshold is not None
                and np.isfinite(current)
                and current > threshold
            ):
                alarms.append(
                    f"{DISPLAY_LABELS[name]}: "
                    f"{current:.1f} > {threshold:.1f} ms"
                )

        rows.extend(["", "ACTIVE WARNINGS"])
        if stale:
            rows.append("- Profiling topic is stale")
        if alarms:
            rows.extend(f"- {alarm}" for alarm in alarms)
        elif not stale:
            rows.append("- None")

        rows.extend([
            "",
            "Interpretation",
            "- Now: newest sample",
            "- Median: rolling-window median",
            "- Max: rolling-window maximum",
        ])

        return "\n".join(rows)

    def update(self) -> None:
        now = time.monotonic()
        if now - self.last_draw < 1.0 / self.node.refresh_hz:
            return
        self.last_draw = now

        t, values, last_message = self.node.snapshot()

        if t.size:
            display_t = t - t[-1]

            for name, line in self.pipeline_lines.items():
                line.set_data(display_t, values[name])

            for name, line in self.control_lines.items():
                line.set_data(display_t, values[name])

            left = max(-self.node.history_seconds, float(display_t[0]))
            self.ax_pipeline.set_xlim(left, 0.0)
            self.ax_control.set_xlim(left, 0.0)

            self.ax_pipeline.relim()
            self.ax_pipeline.autoscale_view(scalex=False, scaley=True)
            self.ax_control.relim()
            self.ax_control.autoscale_view(scalex=False, scaley=True)

        self.summary_text.set_text(
            self.build_summary(values, last_message)
        )
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()


def main() -> None:
    rclpy.init()
    node = PoissonLiveDiagnostics()
    window = DiagnosticsWindow(node)

    executor_thread = threading.Thread(
        target=rclpy.spin,
        args=(node,),
        daemon=True,
    )
    executor_thread.start()

    try:
        while rclpy.ok() and plt.fignum_exists(window.figure.number):
            window.update()
            plt.pause(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        executor_thread.join(timeout=1.0)
        plt.close("all")


if __name__ == "__main__":
    main()