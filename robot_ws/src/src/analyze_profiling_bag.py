"""
STEP-BY-STEP WORKFLOW FOR GENERATING PROFILING RESULTS
============================================================

1. SOURCE ROS 2
---------------
Before doing anything, source ROS 2 environment in the terminal.

2. RUN NODE
-----------------------
Launch the node that publishes profiling data on:

    /poisson/profiling_data

Make sure the code is actively running and the profiling publisher is enabled.

You can verify the topic exists with:
    ros2 topic list | grep profiling

And check that messages are coming through with:
    ros2 topic echo /poisson/profiling_data

3. RECORD THE BAG
-----------------
In a NEW terminal, source ROS 2 again, then record the profiling topic:

    ros2 bag record /poisson/profiling_data

Give the bag a cleaner name:
    ros2 bag record /poisson/profiling_data -o profiling_run1

Let it record while the robot/system is running your experiment.

4. STOP RECORDING
-----------------
When the experiment is finished, stop the bag recording with:

    Ctrl + C

This will create a rosbag folder, for example:
    profiling_run1/

Inside it there will be metadata and recorded message data.

5. CHECK THE BAG
----------------
Before analyzing, make sure the bag actually contains the profiling topic:

    ros2 bag info profiling_run1

You should see:
    /poisson/profiling_data

6. RUN THIS ANALYSIS SCRIPT
---------------------------
Once the bag exists, run:

    python analyze_profiling_bag.py --bag thesis_profiling_run1 --out profiling_results

This script will:
    - read the bag
    - extract all profiling samples
    - save raw data to CSV
    - compute summary statistics
    - generate plots
    - write a short report

7. LOOK AT THE OUTPUT FILES
---------------------------
The output folder will contain files like:

    profiling_raw.csv
    profiling_summary.csv
    profiling_summary.md
    total_latency_timeseries.png
    total_latency_histogram.png
    total_latency_cdf.png
    stage_breakdown_bar.png
    stage_percent_bar.png
    stage_timeseries.png
    stacked_mean_pipeline.png
    pipeline_sum_vs_end_to_end.png
    unattributed_latency.png
    report.txt

The most important ones for the thesis are:
    - profiling_summary.csv
    - stage_breakdown_bar.png
    - total_latency_timeseries.png
    - total_latency_cdf.png
    - report.txt

"""

#!/usr/bin/env python3
"""
Extract and analyze ROS 2 profiling data from /poisson/profiling_data.

Expected message type:
    std_msgs/msg/Float32MultiArray

Expected field order from the ROS node:
    0  occupancy_preprocess_ms
    1  semantic_fusion_ms
    2  geometry_shaping_ms
    3  guidance_boundary_setup_ms
    4  guidance_social_expansion_ms
    5  guidance_laplace_ms
    6  guidance_copyout_ms
    7  safety_field_solve_ms
    8  dhdt_update_ms
    9  predictive_control_ms
    10 realtime_filter_ms
    11 command_dispatch_ms
    12 field_data_age_ms
    13 end_to_end_grid_ms

Usage:
    python analyze_profiling_bag.py --bag /path/to/rosbag2_2026_04_17-14_32_10
    python analyze_profiling_bag.py --bag /path/to/bag --topic /poisson/profiling_data --out results

Requirements:
    pip install pandas numpy matplotlib
    ROS 2 environment sourced so rosbag2_py is available
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py


FIELD_NAMES: List[str] = [
    "occupancy_preprocess_ms",
    "semantic_fusion_ms",
    "geometry_shaping_ms",
    "guidance_boundary_setup_ms",
    "guidance_social_expansion_ms",
    "guidance_laplace_ms",
    "guidance_copyout_ms",
    "safety_field_solve_ms",
    "dhdt_update_ms",
    "predictive_control_ms",
    "realtime_filter_ms",
    "command_dispatch_ms",
    "field_data_age_ms",
    "end_to_end_grid_ms",
]

PIPELINE_STAGE_COLUMNS: List[str] = [
    "occupancy_preprocess_ms",
    "semantic_fusion_ms",
    "geometry_shaping_ms",
    "guidance_boundary_setup_ms",
    "guidance_social_expansion_ms",
    "guidance_laplace_ms",
    "guidance_copyout_ms",
    "safety_field_solve_ms",
    "dhdt_update_ms",
    "predictive_control_ms",
    "realtime_filter_ms",
    "command_dispatch_ms",
]

DISPLAY_NAMES = {
    "occupancy_preprocess_ms": "Occupancy preprocess",
    "semantic_fusion_ms": "Semantic fusion",
    "geometry_shaping_ms": "Geometry shaping",
    "guidance_boundary_setup_ms": "Guidance boundary",
    "guidance_social_expansion_ms": "Guidance social",
    "guidance_laplace_ms": "Guidance Laplace",
    "guidance_copyout_ms": "Guidance copyout",
    "safety_field_solve_ms": "Safety field solve",
    "dhdt_update_ms": "dh/dt update",
    "predictive_control_ms": "Predictive control",
    "realtime_filter_ms": "Realtime filter",
    "command_dispatch_ms": "Command dispatch",
    "field_data_age_ms": "Field data age",
    "end_to_end_grid_ms": "End-to-end grid",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze ROS 2 profiling bag data.")
    parser.add_argument("--bag", required=True, help="Path to rosbag2 directory.")
    parser.add_argument(
        "--topic",
        default="/poisson/profiling_data",
        help="Topic containing Float32MultiArray profiling samples.",
    )
    parser.add_argument(
        "--out",
        default="profiling_results",
        help="Output directory for CSVs and plots.",
    )
    parser.add_argument(
        "--control-period-ms",
        type=float,
        default=50.0,
        help="Reference control/update period for feasibility plots and stats.",
    )
    parser.add_argument(
        "--max-series-plots",
        type=int,
        default=8,
        help="Max number of per-stage time-series plots to overlay in one figure.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def discover_topic_type(bag_path: str, topic_name: str) -> str:
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    try:
        reader.open(storage_options, converter_options)
    except RuntimeError:
        # Fallback for non-sqlite storage such as mcap
        storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="mcap")
        reader.open(storage_options, converter_options)

    topics = reader.get_all_topics_and_types()
    topic_type_map = {topic.name: topic.type for topic in topics}

    if topic_name not in topic_type_map:
        available = "\n".join(sorted(topic_type_map.keys()))
        raise ValueError(
            f"Topic '{topic_name}' not found in bag.\nAvailable topics:\n{available}"
        )

    return topic_type_map[topic_name]


def open_reader(bag_path: str) -> rosbag2_py.SequentialReader:
    reader = rosbag2_py.SequentialReader()
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    # Try sqlite3 first, then mcap.
    for storage_id in ("sqlite3", "mcap"):
        try:
            storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id)
            reader.open(storage_options, converter_options)
            return reader
        except RuntimeError:
            continue

    raise RuntimeError(
        f"Could not open bag at '{bag_path}' with sqlite3 or mcap storage."
    )


def extract_topic_to_dataframe(bag_path: str, topic_name: str) -> pd.DataFrame:
    topic_type = discover_topic_type(bag_path, topic_name)
    msg_type = get_message(topic_type)
    reader = open_reader(bag_path)

    rows: List[List[float]] = []
    timestamps_ns: List[int] = []

    while reader.has_next():
        topic, raw_data, timestamp_ns = reader.read_next()
        if topic != topic_name:
            continue

        msg = deserialize_message(raw_data, msg_type)

        if not hasattr(msg, "data"):
            raise TypeError(
                f"Topic '{topic_name}' is type '{topic_type}', but has no 'data' field."
            )

        data = list(msg.data)
        if len(data) != len(FIELD_NAMES):
            raise ValueError(
                f"Expected {len(FIELD_NAMES)} profiling values, got {len(data)} "
                f"for topic '{topic_name}'."
            )

        rows.append(data)
        timestamps_ns.append(timestamp_ns)

    if not rows:
        raise ValueError(f"No messages found on topic '{topic_name}'.")

    df = pd.DataFrame(rows, columns=FIELD_NAMES)
    df.insert(0, "timestamp_ns", timestamps_ns)
    df["time_s"] = (df["timestamp_ns"] - df["timestamp_ns"].iloc[0]) * 1e-9
    df["sample_idx"] = np.arange(len(df))

    # Aggregate a "pipeline sum" from stage measurements
    df["pipeline_stage_sum_ms"] = df[PIPELINE_STAGE_COLUMNS].sum(axis=1)

    # Residual between explicit stage sum and reported end-to-end
    df["unattributed_ms"] = df["end_to_end_grid_ms"] - df["pipeline_stage_sum_ms"]

    return df


def make_summary_table(df: pd.DataFrame, control_period_ms: float) -> pd.DataFrame:
    summary_rows = []

    for col in FIELD_NAMES:
        series = df[col].dropna().astype(float)

        mean_ms = series.mean()
        std_ms = series.std(ddof=1) if len(series) > 1 else 0.0
        median_ms = series.median()
        p95_ms = series.quantile(0.95)
        p99_ms = series.quantile(0.99)
        max_ms = series.max()
        min_ms = series.min()

        percent_of_total = np.nan
        if col != "field_data_age_ms":
            total_mean = df["end_to_end_grid_ms"].mean()
            if total_mean > 0:
                percent_of_total = 100.0 * mean_ms / total_mean

        over_period_rate = 100.0 * (series > control_period_ms).mean()

        summary_rows.append(
            {
                "field": col,
                "display_name": DISPLAY_NAMES.get(col, col),
                "mean_ms": mean_ms,
                "std_ms": std_ms,
                "median_ms": median_ms,
                "p95_ms": p95_ms,
                "p99_ms": p99_ms,
                "min_ms": min_ms,
                "max_ms": max_ms,
                "percent_of_total_mean": percent_of_total,
                "exceeds_control_period_percent": over_period_rate,
            }
        )

    summary = pd.DataFrame(summary_rows)
    return summary


def save_markdown_table(summary: pd.DataFrame, path: Path) -> None:
    table_cols = [
        "display_name",
        "mean_ms",
        "std_ms",
        "p95_ms",
        "p99_ms",
        "max_ms",
        "percent_of_total_mean",
    ]
    md_df = summary[table_cols].copy()

    for col in md_df.columns[1:]:
        md_df[col] = md_df[col].map(
            lambda x: f"{x:.2f}" if pd.notna(x) and isinstance(x, (int, float, np.floating)) else ""
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write(md_df.to_markdown(index=False))


def plot_total_latency(df: pd.DataFrame, out_dir: Path, control_period_ms: float) -> None:
    plt.figure(figsize=(11, 5))
    plt.plot(df["sample_idx"], df["end_to_end_grid_ms"])
    plt.axhline(control_period_ms, linestyle="--")
    plt.xlabel("Sample index")
    plt.ylabel("Latency (ms)")
    plt.title("End-to-end grid latency over samples")
    plt.tight_layout()
    plt.savefig(out_dir / "total_latency_timeseries.png", dpi=300)
    plt.close()


def plot_total_latency_histogram(df: pd.DataFrame, out_dir: Path, control_period_ms: float) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(df["end_to_end_grid_ms"], bins=40)
    plt.axvline(control_period_ms, linestyle="--")
    plt.xlabel("End-to-end latency (ms)")
    plt.ylabel("Count")
    plt.title("Histogram of end-to-end grid latency")
    plt.tight_layout()
    plt.savefig(out_dir / "total_latency_histogram.png", dpi=300)
    plt.close()


def plot_total_latency_cdf(df: pd.DataFrame, out_dir: Path, control_period_ms: float) -> None:
    x = np.sort(df["end_to_end_grid_ms"].to_numpy())
    y = np.arange(1, len(x) + 1) / len(x)

    plt.figure(figsize=(8, 5))
    plt.plot(x, y)
    plt.axvline(control_period_ms, linestyle="--")
    plt.xlabel("End-to-end latency (ms)")
    plt.ylabel("Empirical CDF")
    plt.title("CDF of end-to-end grid latency")
    plt.tight_layout()
    plt.savefig(out_dir / "total_latency_cdf.png", dpi=300)
    plt.close()


def plot_stage_breakdown_bar(summary: pd.DataFrame, out_dir: Path) -> None:
    stage_summary = summary[summary["field"].isin(PIPELINE_STAGE_COLUMNS)].copy()

    plt.figure(figsize=(12, 6))
    plt.bar(stage_summary["display_name"], stage_summary["mean_ms"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean latency (ms)")
    plt.title("Mean latency by stage")
    plt.tight_layout()
    plt.savefig(out_dir / "stage_breakdown_bar.png", dpi=300)
    plt.close()


def plot_stage_percent_bar(summary: pd.DataFrame, out_dir: Path) -> None:
    stage_summary = summary[summary["field"].isin(PIPELINE_STAGE_COLUMNS)].copy()

    plt.figure(figsize=(12, 6))
    plt.bar(stage_summary["display_name"], stage_summary["percent_of_total_mean"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Percent of mean end-to-end latency (%)")
    plt.title("Relative contribution of each stage")
    plt.tight_layout()
    plt.savefig(out_dir / "stage_percent_bar.png", dpi=300)
    plt.close()


def plot_stage_timeseries(df: pd.DataFrame, out_dir: Path, max_series_plots: int) -> None:
    # Show the heaviest stages first based on mean latency
    mean_order = (
        df[PIPELINE_STAGE_COLUMNS].mean().sort_values(ascending=False).index.tolist()
    )
    selected = mean_order[:max_series_plots]

    plt.figure(figsize=(12, 6))
    for col in selected:
        plt.plot(df["sample_idx"], df[col], label=DISPLAY_NAMES.get(col, col))
    plt.xlabel("Sample index")
    plt.ylabel("Latency (ms)")
    plt.title("Per-stage latency over samples")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "stage_timeseries.png", dpi=300)
    plt.close()


def plot_stacked_mean_pipeline(summary: pd.DataFrame, out_dir: Path) -> None:
    stage_summary = summary[summary["field"].isin(PIPELINE_STAGE_COLUMNS)].copy()

    values = stage_summary["mean_ms"].to_numpy()
    labels = stage_summary["display_name"].tolist()

    plt.figure(figsize=(10, 2.8))
    left = 0.0
    for value, label in zip(values, labels):
        plt.barh(["Mean pipeline"], [value], left=left, label=label)
        left += value

    plt.xlabel("Latency (ms)")
    plt.title("Stacked mean pipeline latency composition")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_dir / "stacked_mean_pipeline.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_pipeline_vs_reported(df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(11, 5))
    plt.plot(df["sample_idx"], df["pipeline_stage_sum_ms"], label="Sum of stage timings")
    plt.plot(df["sample_idx"], df["end_to_end_grid_ms"], label="Reported end-to-end grid")
    plt.xlabel("Sample index")
    plt.ylabel("Latency (ms)")
    plt.title("Stage-sum latency vs reported end-to-end latency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "pipeline_sum_vs_end_to_end.png", dpi=300)
    plt.close()


def plot_unattributed_latency(df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(11, 4.5))
    plt.plot(df["sample_idx"], df["unattributed_ms"])
    plt.xlabel("Sample index")
    plt.ylabel("Latency residual (ms)")
    plt.title("Residual: end-to-end minus summed stage timings")
    plt.tight_layout()
    plt.savefig(out_dir / "unattributed_latency.png", dpi=300)
    plt.close()


def write_text_report(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    out_dir: Path,
    control_period_ms: float,
) -> None:
    total = df["end_to_end_grid_ms"]
    worst_stage = (
        summary[summary["field"].isin(PIPELINE_STAGE_COLUMNS)]
        .sort_values("mean_ms", ascending=False)
        .iloc[0]
    )

    exceed_rate = 100.0 * (total > control_period_ms).mean()

    lines = [
        "Profiling analysis report",
        "========================",
        "",
        f"Number of profiling samples: {len(df)}",
        f"Mean end-to-end latency: {total.mean():.3f} ms",
        f"Median end-to-end latency: {total.median():.3f} ms",
        f"95th percentile end-to-end latency: {total.quantile(0.95):.3f} ms",
        f"99th percentile end-to-end latency: {total.quantile(0.99):.3f} ms",
        f"Max end-to-end latency: {total.max():.3f} ms",
        f"Reference control period: {control_period_ms:.3f} ms",
        f"Samples exceeding control period: {exceed_rate:.2f} %",
        "",
        f"Dominant stage by mean latency: {worst_stage['display_name']}",
        f"Dominant stage mean latency: {worst_stage['mean_ms']:.3f} ms",
        f"Dominant stage share of mean end-to-end latency: {worst_stage['percent_of_total_mean']:.2f} %",
        "",
        f"Mean summed stage latency: {df['pipeline_stage_sum_ms'].mean():.3f} ms",
        f"Mean reported end-to-end latency: {df['end_to_end_grid_ms'].mean():.3f} ms",
        f"Mean residual (end-to-end - stage sum): {df['unattributed_ms'].mean():.3f} ms",
    ]

    with open(out_dir / "report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    args = parse_args()

    bag_path = Path(args.bag)
    out_dir = Path(args.out)

    if not bag_path.exists():
        raise FileNotFoundError(f"Bag path does not exist: {bag_path}")

    ensure_dir(out_dir)

    print(f"[1/5] Reading bag from: {bag_path}")
    df = extract_topic_to_dataframe(str(bag_path), args.topic)

    print(f"[2/5] Saving raw CSV")
    df.to_csv(out_dir / "profiling_raw.csv", index=False)

    print(f"[3/5] Computing summary statistics")
    summary = make_summary_table(df, args.control_period_ms)
    summary.to_csv(out_dir / "profiling_summary.csv", index=False)
    save_markdown_table(summary, out_dir / "profiling_summary.md")

    print(f"[4/5] Generating plots")
    plot_total_latency(df, out_dir, args.control_period_ms)
    plot_total_latency_histogram(df, out_dir, args.control_period_ms)
    plot_total_latency_cdf(df, out_dir, args.control_period_ms)
    plot_stage_breakdown_bar(summary, out_dir)
    plot_stage_percent_bar(summary, out_dir)
    plot_stage_timeseries(df, out_dir, args.max_series_plots)
    plot_stacked_mean_pipeline(summary, out_dir)
    plot_pipeline_vs_reported(df, out_dir)
    plot_unattributed_latency(df, out_dir)

    print(f"[5/5] Writing report")
    write_text_report(df, summary, out_dir, args.control_period_ms)

    print("\nDone.")
    print(f"Results saved to: {out_dir.resolve()}")
    print("Generated files:")
    for p in sorted(out_dir.iterdir()):
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()

"""
