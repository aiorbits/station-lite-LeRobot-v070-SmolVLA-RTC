#!/usr/bin/env python3
"""Analyze a recorded single-chunk right-joint trace without robot access."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from joint_trace import SESSION_SCHEMA, TRACE_SCHEMA


def motion_metrics(q):
    dq = np.diff(q, axis=0)
    length = np.linalg.norm(dq, axis=1)
    reversals = ((dq[:-1] * dq[1:] < 0)
                 & (np.abs(dq[:-1]) > 0.01)
                 & (np.abs(dq[1:]) > 0.01))
    return {"path_length_rad": float(length.sum()),
            "net_displacement_rad": float(np.linalg.norm(q[-1] - q[0])),
            "max_single_joint_step_rad": float(np.abs(dq).max()),
            "per_joint_reversals_gt_0p01rad": reversals.sum(axis=0).astype(int).tolist()}


def analyze(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("schema") != TRACE_SCHEMA or metadata.get("status") != "trace_complete":
        raise ValueError("trace is incomplete or has an unexpected schema")
    with (directory / "samples.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != metadata["sample_count"] or len(rows) < 3:
        raise ValueError("trace sample count mismatch or too few readings")
    start_ns = np.asarray([int(row["read_start_monotonic_ns"]) for row in rows], dtype=np.int64)
    finish_ns = np.asarray([int(row["read_end_monotonic_ns"]) for row in rows], dtype=np.int64)
    q = np.asarray([[float(row[f"q{j}"]) for j in range(6)] for row in rows])
    gripper = np.asarray([float(row["gripper_m"]) for row in rows])
    plan = np.load(directory / "right_joint_waypoints.npy")
    if (q.shape[1] != 6 or plan.ndim != 2 or plan.shape[1] != 6
            or not np.isfinite(q).all() or not np.isfinite(gripper).all()
            or not np.isfinite(plan).all() or np.any(finish_ns < start_ns)
            or np.any(np.diff(finish_ns) <= 0)):
        raise ValueError("invalid state or non-monotonic timestamps")
    interval_ms = np.diff(finish_ns) / 1e6
    read_latency_ms = (finish_ns - start_ns) / 1e6
    nearest = np.min(np.max(np.abs(q[:, None, :] - plan[None, :, :]), axis=2), axis=1)
    result = {"schema": TRACE_SCHEMA, "sample_count": len(rows),
              "planned_waypoint_count": int(len(plan)),
              "send_call_duration_ms": (metadata["send_return_monotonic_ns"]
                                        - metadata["send_begin_monotonic_ns"]) / 1e6,
              "sample_interval_median_ms": float(np.median(interval_ms)),
              "sample_interval_p95_ms": float(np.percentile(interval_ms, 95)),
              "read_latency_median_ms": float(np.median(read_latency_ms)),
              "trace_elapsed_s": float((finish_ns[-1] - finish_ns[0]) / 1e9),
              "nominal_duration_s": metadata["nominal_duration_s"],
              "gripper_range_m": [float(gripper.min()), float(gripper.max())],
              "planned_joint_targets": motion_metrics(plan),
              "measured_joint_states": motion_metrics(q),
              "nearest_any_waypoint_error_median_rad": float(np.median(nearest)),
              "nearest_any_waypoint_error_p95_rad": float(np.percentile(nearest, 95)),
              "interpretation_note": "nearest waypoint ignores time; it cannot prove time-aligned tracking. Controller-internal interpolation and timestamps unavailable."}
    (directory / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def analyze_session(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    if (metadata.get("schema") != SESSION_SCHEMA
            or metadata.get("status") != "trace_complete"):
        raise ValueError("trace session is incomplete or has an unexpected schema")
    with (directory / "samples.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != metadata["sample_count"] or len(rows) < 3:
        raise ValueError("session sample count mismatch or too few readings")
    finish_ns = np.asarray([int(row["read_end_monotonic_ns"]) for row in rows],
                           dtype=np.int64)
    q = np.asarray([[float(row[f"q{i}"]) for i in range(6)] for row in rows])
    gripper = np.asarray([float(row["gripper_m"]) for row in rows])
    if (not np.isfinite(q).all() or not np.isfinite(gripper).all()
            or np.any(np.diff(finish_ns) <= 0)):
        raise ValueError("invalid session state or non-monotonic timestamps")
    intervals_ms = np.diff(finish_ns) / 1e6
    session_result = {
        "schema": SESSION_SCHEMA,
        "sample_count": len(rows),
        "sent_chunks": metadata.get("sent_chunks", 0),
        "sample_interval_median_ms": float(np.median(intervals_ms)),
        "sample_interval_p95_ms": float(np.percentile(intervals_ms, 95)),
        "gripper_range_m": [float(gripper.min()), float(gripper.max())],
        "measured_joint_states": motion_metrics(q),
        "chunks": [],
        "interpretation_note": (
            "Controller command samples are unavailable in this session; "
            "planned waypoints and measured states cannot be compared at matching times."),
    }
    for chunk_dir in sorted((directory / "chunks").glob("chunk_*")):
        chunk_meta = json.loads((chunk_dir / "metadata.json").read_text())
        chunk = {"loop_index": chunk_meta["loop_index"],
                 "status": chunk_meta["status"],
                 "send_begin_monotonic_ns": chunk_meta.get("send_begin_monotonic_ns"),
                 "send_return_monotonic_ns": chunk_meta.get("send_return_monotonic_ns"),
                 "nominal_duration_s": chunk_meta.get("nominal_duration_s"),
                 "switch_delay_s": chunk_meta.get("switch_delay_s"),
                 "controller_sample_error": chunk_meta.get("command_sample_error")}
        plan_path = chunk_dir / "right_joint_waypoints.npy"
        if plan_path.exists():
            target = np.load(plan_path)
            chunk["planned_waypoints"] = motion_metrics(target)
            chunk["planned_waypoint_count"] = int(len(target))
        command_path = chunk_dir / "controller_waypoint_samples.npy"
        if command_path.exists():
            command = np.load(command_path)
            controller_q = command[:, 1:7]
            chunk["controller_sample_count"] = int(len(command))
            chunk["controller_t_rel_range_s"] = [
                float(command[:, 0].min()), float(command[:, 0].max())]
            chunk["controller_sampled_joint_path"] = motion_metrics(controller_q)
            session_result["interpretation_note"] = (
                "Controller samples use relative times; verify their epoch and "
                "prefix semantics before claiming time-aligned tracking error.")
        if chunk_meta.get("send_begin_monotonic_ns") is not None:
            begin_ns = int(chunk_meta["send_begin_monotonic_ns"])
            end_ns = begin_ns + int((chunk_meta["nominal_duration_s"]
                                     + chunk_meta["switch_delay_s"] + 0.1) * 1e9)
            mask = (finish_ns >= begin_ns) & (finish_ns <= end_ns)
            if np.count_nonzero(mask) >= 3:
                chunk["measured_samples_in_window"] = int(mask.sum())
                chunk["measured_motion_in_window"] = motion_metrics(q[mask])
        session_result["chunks"].append(chunk)
    (directory / "analysis.json").write_text(
        json.dumps(session_result, indent=2) + "\n")
    return session_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path)
    args = parser.parse_args()
    metadata_path = args.trace_dir / "metadata.json"
    if not metadata_path.exists():
        parser.error(f"missing trace metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    result = (analyze_session(args.trace_dir)
              if metadata.get("schema") == SESSION_SCHEMA
              else analyze(args.trace_dir))
    print(json.dumps(result, indent=2))
