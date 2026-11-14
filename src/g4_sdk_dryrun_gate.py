#!/usr/bin/env python3
"""Isolated CAN-free vendor SDK continuous IK and planning dry-run gate for G4.

Must be run inside `sudo unshare --net` (loopback `lo` only, no CAN interfaces) on a
workstation where the vendor SDK (`$STARTOUCH_SDK_PY_DIR`) is installed.
Validates conditioned EEF paths, continuous seeded IK, post-smoothing joint limits/steps,
and vendor `plan_joint_waypoints_with_gripper` timelines without publishing motion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
from pathlib import Path

import numpy as np

CANDIDATE_DIR = Path(__file__).resolve().parent
if str(CANDIDATE_DIR) not in sys.path:
    sys.path.insert(0, str(CANDIDATE_DIR))

from station_lite_smolvla_right7_guarded import (  # noqa: E402
    GRIPPER_MAX_M,
    JOINT_LOW,
    JOINT_UP,
    select_initial_prefix,
    smooth_joint_waypoints,
    validate_post_smoothing_waypoints,
)

LOW = np.asarray(JOINT_LOW, dtype=float)
UP = np.asarray(JOINT_UP, dtype=float)
BUNDLE_SCHEMA = "v070_g4_sdk_handoff_bundle_v1"
REPORT_SCHEMA = "v070_g4_sdk_dryrun_report_v1"


def enforce_loopback_only_net(interfaces=None):
    """Refuse execution if any non-loopback network or CAN interface is visible."""
    if interfaces is None:
        interfaces = [name for _, name in socket.if_nameindex()]
    non_lo = [n for n in interfaces if n != "lo"]
    if non_lo:
        raise SystemExit(
            f"Refusing: run inside sudo unshare --net; only lo may exist (saw {interfaces})"
        )
    return list(interfaces)


def _solve_ik_step(arm, pos, euler, seed):
    if hasattr(arm, "solve_ik"):
        return arm.solve_ik(list(pos), list(euler), list(seed))
    if hasattr(arm, "solve_ik_euler"):
        return arm.solve_ik_euler(list(pos), list(euler), list(seed))
    raise AttributeError("Arm object provides neither solve_ik nor solve_ik_euler")


class _PrefixArmAdapter:
    """Adapt startouch.ArmController (solve_ik) or SingleArm (solve_ik_euler) to select_initial_prefix."""

    def __init__(self, arm, start_q):
        self._arm = arm
        self._start_q = np.asarray(start_q, dtype=float).reshape(6)

    def get_joint_positions(self):
        return self._start_q.tolist()

    def solve_ik_euler(self, pos, euler, seed):
        return _solve_ik_step(self._arm, pos, euler, seed)


def validate_planned_timeline(planned, *, expected_duration_s=None, margin=0.02):
    """Validate vendor planner dense timeline samples and return kinematic summary."""
    arr = np.asarray(planned, dtype=float)
    if arr.ndim != 2 or len(arr) < 2 or arr.shape[1] < 15 or not np.isfinite(arr).all():
        raise ValueError(f"Invalid planned trajectory shape or non-finite values: {arr.shape}")
    dt = np.diff(arr[:, 0])
    if np.any(dt <= 0.0):
        raise ValueError("Planned trajectory timestamps are not strictly increasing")
    q = arr[:, 1:7]
    if np.any(q < LOW) or np.any(q > UP):
        raise ValueError("Planned trajectory samples exceed physical joint limits")
    soft_outside = int(np.sum(np.any((q < LOW + margin) | (q > UP - margin), axis=1)))
    duration_s = float(arr[-1, 0] - arr[0, 0])
    q_speed = np.diff(q, axis=0) / dt[:, None]
    q_accel = (
        np.diff(q_speed, axis=0) / ((dt[:-1] + dt[1:])[:, None] / 2.0)
        if len(dt) > 1
        else np.zeros((0, 6))
    )
    nominal_hz = float(1.0 / np.median(dt)) if np.median(dt) > 0 else None
    duration_error_s = (
        abs(duration_s - float(expected_duration_s))
        if expected_duration_s is not None
        else None
    )
    return {
        "samples": int(len(arr)),
        "columns": int(arr.shape[1]),
        "duration_s": duration_s,
        "expected_duration_s": None if expected_duration_s is None else float(expected_duration_s),
        "duration_error_s": duration_error_s,
        "nominal_hz": nominal_hz,
        "min_dt_s": float(np.min(dt)),
        "max_dt_s": float(np.max(dt)),
        "samples_outside_soft_margin": soft_outside,
        "max_joint_speed_rad_s": float(np.max(np.abs(q_speed))) if q_speed.size else 0.0,
        "max_joint_accel_rad_s2": float(np.max(np.abs(q_accel))) if q_accel.size else 0.0,
    }


def evaluate_case_with_arm(arm, case, *, record_trajectories=True):
    """Validate one G4 case: continuous IK -> post-smoothing checks -> vendor dry-run planner."""
    start_q = np.asarray(case["seed"], dtype=float).reshape(-1)
    start_action = np.asarray(case["start_action"], dtype=float).reshape(-1)
    margin = float(case.get("margin_rad", 0.02))
    loop_index = int(case.get("loop_index", 1))
    omitted_prefix = None
    inserted_prefix = None
    if case.get("requires_initial_prefix_selection") and case.get("initial_prefix_candidates") is not None:
        prefix_cand = np.asarray(case["initial_prefix_candidates"], dtype=float)
        kept_prefix, omitted_prefix = select_initial_prefix(
            _PrefixArmAdapter(arm, start_q), prefix_cand, margin
        )
        inserted_prefix = int(len(kept_prefix))
        raw_src = np.asarray(case["raw_actions"], dtype=float)
        actions = np.concatenate([kept_prefix, raw_src], axis=0)[:50]
    else:
        actions = np.asarray(case["actions"], dtype=float)
    do_smooth = bool(case.get("smooth_waypoints", True))
    smooth_alpha = float(case.get("waypoint_smooth_alpha", 0.20))
    smooth_passes = int(case.get("waypoint_smooth_passes", 1))
    duration_s = float(case.get("duration_s", len(actions) / 30.0 / 0.2))
    step_limit_rad = float(case.get("max_interior_joint_step_rad", 0.25 if loop_index == 0 else 0.07))

    if start_q.shape != (6,) or not np.isfinite(start_q).all():
        raise ValueError(f"Invalid start_q for case {case.get('name')}")
    if start_action.shape != (7,) or not np.isfinite(start_action).all():
        raise ValueError(f"Invalid start_action for case {case.get('name')}")
    if actions.ndim != 2 or len(actions) < 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        raise ValueError(f"Invalid actions for case {case.get('name')}")

    seed = start_q.copy()
    records = []
    for idx, action in enumerate(actions):
        rec = {"index": idx, "seed": seed.tolist(), "target": action.tolist()}
        try:
            q_raw, ok = _solve_ik_step(arm, action[:3], action[3:6], seed)
            q_arr = np.asarray(q_raw, dtype=float).reshape(-1)
            finite = q_arr.shape == (6,) and bool(np.isfinite(q_arr).all())
            physical = finite and bool(np.all(q_arr >= LOW) and np.all(q_arr <= UP))
            soft = finite and bool(np.all(q_arr >= LOW + margin) and np.all(q_arr <= UP - margin))
            step = float(np.max(np.abs(q_arr - seed))) if finite else None
            rec.update(
                solver_ok=bool(ok),
                finite=finite,
                q=q_arr.tolist() if finite else None,
                physical_ok=physical,
                soft_ok=soft,
                max_joint_step_rad=step,
                outside_soft_joint_idx=(
                    np.flatnonzero((q_arr < LOW + margin) | (q_arr > UP - margin)).tolist()
                    if finite
                    else None
                ),
            )
            if ok and physical:
                seed = q_arr
        except Exception as exc:
            rec.update(solver_ok=False, finite=False, error=repr(exc))
        records.append(rec)

    failures = [r["index"] for r in records if not r.get("solver_ok") or not r.get("finite")]
    xyz_steps = 1000.0 * np.linalg.norm(
        np.diff(np.vstack([start_action, actions])[:, :3], axis=0), axis=1
    )
    grip_steps = 1000.0 * np.abs(np.diff(np.r_[start_action[6], actions[:, 6]]))

    reasons = []
    if failures:
        reasons.append("ik")
    solved_out_phys = sum(r.get("solver_ok", False) and r.get("finite", False) and not r.get("physical_ok", False) for r in records)
    solved_out_soft = sum(r.get("solver_ok", False) and r.get("finite", False) and not r.get("soft_ok", False) for r in records)
    if solved_out_phys or solved_out_soft:
        reasons.append("limits")
    if np.any(start_q < LOW) or np.any(start_q > UP):
        reasons.append("start_physical_limits")
    if np.any(actions[:, 6] < 0.0) or np.any(actions[:, 6] > GRIPPER_MAX_M):
        reasons.append("gripper_range")
    if loop_index == 0:
        if float(xyz_steps[0]) > 25.0:
            reasons.append("initial_target_gap")
    elif float(xyz_steps.max()) > 15.0:
        reasons.append("cartesian_step")
    if float(grip_steps.max()) > 15.0:
        reasons.append("gripper_step")

    raw_q_path = None
    smoothed_q_path = None
    post_smooth_stats = None
    q_steps = None if failures else np.asarray([r["max_joint_step_rad"] for r in records], dtype=float)
    if q_steps is not None:
        first_gap_limit = 0.25 if loop_index == 0 else 0.05
        if float(q_steps[0]) > first_gap_limit:
            reasons.append("first_joint_gap")
        if len(q_steps) > 1 and float(q_steps[1:].max()) > step_limit_rad:
            reasons.append("refined_joint_step")
        raw_q_path = np.asarray([r["q"] for r in records], dtype=float)
        if do_smooth:
            smoothed_q_path = np.asarray(
                smooth_joint_waypoints(raw_q_path, alpha=smooth_alpha, passes=smooth_passes),
                dtype=float,
            )
        else:
            smoothed_q_path = raw_q_path.copy()
        try:
            post_smooth_limit = max(step_limit_rad, 0.25 if case.get("loop_index") == 0 else step_limit_rad)
            post_smooth_stats = validate_post_smoothing_waypoints(
                smoothed_q_path,
                raw_waypoints=raw_q_path if do_smooth else None,
                margin=margin,
                max_soft_clip_rad=0.0,
                max_joint_step_rad=post_smooth_limit,
            )
        except ValueError as exc:
            reasons.append("post_smooth_validation")
            post_smooth_stats = {"error": repr(exc)}

    planner_info = {"planner": "not_run", "timeline": None}
    planned_summary = None
    if not reasons and smoothed_q_path is not None:
        try:
            planned = arm.plan_joint_waypoints_with_gripper(
                start_q.tolist(),
                smoothed_q_path.tolist(),
                (actions[:, 6] / GRIPPER_MAX_M).tolist(),
                time_sec=duration_s,
            )
            planned_arr = np.asarray(planned, dtype=float)
            planned_summary = validate_planned_timeline(
                planned_arr, expected_duration_s=duration_s, margin=margin
            )
            planner_info = {"planner": "passed", "timeline": planned_summary}
        except Exception as exc:
            reasons.append("planner")
            planner_info = {"planner": "failed", "error": repr(exc), "timeline": None}

    out = {
        "name": case["name"],
        "source": case.get("source"),
        "phase": case.get("phase"),
        "loop_index": loop_index,
        "omitted_prefix_candidates": omitted_prefix,
        "inserted_prefix_points": inserted_prefix,
        "checked": len(records),
        "solver_failed": len(failures),
        "first_solver_failure": failures[0] if failures else None,
        "solved_outside_physical": int(solved_out_phys),
        "solved_outside_soft": int(solved_out_soft),
        "max_xyz_step_mm": float(xyz_steps.max()),
        "max_xyz_step_target_index": int(xyz_steps.argmax()),
        "first_xyz_step_mm": float(xyz_steps[0]),
        "max_gripper_step_mm": float(grip_steps.max()),
        "first_joint_gap_rad": None if q_steps is None else float(q_steps[0]),
        "max_interior_joint_step_rad": (
            None if (q_steps is None or len(q_steps) <= 1) else float(q_steps[1:].max())
        ),
        "post_smoothing": post_smooth_stats,
        "reject": reasons,
        "passed": len(reasons) == 0 and planner_info.get("planner") == "passed",
        **planner_info,
    }
    if record_trajectories:
        out["raw_actions"] = case.get("raw_actions")
        out["conditioned_actions"] = actions.tolist()
        out["raw_joint_targets"] = None if raw_q_path is None else raw_q_path.tolist()
        out["smoothed_joint_targets"] = None if smoothed_q_path is None else smoothed_q_path.tolist()
    return out


def audit_sdk_hold_cleanup_capabilities(arm):
    """Inspect vendor ArmController methods available for non-homing safe abort."""
    return {
        "has_cleanup": callable(getattr(arm, "cleanup", None)),
        "has_stop": callable(getattr(arm, "stop", None)),
        "has_hold": callable(getattr(arm, "hold", None)),
        "has_disable": callable(getattr(arm, "disable", None)),
        "has_go_home": callable(getattr(arm, "go_home", None)),
    }


def run_bundle_evaluation(arm, bundle, *, input_sha256="", interfaces=None,
                          sdk_file=None, sdk_version=None,
                          record_trajectories=True):
    """Run G4 evaluation across every case in a validated handoff bundle."""
    cases_out = []
    for case in bundle["cases"]:
        res = evaluate_case_with_arm(
            arm, case, record_trajectories=record_trajectories
        )
        cases_out.append(res)
    passed_count = sum(1 for c in cases_out if c["passed"])
    return {
        "schema": REPORT_SCHEMA,
        "bundle_schema": bundle.get("schema"),
        "input_sha256": input_sha256,
        "interfaces": list(interfaces or ["lo"]),
        "hardware_connected": False,
        "sdk_file": sdk_file,
        "sdk_version": sdk_version,
        "sdk_capabilities": audit_sdk_hold_cleanup_capabilities(arm),
        "cases_total": len(cases_out),
        "cases_passed": passed_count,
        "cases_failed": len(cases_out) - passed_count,
        "all_passed": passed_count == len(cases_out) and len(cases_out) > 0,
        "cases": cases_out,
    }


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True, help="G4 SDK handoff bundle JSON")
    ap.add_argument("--output", type=Path, required=True, help="New output report JSON path")
    ap.add_argument(
        "--sdk-interface-dir",
        type=Path,
        default=Path(os.environ.get("STARTOUCH_SDK_PY_DIR", "./third_party/startouch_sdk/interface_py")),
        help="Vendor SDK interface_py directory (default: $STARTOUCH_SDK_PY_DIR or ./third_party/startouch_sdk/interface_py)",
    )
    ap.add_argument(
        "--compact",
        action="store_true",
        help="Omit per-step trajectory arrays from output JSON to keep file small",
    )
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    interfaces = enforce_loopback_only_net()
    if args.output.exists():
        raise SystemExit(f"Output already exists; choose a new report path: {args.output}")
    raw_bytes = args.input.read_bytes()
    bundle = json.loads(raw_bytes)
    if bundle.get("schema") != BUNDLE_SCHEMA or not bundle.get("cases"):
        raise SystemExit(f"Unexpected bundle schema or empty cases: {bundle.get('schema')}")

    sdk_dir = str(args.sdk_interface_dir)
    if not os.path.isdir(sdk_dir):
        raise SystemExit(
            f"Vendor SDK directory not found on this host: {sdk_dir}. "
            "Copy g4_sdk_handoff_bundle.json and src/ to <ROBOT_HOST> and run inside sudo unshare --net."
        )
    if sdk_dir not in sys.path:
        sys.path.insert(0, sdk_dir)
    import startouch  # noqa: E402

    arm = None
    report = None
    cleanup_returned = False
    try:
        arm = startouch.ArmController(
            can_interface="no_robot_can",
            enable_fd=False,
            gripper_exist=True,
            dry_run=True,
        )
        report = run_bundle_evaluation(
            arm,
            bundle,
            input_sha256=hashlib.sha256(raw_bytes).hexdigest(),
            interfaces=interfaces,
            sdk_file=getattr(startouch, "__file__", None),
            sdk_version=getattr(startouch, "__version__", None),
            record_trajectories=not args.compact,
        )
    finally:
        try:
            if arm is not None:
                arm.cleanup()
                cleanup_returned = True
        finally:
            if report is not None:
                report["cleanup_returned"] = cleanup_returned
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(report, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
                print("REPORT:", args.output, flush=True)
    return 0 if (report and report["all_passed"] and cleanup_returned) else 1


if __name__ == "__main__":
    raise SystemExit(main())

