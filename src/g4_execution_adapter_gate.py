#!/usr/bin/env python3
"""G4 execution adapter, frozen replay, archived dry-run audit, and lifecycle gate."""
from __future__ import annotations

import argparse
import http.server
import json
import os
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
CANDIDATE_DIR = THIS_DIR
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import build_grasp_replay_cases as grasp_cases  # noqa: E402
import g3_http_compatibility_gate as g3  # noqa: E402
import g4_sdk_dryrun_gate as g4_sdk  # noqa: E402
import station_lite_smolvla_right7_guarded as guarded  # noqa: E402
from v070_common import (  # noqa: E402
    load_replay_manifest,
    sha256_file,
    verify_env_identity,
)

DEFAULT_SMOLVLA_ROOT = Path(os.environ.get("SMOLVLA_LOCAL_ROOT", "."))
DEFAULT_MANIFEST = THIS_DIR.parent / "manifests/g0_replay_manifest.json"
DEFAULT_BASELINE_MANIFEST = DEFAULT_SMOLVLA_ROOT / "releases/right7_20261113/BASELINE_MANIFEST.json"
DEFAULT_CHECKPOINT = Path(os.environ.get("SMOLVLA_CHECKPOINT", "./checkpoints/right7_pretrained_model"))

EXPECTED_NEW9_RUNS = {
    "right7_R8_yellow_pick_place_20261113_122503",
    "right7_R2_yellow_pick_place_20261113_122628",
    "right7_R1_yellow_pick_place_20261113_122818",
    "right7_R1_45degree_yellow_pick_place_20261113_123030",
    "right7_R1_45degree_with_2moreblocks_yellow_pick_place_20261113_123253",
    "right7_R9_with_2moreblocks_yellow_pick_place_20261113_123510",
    "right7_R2R3R5R6_with_2moreblocks_red_pick_place_20261113_123819",
    "right7_R5_60degree_with_2moreblocks_cyan_pick_place_20261113_124040",
}


def _wrap_euler_max_abs_deg(a: np.ndarray, b: np.ndarray) -> float:
    delta = (a - b + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.degrees(np.max(np.abs(delta))))


def _new9_runtime_args() -> SimpleNamespace:
    return SimpleNamespace(
        speed=0.1,
        rate=30.0,
        exec_steps=50,
        switch_after=20,
        blend_overlap=10,
        margin=0.02,
        max_soft_clip_rad=0.0,
        phase_speeds=True,
        approach_speed=0.20,
        transport_speed=0.25,
        place_speed=0.24,
        grasp_duration_s=2.5,
        smooth_waypoints=True,
        waypoint_smooth_alpha=0.20,
        waypoint_smooth_passes=1,
        hover_z_m=0.200,
        settled_bridge_max_mm=40.0,
        placement_bridge_max_mm=50.0,
        max_initial_gap_mm=25.0,
        max_adaptive_gap_mm=25.0,
        max_rotation_gap_deg=12.0,
        max_gripper_gap_mm=15.0,
        wrist_align_tol_px=20.0,
        strict_wrist_alignment=False,
        wrist_obb_refine=True,
        high_cam_guidance=True,
        condition_final_grasp=True,
        hold_grasp_pose=True,
        place_after_grasp=True,
        initial_bridge_steps=6,
    )


def audit_archived_new9_vendor_trajectories(smolvla_root: Path) -> dict[str, Any]:
    """Audit all saved aligned segments/waypoints and vendor dry-run plans in diagnostics/20261113_live."""
    live_dir = smolvla_root / "diagnostics/20261113_live"
    snap_dirs = sorted(p for p in live_dir.glob("*_snapshot") if p.is_dir())
    margin = 0.02
    lo = guarded.JOINT_LOW + margin
    hi = guarded.JOINT_UP - margin

    dry_run_files = 0
    new9_dry_run_files = 0
    dry_run_physical_violations = 0
    dry_run_nonfinite_or_nonmono = 0
    dry_run_hz_values: list[float] = []
    dry_run_peak_speed = 0.0
    dry_run_peak_accel = 0.0

    wp_files = 0
    new9_wp_files = 0
    wp_physical_violations = 0
    wp_soft_violations = 0
    wp_post_smooth_failures = 0
    wp_max_joint_step = 0.0
    wp_max_xyz_step_mm = 0.0
    wp_max_grip_step_mm = 0.0

    anomalies: list[dict[str, Any]] = []

    for sdir in snap_dirs:
        run_id = sdir.name[: -len("_snapshot")]
        is_new9 = run_id in EXPECTED_NEW9_RUNS
        for dry_path in sorted(list(sdir.glob("*_controller_dry_run.npy")) + list(sdir.glob("*_bridge_controller_plan.npy"))):
            dry_run_files += 1
            if is_new9:
                new9_dry_run_files += 1
            arr = np.load(dry_path)
            try:
                summary = g4_sdk.validate_planned_timeline(arr, margin=margin)
                if summary["nominal_hz"] is not None:
                    dry_run_hz_values.append(summary["nominal_hz"])
                dry_run_peak_speed = max(dry_run_peak_speed, summary["max_joint_speed_rad_s"])
                dry_run_peak_accel = max(dry_run_peak_accel, summary["max_joint_accel_rad_s2"])
            except Exception as exc:
                dry_run_nonfinite_or_nonmono += 1
                if "physical" in str(exc).lower():
                    dry_run_physical_violations += 1
                anomalies.append({"run": run_id, "file": dry_path.name, "error": repr(exc)})

        for wp_path in sorted(sdir.glob("loop_*_aligned_waypoints.npy")):
            wp_files += 1
            if is_new9:
                new9_wp_files += 1
            wp = np.asarray(np.load(wp_path), dtype=float)
            loop_prefix = wp_path.name.split("_aligned_waypoints.npy")[0]
            seg_path = sdir / f"{loop_prefix}_aligned_segment.npy"
            seg = np.asarray(np.load(seg_path), dtype=float) if seg_path.exists() else None
            if np.any(wp < guarded.JOINT_LOW) or np.any(wp > guarded.JOINT_UP):
                wp_physical_violations += 1
                anomalies.append({"run": run_id, "file": wp_path.name, "error": "physical_limit_violation"})
            if np.any(wp < lo) or np.any(wp > hi):
                wp_soft_violations += 1
                anomalies.append({"run": run_id, "file": wp_path.name, "error": "soft_margin_violation"})
            if len(wp) > 1:
                wp_max_joint_step = max(wp_max_joint_step, float(np.max(np.abs(np.diff(wp, axis=0)))))
            if seg is not None and len(seg) > 1:
                wp_max_xyz_step_mm = max(
                    wp_max_xyz_step_mm,
                    float(1000.0 * np.max(np.linalg.norm(np.diff(seg[:, :3], axis=0), axis=1))),
                )
                wp_max_grip_step_mm = max(
                    wp_max_grip_step_mm,
                    float(1000.0 * np.max(np.abs(np.diff(seg[:, 6])))),
                )
            try:
                sm = guarded.smooth_joint_waypoints(wp, alpha=0.20, passes=1)
                guarded.validate_post_smoothing_waypoints(
                    sm, raw_waypoints=wp, margin=margin, max_soft_clip_rad=0.0, max_joint_step_rad=0.08
                )
            except ValueError as exc:
                wp_post_smooth_failures += 1
                anomalies.append({"run": run_id, "file": wp_path.name, "error": f"post_smooth:{exc}"})

    return {
        "snapshot_dirs_checked": len(snap_dirs),
        "new9_runs_expected": len(EXPECTED_NEW9_RUNS),
        "controller_dry_run_plans_checked": dry_run_files,
        "new9_controller_dry_run_plans_checked": new9_dry_run_files,
        "controller_dry_run_physical_violations": dry_run_physical_violations,
        "controller_dry_run_errors": dry_run_nonfinite_or_nonmono,
        "controller_dry_run_median_hz": float(np.median(dry_run_hz_values)) if dry_run_hz_values else None,
        "controller_dry_run_peak_joint_speed_rad_s": dry_run_peak_speed,
        "controller_dry_run_peak_joint_accel_rad_s2": dry_run_peak_accel,
        "aligned_waypoints_checked": wp_files,
        "new9_aligned_waypoints_checked": new9_wp_files,
        "aligned_waypoints_physical_violations": wp_physical_violations,
        "aligned_waypoints_soft_violations": wp_soft_violations,
        "aligned_waypoints_max_joint_step_rad": wp_max_joint_step,
        "aligned_segment_max_xyz_step_mm": wp_max_xyz_step_mm,
        "aligned_segment_max_grip_step_mm": wp_max_grip_step_mm,
        "post_smoothing_recheck_failures": wp_post_smooth_failures,
        "anomalies": anomalies,
        "passed": (
            dry_run_files > 0
            and wp_files > 0
            and dry_run_physical_violations == 0
            and dry_run_nonfinite_or_nonmono == 0
            and wp_physical_violations == 0
            and wp_soft_violations == 0
            and wp_post_smooth_failures == 0
        ),
    }


def _find_run_held_grip(run_snap_dir: Path, loop_index: int, fallback_grip: float) -> float:
    for prev_idx in range(loop_index - 1, 0, -1):
        grasp_file = run_snap_dir / f"loop_{prev_idx:02d}_conditioned_grasp.npy"
        next_state = run_snap_dir / f"loop_{prev_idx + 1:02d}_state.json"
        if grasp_file.exists() and next_state.exists():
            st = json.loads(next_state.read_text(encoding="utf-8"))
            return float(st["state"][6])
    return float(fallback_grip)


def _find_prev_wrist_obb(run_snap_dir: Path, loop_index: int, target_color: str) -> dict[str, Any] | None:
    prev_obb = None
    for idx in range(1, loop_index):
        wrist_img = run_snap_dir / f"loop_{idx:02d}_cam_right_wrist.jpg"
        if wrist_img.exists():
            obb = grasp_cases.detect_wrist_block_obb(
                wrist_img.read_bytes(), target_color=target_color, prev_obb=prev_obb
            )
            if obb is not None:
                prev_obb = obb
    return prev_obb


def _condition_and_guard_window(
    smolvla_root: Path,
    window: dict[str, Any],
    raw_actions: np.ndarray,
    args_ns: SimpleNamespace,
) -> dict[str, Any]:
    """Run candidate calibration, phase classification, conditioning, and EEF guards on one G0 window."""
    state_path = smolvla_root / window["state_rel_path"]
    run_snap_dir = state_path.parent
    state_data = json.loads(state_path.read_text(encoding="utf-8"))
    state_arr = np.asarray(state_data["state"], dtype=float).reshape(7)
    start_pos = np.asarray(state_data["right_ee_position_m"], dtype=float).reshape(3)
    start_euler = np.asarray(state_data["right_ee_euler_rad"], dtype=float).reshape(3)
    measured_grip = float(state_arr[6])
    start_pose7 = np.r_[start_pos, start_euler, measured_grip]

    loop_index = int(window["loop_index"])
    phase_cat = str(window["phase_category"])
    target_color = str(window["target_color"])

    cam_high_bytes = (smolvla_root / window["images"]["observation.images.cam_high"]["rel_path"]).read_bytes()
    cam_wrist_bytes = (smolvla_root / window["images"]["observation.images.cam_right_wrist"]["rel_path"]).read_bytes()

    loop0_high_path = run_snap_dir / "loop_00_cam_high.jpg"
    high_cam_target = None
    if loop0_high_path.exists():
        high_cam_target = grasp_cases.detect_high_cam_grid_target(
            loop0_high_path.read_bytes(), target_color=target_color
        )
    elif loop_index == 0:
        high_cam_target = grasp_cases.detect_high_cam_grid_target(
            cam_high_bytes, target_color=target_color
        )

    actions = np.asarray(raw_actions, dtype=float)
    defer_grasp = False
    wrist_obb = None
    target_obb = None
    alignment_info = None
    handoff_diag = None
    guard_rejects: list[str] = []
    initial_prefix_candidates = None

    if loop_index == 0:
        phase = "approach"
        raw_gap_mm = float(1000.0 * np.linalg.norm(actions[0, :3] - start_pos))
        rot_gap_deg = _wrap_euler_max_abs_deg(actions[0, 3:6], start_euler)
        grip_gap_mm = float(1000.0 * abs(actions[0, 6] - measured_grip))
        prefix = guarded.make_initial_prefix(start_pose7, actions[0], args_ns.initial_bridge_steps)
        initial_prefix_candidates = prefix
        # Note: exact omission count of leading soft-margin bridge rows runs on robot SDK arm via select_initial_prefix.
        segment = np.concatenate([prefix, actions], axis=0)[: args_ns.exec_steps]
        bridged_gap_mm = float(1000.0 * np.linalg.norm(segment[0, :3] - start_pos))
        if bridged_gap_mm > args_ns.max_initial_gap_mm or rot_gap_deg > args_ns.max_rotation_gap_deg or grip_gap_mm > args_ns.max_gripper_gap_mm:
            guard_rejects.append("initial_target_gap")
    else:
        prev_obb = _find_prev_wrist_obb(run_snap_dir, loop_index, target_color)
        wrist_obb = grasp_cases.detect_wrist_block_obb(
            cam_wrist_bytes, target_color=target_color, prev_obb=prev_obb
        )
        completed_grasp = phase_cat in ("transport_held", "placement_release")
        total_loops = max(6, loop_index + (2 if phase_cat == "transport_held" else 1))
        defer_grasp, _ = guarded.evaluate_grasp_deferral_and_budget(
            actions,
            index=loop_index,
            total_loops=total_loops,
            condition_final_grasp=True,
            completed_conditioned_grasp=completed_grasp,
            place_after_grasp=True,
            max_total_loops=10,
            wrist_obb=wrist_obb,
            ee_z_m=float(start_pos[2]),
            ee_pose6=start_pose7[:6],
        )
        conditioned_grasp = loop_index >= 3 and not defer_grasp and not completed_grasp
        is_placement_chunk = completed_grasp

        segment = actions.copy()
        if conditioned_grasp:
            phase = "grasp"
            if not np.any(actions[:, 6] < 0.04):
                guard_rejects.append("grasp_no_closure_predicted")
            else:
                alignment_info = grasp_cases.check_wrist_grasp_alignment(
                    wrist_obb,
                    ee_z_m=float(start_pose7[2]),
                    ee_pose6=start_pose7[:6],
                    tol_u_px=args_ns.wrist_align_tol_px,
                    tol_v_px=args_ns.wrist_align_tol_px,
                )
                if not alignment_info["aligned"]:
                    gross = (
                        (alignment_info["du"] is not None and abs(alignment_info["du"]) > 30.0)
                        or (alignment_info["dv"] is not None and abs(alignment_info["dv"]) > 30.0)
                    )
                    if gross or args_ns.strict_wrist_alignment:
                        guard_rejects.append("grasp_gross_misaligned")
                if not guard_rejects:
                    try:
                        segment, _, _ = grasp_cases.condition_grasp_actions(
                            start_pose7,
                            actions,
                            max_position_m=args_ns.settled_bridge_max_mm / 1000.0,
                            hold_pose=True,
                            wrist_obb=wrist_obb,
                        )
                    except ValueError as exc:
                        guard_rejects.append(f"condition_grasp:{exc}")
        elif is_placement_chunk:
            held_grip = _find_run_held_grip(run_snap_dir, loop_index, measured_grip)
            target_obb = grasp_cases.detect_wrist_target_obb(cam_wrist_bytes)
            try:
                segment, _, _ = grasp_cases.condition_placement_actions(
                    start_pose7,
                    actions,
                    held_grip=held_grip,
                    max_position_m=args_ns.placement_bridge_max_mm / 1000.0,
                    target_obb=target_obb,
                )
            except ValueError as exc:
                guard_rejects.append(f"condition_placement:{exc}")
            release_threshold = max(0.040, min(0.070, float(held_grip) + 0.010))
            phase = "place" if np.any(segment[:, 6] >= release_threshold) else "transport"
        else:
            phase = "approach"
            grid_hover = high_cam_target["hover_pose"] if (high_cam_target and loop_index == 1) else None
            try:
                segment, _, _ = grasp_cases.condition_approach_actions(
                    start_pose7,
                    actions,
                    max_position_m=args_ns.settled_bridge_max_mm / 1000.0,
                    wrist_obb=wrist_obb,
                    hover_z_m=args_ns.hover_z_m,
                    grid_hover_pose=grid_hover,
                )
            except ValueError as exc:
                guard_rejects.append(f"condition_approach:{exc}")

        prev_seg = np.asarray(state_data["previous_action_segment"], dtype=float)
        prev_wp = np.asarray(state_data["previous_right_waypoints"], dtype=float)
        handoff_diag = guarded.diagnose_completed_handoff(
            prev_seg,
            prev_wp,
            segment,
            state_arr[:6],
            state_arr[:6],
            measured_grip,
            args_ns.max_adaptive_gap_mm,
            args_ns.max_rotation_gap_deg,
            args_ns.max_gripper_gap_mm,
            require_closed=False,
        )
        if (
            not conditioned_grasp
            and not is_placement_chunk
            and handoff_diag["reason"] in ("position_gap", "completed_chunk_pass")
            and (handoff_diag["position_gap_mm"] > 6.0 or handoff_diag["rotation_gap_deg"] > 1.5)
            and handoff_diag["position_gap_mm"] <= args_ns.settled_bridge_max_mm
        ):
            prefix = guarded.make_initial_prefix(start_pose7, segment[0], 6, max_position_m=args_ns.settled_bridge_max_mm / 1000.0)
            segment = np.concatenate([prefix, segment], axis=0)
            handoff_diag = guarded.diagnose_completed_handoff(
                prev_seg,
                prev_wp,
                segment,
                state_arr[:6],
                state_arr[:6],
                measured_grip,
                args_ns.max_adaptive_gap_mm,
                args_ns.max_rotation_gap_deg,
                args_ns.max_gripper_gap_mm,
                require_closed=False,
            )
        if handoff_diag["reason"] != "completed_chunk_pass":
            guard_rejects.append(f"handoff:{handoff_diag['reason']}")

    _, chunk_duration_s = guarded.get_phase_speed_and_duration(phase, len(segment), args_ns)
    xyz_steps_mm = 1000.0 * np.linalg.norm(
        np.diff(np.vstack([start_pose7, segment])[:, :3], axis=0), axis=1
    )
    grip_steps_mm = 1000.0 * np.abs(np.diff(np.r_[start_pose7[6], segment[:, 6]]))
    # Production enforces max_xyz_step_mm <= 15.0 mm for adaptive/settled chunks (index > 0);
    # loop_index == 0 validates joint-space 0.25 rad + bounded initial gap + controller dry-run.
    if loop_index > 0 and float(xyz_steps_mm.max()) > 15.0:
        guard_rejects.append(f"cartesian_step_gt_15mm:{xyz_steps_mm.max():.2f}")
    if float(grip_steps_mm.max()) > 15.0:
        guard_rejects.append(f"gripper_step_gt_15mm:{grip_steps_mm.max():.2f}")
    if np.any(segment[:, 6] < 0.0) or np.any(segment[:, 6] > guarded.GRIPPER_MAX_M):
        guard_rejects.append("gripper_out_of_range")

    return {
        "window_id": window["window_id"],
        "loop_index": loop_index,
        "phase": phase,
        "defer_grasp": bool(defer_grasp),
        "chunk_duration_s": float(chunk_duration_s),
        "segment": segment,
        "initial_prefix_candidates": initial_prefix_candidates,
        "start_q": state_arr[:6],
        "start_action": start_pose7,
        "high_cam_cell": None if high_cam_target is None else high_cam_target.get("cell"),
        "wrist_obb_mode": None if wrist_obb is None else wrist_obb.get("grasp_axis_mode"),
        "target_obb_present": target_obb is not None,
        "max_xyz_step_mm": float(xyz_steps_mm.max()),
        "max_grip_step_mm": float(grip_steps_mm.max()),
        "guard_rejects": guard_rejects,
        "passed_host_guards": len(guard_rejects) == 0,
    }


def replay_g0_through_candidate_adapter(
    *,
    smolvla_root: Path,
    manifest_path: Path,
    checkpoint: Path,
    server_script: Path,
    host: str,
    port: int,
    seed: int,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run all 12 G0 windows live against candidate HTTP server (127.0.0.1:8011) and compare vs 0.6.2 baseline conditioning."""
    if port == g3.RESERVED_PRODUCTION_PORT:
        raise SystemExit(f"Refusing to run against reserved production port {port}")
    manifest = load_replay_manifest(manifest_path)
    args_ns = _new9_runtime_args()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "g4_candidate_server.log"

    proc, drain_thread, health, ready_s = g3._start_candidate_server(
        server_script=server_script,
        checkpoint=checkpoint,
        host=host,
        port=port,
        seed=seed,
        seed_mode="fixed",
        log_path=log_path,
    )
    base_url = f"http://{host}:{port}"
    try:
        # Verify BOTH candidate client health check AND G3 provenance health check.
        guarded_health = guarded.validate_right7_health(base_url, str(checkpoint))
        g3.validate_right7_health(
            guarded_health,
            expected_checkpoint=str(checkpoint),
            expected_seed_mode="fixed",
            expected_lerobot="0.7.0",
        )
        session_id = f"g4-adapter-{uuid.uuid4().hex[:10]}"
        window_comparisons = []
        sdk_cases = []
        base_sdk_cases = []
        for idx, win in enumerate(manifest["windows"]):
            payload = g3.build_http_payload_for_window(
                smolvla_root, win, session_id=session_id, seq=idx
            )
            status, _, resp = g3._http_request_raw(
                base_url + "/predict",
                method="POST",
                body=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                timeout=30.0,
            )
            if status != 200 or not isinstance(resp, dict):
                raise RuntimeError(f"/predict failed on {win['window_id']}: status={status}, resp={resp}")
            v070_raw = guarded.validate_predict_response(
                resp,
                expected_session_id=session_id,
                expected_seq=idx,
                expected_seed_mode="fixed",
                expected_seed_base=guarded_health.get("seed_base"),
            )
            base_raw = np.asarray(np.load(smolvla_root / win["raw_actions_rel_path"]), dtype=float)

            v070_res = _condition_and_guard_window(smolvla_root, win, v070_raw, args_ns)
            base_res = _condition_and_guard_window(smolvla_root, win, base_raw, args_ns)

            seg70 = v070_res["segment"]
            seg62 = base_res["segment"]
            same_len = len(seg70) == len(seg62)
            cond_xyz_mm = (
                float(1000.0 * np.max(np.linalg.norm(seg70[:, :3] - seg62[:, :3], axis=1)))
                if same_len
                else None
            )
            cond_rot_deg = (
                _wrap_euler_max_abs_deg(seg70[:, 3:6], seg62[:, 3:6]) if same_len else None
            )
            cond_grip_mm = (
                float(1000.0 * np.max(np.abs(seg70[:, 6] - seg62[:, 6]))) if same_len else None
            )

            raw_xyz_mm = float(1000.0 * np.max(np.linalg.norm(v070_raw[:, :3] - base_raw[:, :3], axis=1)))
            raw_rot_deg = _wrap_euler_max_abs_deg(v070_raw[:, 3:6], base_raw[:, 3:6])
            raw_grip_mm = float(1000.0 * np.max(np.abs(v070_raw[:, 6] - base_raw[:, 6])))

            window_comparisons.append({
                "window_id": win["window_id"],
                "loop_index": win["loop_index"],
                "phase_v070": v070_res["phase"],
                "phase_base": base_res["phase"],
                "phase_match": v070_res["phase"] == base_res["phase"],
                "defer_match": v070_res["defer_grasp"] == base_res["defer_grasp"],
                "high_cam_cell": v070_res["high_cam_cell"],
                "wrist_obb_mode": v070_res["wrist_obb_mode"],
                "conditioned_len_v070": len(seg70),
                "conditioned_len_base": len(seg62),
                "duration_s_v070": v070_res["chunk_duration_s"],
                "max_xyz_step_mm_v070": v070_res["max_xyz_step_mm"],
                "max_grip_step_mm_v070": v070_res["max_grip_step_mm"],
                "passed_host_guards_v070": v070_res["passed_host_guards"],
                "passed_host_guards_base": base_res["passed_host_guards"],
                "guard_rejects_v070": v070_res["guard_rejects"],
                "raw_max_xyz_diff_mm": raw_xyz_mm,
                "raw_max_rot_diff_deg": raw_rot_deg,
                "raw_max_grip_diff_mm": raw_grip_mm,
                "conditioned_max_xyz_diff_mm": cond_xyz_mm,
                "conditioned_max_rot_diff_deg": cond_rot_deg,
                "conditioned_max_grip_diff_mm": cond_grip_mm,
            })
            case_common = {
                "name": f"v070_{win['window_id']}",
                "source": f"g4_live_127.0.0.1:{port}_fixed_seed{seed}",
                "window_id": win["window_id"],
                "loop_index": win["loop_index"],
                "phase": v070_res["phase"],
                "duration_s": v070_res["chunk_duration_s"],
                "margin_rad": args_ns.margin,
                "smooth_waypoints": args_ns.smooth_waypoints,
                "waypoint_smooth_alpha": args_ns.waypoint_smooth_alpha,
                "waypoint_smooth_passes": args_ns.waypoint_smooth_passes,
                "max_interior_joint_step_rad": 0.07 if win["loop_index"] > 0 else 0.25,
                "requires_initial_prefix_selection": bool(win["loop_index"] == 0),
                "initial_prefix_candidates": (
                    None
                    if v070_res["initial_prefix_candidates"] is None
                    else v070_res["initial_prefix_candidates"].tolist()
                ),
                "seed": v070_res["start_q"].tolist(),
                "start_action": v070_res["start_action"].tolist(),
                "actions": seg70.tolist(),
                "raw_actions": v070_raw.tolist(),
            }
            sdk_cases.append(case_common)
            base_sdk_cases.append({
                **case_common,
                "name": f"base062_{win['window_id']}",
                "source": win["raw_actions_rel_path"],
                "phase": base_res["phase"],
                "duration_s": base_res["chunk_duration_s"],
                "initial_prefix_candidates": (
                    None
                    if base_res["initial_prefix_candidates"] is None
                    else base_res["initial_prefix_candidates"].tolist()
                ),
                "actions": seg62.tolist(),
                "raw_actions": base_raw.tolist(),
            })
    finally:
        g3._stop_server(proc, drain_thread, host, port)

    bundle = {
        "schema": g4_sdk.BUNDLE_SCHEMA,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": str(checkpoint),
        "manifest_id": manifest["manifest_id"],
        "hardware_connected": False,
        "cases": sdk_cases,
        "baseline_reference_cases": base_sdk_cases,
    }
    summary = {
        "cold_startup_seconds": round(ready_s, 3),
        "windows_evaluated": len(window_comparisons),
        "all_passed_host_guards_v070": all(w["passed_host_guards_v070"] for w in window_comparisons),
        "all_passed_host_guards_base": all(w["passed_host_guards_base"] for w in window_comparisons),
        "phase_agreements": sum(w["phase_match"] for w in window_comparisons),
        "defer_agreements": sum(w["defer_match"] for w in window_comparisons),
        "max_conditioned_xyz_diff_mm": max(
            w["conditioned_max_xyz_diff_mm"] for w in window_comparisons if w["conditioned_max_xyz_diff_mm"] is not None
        ),
        "max_conditioned_rot_diff_deg": max(
            w["conditioned_max_rot_diff_deg"] for w in window_comparisons if w["conditioned_max_rot_diff_deg"] is not None
        ),
        "max_conditioned_grip_diff_mm": max(
            w["conditioned_max_grip_diff_mm"] for w in window_comparisons if w["conditioned_max_grip_diff_mm"] is not None
        ),
        "windows": window_comparisons,
    }
    return summary, bundle


class _MockPolicyHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    state_ref: dict[str, Any] = {}

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_error(404)
            return
        body = json.dumps({
            "schema": guarded.SCHEMA,
            "source_schema": guarded.REQUEST_SCHEMA,
            "checkpoint": str(DEFAULT_CHECKPOINT),
            "chunk_size": 50,
            "state_dim": 7,
            "action_dim": 7,
            "task": guarded.TASK,
            "device": "cuda",
            "seed_mode": "fixed",
            "seed_base": 42,
            "hardware_connected": False,
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path != "/predict":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(length).decode("utf-8"))
        st = self.state_ref
        mode = st.get("mode", "ok")
        if mode == "http_exception":
            self.send_error(500, "Injected /predict fault")
            return
        actions = np.zeros((50, 7), dtype=float)
        if mode == "guard_gap_fail":
            actions[:, 0] = 0.500
        else:
            actions[:, 0] = 0.130
            actions[:, 1] = 0.005
            actions[:, 2] = float(st.get("target_z", 0.195))
            actions[:, 6] = float(req["state"][6])
        body = json.dumps({
            "schema": guarded.SCHEMA,
            "source_schema": guarded.REQUEST_SCHEMA,
            "session_id": req["session_id"],
            "seq": req["seq"],
            "action_dim": 7,
            "hardware_connected": False,
            "sampling_seed": 42,
            "actions": actions.tolist(),
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_lifecycle_audit_suite() -> dict[str, Any]:
    """Exercise all candidate lifecycle paths (startup, normal, holding, low-Z, guard, exception, Ctrl-C, cleanup)."""
    mock_state = {"mode": "ok", "target_z": 0.195}
    _MockPolicyHandler.state_ref = mock_state
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _MockPolicyHandler)
    port = srv.server_port
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    server_url = f"http://127.0.0.1:{port}"
    scenarios: list[dict[str, Any]] = []

    class _FakeInnerArm:
        def plan_joint_waypoints_with_gripper(self, start_q, wp, grip, time_sec=1.0):
            arr = np.zeros((10, 15), dtype=float)
            arr[:, 0] = np.linspace(0.0, float(time_sec), 10)
            arr[:, 1:7] = np.asarray(start_q, dtype=float)
            return arr

    class _FakeRightArm:
        def __init__(self, *, ee_z=0.200, grip=0.068, interrupt_on_send_call=None, drift_after_send=False):
            self.arm = _FakeInnerArm()
            self.ee_z = ee_z
            self.grip = grip
            self.interrupt_on_send_call = interrupt_on_send_call
            self.drift_after_send = drift_after_send
            self.sent = 0
            self.q_reads_after_send = 0

        def get_joint_positions(self):
            base = [0.0, 0.2, -0.3, 0.1, 0.0, 0.0]
            if self.drift_after_send and self.sent > 0:
                self.q_reads_after_send += 1
                if self.q_reads_after_send >= 2:
                    return [0.025, 0.2, -0.3, 0.1, 0.0, 0.0]
            return base

        def get_gripper_distance(self):
            return self.grip

        def get_ee_pose_euler(self):
            return [0.127, 0.002, self.ee_z], [0.0, 0.0, 0.0]

        def solve_ik_euler(self, pos, euler, seed):
            return [0.01, 0.21, -0.31, 0.1, 0.0, 0.0], True

        def update_joint_waypoint_chunk_with_gripper(self, wp, grip, time_sec=1.0, switch_delay_sec=0.05):
            self.sent += 1
            if self.interrupt_on_send_call is not None and self.sent >= self.interrupt_on_send_call:
                raise KeyboardInterrupt("Simulated operator Ctrl-C")

    def _run_case(
        name: str,
        cli_args: list[str],
        *,
        mode: str = "ok",
        ee_z: float = 0.200,
        target_z: float = 0.195,
        grip: float = 0.068,
        q_pos: list[float] | None = None,
        interrupt_on_send_call: int | None = None,
        drift_after_send: bool = False,
        with_teardown: bool = True,
        use_hold_confirm_cb: bool = True,
        second_interrupt_in_hold: bool = False,
        force_non_tty: bool = False,
    ) -> dict[str, Any]:
        mock_state["mode"] = mode
        mock_state["target_z"] = target_z
        counters = {
            "robot_init": 0,
            "home_calls": 0,
            "img_closed": 0,
            "disconnect_calls": 0,
            "safe_stop_calls": 0,
            "cleanup_calls": 0,
            "arm_cleanup_calls": 0,
            "hold_confirm_calls": 0,
            "disconnect_at_hold_confirm": None,
        }

        class _SimpleLeftArm:
            def cleanup(self):
                counters["arm_cleanup_calls"] += 1

        class _FakeRightArmReal(_FakeRightArm):
            def __init__(self, **kw):
                super().__init__(**kw)
                self._q_pos = list(q_pos) if q_pos is not None else [0.0, 0.2, -0.3, 0.1, 0.0, 0.0]

            def get_joint_positions(self):
                if q_pos is not None:
                    return list(self._q_pos)
                return super().get_joint_positions()

            if with_teardown:
                def cleanup(self):
                    counters["arm_cleanup_calls"] += 1

        class _FakeRobot:
            def __init__(self):
                counters["robot_init"] += 1
                self.left_arm = _SimpleLeftArm() if with_teardown else type("_NoCleanupArm", (), {})()
                self.right_arm = _FakeRightArmReal(
                    ee_z=ee_z,
                    grip=grip,
                    interrupt_on_send_call=interrupt_on_send_call,
                    drift_after_send=drift_after_send,
                )

            def go_home(self):
                counters["home_calls"] += 1

            def safe_stop(self):
                counters["safe_stop_calls"] += 1

            if with_teardown:
                def disconnect(self):
                    counters["disconnect_calls"] += 1
                    self.safe_stop()
                    self.left_arm.cleanup()
                    self.right_arm.cleanup()

        class _FakeImg:
            def __init__(self, host="127.0.0.1", auto_subscribe=False):
                self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

            def start_subscriptions(self):
                return

            def get_head_frame(self):
                return self.frame, 1.0

            def get_right_wrist_frame(self):
                return self.frame, 1.0

            def close(self):
                counters["img_closed"] += 1

        def _confirm_cb(_audit):
            counters["hold_confirm_calls"] += 1
            counters["disconnect_at_hold_confirm"] = counters["disconnect_calls"]
            if second_interrupt_in_hold:
                raise KeyboardInterrupt("Second operator interrupt during hold pause")

        with tempfile.TemporaryDirectory() as td:
            audit_path = Path(td) / f"{name}.json"
            full_argv = [
                "--server", server_url,
                "--expected-checkpoint", str(DEFAULT_CHECKPOINT),
                "--loops", "1",
                "--exec-steps", "50",
                "--switch-after", "20",
                "--blend-overlap", "10",
                "--speed", "1.0",
                "--rate", "100.0",
                "--send-right-chunks",
                "--no-smooth-waypoints",
                "--lifecycle-audit-json", str(audit_path),
                *cli_args,
            ]
            exc_type = None
            t0 = time.perf_counter()
            orig_sleep = guarded.time.sleep
            orig_stdin = guarded.sys.stdin
            guarded.time.sleep = lambda _s: None
            if force_non_tty:
                class _NonTtyStdin:
                    def isatty(self) -> bool:
                        return False

                guarded.sys.stdin = _NonTtyStdin()
            try:
                guarded.main(
                    full_argv,
                    robot_cls=_FakeRobot,
                    image_client_cls=_FakeImg,
                    hold_confirm_fn=_confirm_cb if use_hold_confirm_cb else None,
                )
            except BaseException as exc:
                exc_type = f"{type(exc).__name__}:{exc}"
            finally:
                guarded.sys.stdin = orig_stdin
                guarded.time.sleep = orig_sleep
            elapsed_s = time.perf_counter() - t0
            audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else None
            return {
                "scenario": name,
                "exception": exc_type,
                "elapsed_s": round(elapsed_s, 4),
                **counters,
                "audit": audit,
            }

    try:
        s1 = _run_case("startup_unconfirmed_refused", [])
        s1["passed"] = s1["robot_init"] == 0 and s1["home_calls"] == 0 and "SystemExit" in (s1["exception"] or "")
        scenarios.append(s1)

        s2 = _run_case(
            "non_tty_confirm_before_disable_refused",
            ["--confirm-startup-home"],
            use_hold_confirm_cb=False,
            force_non_tty=True,
        )
        s2["passed"] = (
            s2["robot_init"] == 0
            and s2["home_calls"] == 0
            and "SystemExit" in (s2["exception"] or "")
            and "non-TTY" in (s2["exception"] or "")
        )
        scenarios.append(s2)

        s3 = _run_case("normal_completion_confirmed", ["--confirm-startup-home", "--confirm-exit-home"], ee_z=0.200, grip=0.068)
        s3["passed"] = (
            s3["exception"] is None
            and s3["home_calls"] == 1
            and s3["img_closed"] == 1
            and s3["disconnect_calls"] == 1
            and s3["arm_cleanup_calls"] == 2
            and s3["hold_confirm_calls"] == 0
            and s3["audit"]["exit_reason"] == "normal_completion"
            and s3["audit"]["hold_pause_requested"] is False
            and s3["audit"]["motors_disabled_on_exit"] is True
        )
        scenarios.append(s3)

        s4 = _run_case("normal_completion_unconfirmed_exit_no_home", ["--confirm-startup-home"], ee_z=0.200, grip=0.068)
        s4["passed"] = (
            s4["exception"] is None
            and s4["home_calls"] == 0
            and s4["hold_confirm_calls"] == 1
            and s4["disconnect_at_hold_confirm"] == 0
            and s4["disconnect_calls"] == 1
            and s4["audit"]["hold_pause_kept_motors_enabled"] is True
            and s4["audit"]["hold_pause_confirmed"] is True
            and "exit_home_not_confirmed" in s4["audit"]["home_decision"]["suppress_reasons"]
        )
        scenarios.append(s4)

        s5 = _run_case("normal_completion_holding_object_no_home", ["--confirm-startup-home", "--confirm-exit-home"], ee_z=0.200, target_z=0.195, grip=0.025)
        s5["passed"] = (
            s5["exception"] is None
            and s5["home_calls"] == 0
            and s5["hold_confirm_calls"] == 1
            and s5["disconnect_at_hold_confirm"] == 0
            and s5["disconnect_calls"] == 1
            and s5["audit"]["exit_reason"] == "normal_completion"
            and "holding_object_or_unreleased_grasp" in s5["audit"]["home_decision"]["suppress_reasons"]
        )
        scenarios.append(s5)

        s6 = _run_case("normal_completion_low_z_no_home", ["--confirm-startup-home", "--confirm-exit-home"], ee_z=0.160, target_z=0.162, grip=0.068)
        s6["passed"] = (
            s6["exception"] is None
            and s6["home_calls"] == 0
            and s6["hold_confirm_calls"] == 1
            and s6["disconnect_at_hold_confirm"] == 0
            and s6["disconnect_calls"] == 1
            and s6["audit"]["exit_reason"] == "normal_completion"
            and s6["audit"]["placement_completed"] is False
            and s6["audit"]["post_placement_lift_attempted"] is False
            and "near_table_contact_low_z" in s6["audit"]["home_decision"]["suppress_reasons"]
        )
        scenarios.append(s6)

        s7 = _run_case(
            "guard_rejection_no_home",
            ["--confirm-startup-home", "--confirm-exit-home", "--loops", "2", "--adaptive-refine", "--settled-chunks"],
            drift_after_send=True,
        )
        s7["passed"] = (
            s7["home_calls"] == 0
            and s7["img_closed"] == 1
            and s7["hold_confirm_calls"] == 1
            and s7["disconnect_at_hold_confirm"] == 0
            and s7["disconnect_calls"] == 1
            and s7["audit"]["exit_reason"] == "guard_rejected"
            and "abort_guard_rejected" in s7["audit"]["home_decision"]["suppress_reasons"]
        )
        scenarios.append(s7)

        s8 = _run_case("http_exception_away_from_home_no_home", ["--confirm-startup-home", "--confirm-exit-home"], mode="http_exception")
        s8["passed"] = (
            s8["home_calls"] == 0
            and s8["img_closed"] == 1
            and s8["hold_confirm_calls"] == 1
            and s8["disconnect_at_hold_confirm"] == 0
            and s8["disconnect_calls"] == 1
            and s8["audit"]["exit_reason"] == "exception"
        )
        scenarios.append(s8)

        s9 = _run_case(
            "http_exception_already_at_home_no_pause",
            ["--confirm-startup-home", "--confirm-exit-home"],
            mode="http_exception",
            q_pos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        s9["passed"] = (
            s9["home_calls"] == 0
            and s9["img_closed"] == 1
            and s9["hold_confirm_calls"] == 0
            and s9["disconnect_calls"] == 1
            and s9["audit"]["already_at_home"] is True
            and s9["audit"]["hold_pause_skipped_reason"] == "already_at_home"
        )
        scenarios.append(s9)

        s10 = _run_case(
            "keyboard_interrupt_hold_pause_and_second_interrupt",
            ["--confirm-startup-home", "--confirm-exit-home", "--loops", "2", "--adaptive-refine", "--settled-chunks"],
            interrupt_on_send_call=2,
            second_interrupt_in_hold=True,
        )
        s10["passed"] = (
            "KeyboardInterrupt" in (s10["exception"] or "")
            and s10["home_calls"] == 0
            and s10["img_closed"] == 1
            and s10["hold_confirm_calls"] == 1
            and s10["disconnect_at_hold_confirm"] == 0
            and s10["disconnect_calls"] == 1
            and s10["audit"]["exit_reason"] == "keyboard_interrupt"
            and s10["audit"]["skipped_wait_for_interrupt"] is True
            and s10["audit"]["hold_pause_interrupted"] is True
            and s10["elapsed_s"] < 0.5
        )
        scenarios.append(s10)

        s11 = _run_case(
            "unsafe_exit_immediate_disable_opt_in",
            ["--confirm-startup-home", "--unsafe-exit-action", "immediate_disable"],
            ee_z=0.160,
            target_z=0.162,
            use_hold_confirm_cb=False,
        )
        s11["passed"] = (
            s11["exception"] is None
            and s11["home_calls"] == 0
            and s11["hold_confirm_calls"] == 0
            and s11["disconnect_calls"] == 1
            and s11["audit"]["hold_pause_requested"] is False
            and s11["audit"]["hold_pause_skipped_reason"] == "unsafe_exit_action=immediate_disable"
        )
        scenarios.append(s11)

        s12 = _run_case(
            "teardown_missing_honest_flag",
            ["--confirm-startup-home", "--unsafe-exit-action", "immediate_disable"],
            with_teardown=False,
            use_hold_confirm_cb=False,
        )
        s12["passed"] = (
            s12["home_calls"] == 0
            and s12["img_closed"] == 1
            and s12["audit"]["robot_disconnect_callable"] is False
            and s12["audit"]["robot_cleanup_callable"] is False
            and s12["audit"]["safe_hold_verified"] is False
        )
        scenarios.append(s12)

        def _run_post_placement_lift_case(name: str, *, fail_ik: bool) -> dict[str, Any]:
            c = {
                "home_calls": 0,
                "img_closed": 0,
                "disconnect_calls": 0,
                "hold_confirm_calls": 0,
                "disconnect_at_hold_confirm": None,
                "lift_calls": 0,
                "plan_calls": 0,
            }

            class _Inner:
                def plan_joint_waypoints_with_gripper(self, start_q, wp, grip, time_sec=1.0):
                    c["plan_calls"] += 1
                    arr = np.zeros((10, 15), dtype=float)
                    arr[:, 0] = np.linspace(0.0, float(time_sec), 10)
                    arr[:, 1:7] = np.asarray(start_q, dtype=float)
                    return arr

            class _PostPlaceArm:
                def __init__(self):
                    self.arm = _Inner()
                    self.ee_z = 0.1465
                    self.grip = 0.0643
                    self.q = [0.0, 0.2, -0.3, 0.1, 0.0, 0.0]

                def get_gripper_distance(self):
                    return self.grip

                def get_joint_positions(self):
                    return list(self.q)

                def get_ee_pose_euler(self):
                    return [0.3452, 0.2321, self.ee_z], [0.0, 0.0, 0.0]

                def solve_ik_euler(self, pos, euler, seed):
                    if fail_ik:
                        return [0.0] * 6, False
                    frac = max(0.0, min(1.0, (float(pos[2]) - 0.1465) / 0.0535))
                    return [0.0, 0.2 + 0.05 * frac, -0.3, 0.1, 0.0, 0.0], True

                def update_joint_waypoint_chunk_with_gripper(self, wp, grip, time_sec=1.0, switch_delay_sec=0.05):
                    c["lift_calls"] += 1
                    self.q = list(wp[-1])
                    self.ee_z = 0.2000
                    self.grip = float(grip[-1]) * guarded.GRIPPER_MAX_M

                def cleanup(self):
                    return

            class _PostPlaceRobot:
                def __init__(self):
                    self.left_arm = type("_LeftArm", (), {"cleanup": lambda self: None})()
                    self.right_arm = _PostPlaceArm()

                def go_home(self):
                    c["home_calls"] += 1

                def disconnect(self):
                    c["disconnect_calls"] += 1

            class _PostPlaceImg:
                def close(self):
                    c["img_closed"] += 1

            def _confirm_cb(_audit):
                c["hold_confirm_calls"] += 1
                c["disconnect_at_hold_confirm"] = c["disconnect_calls"]

            audit = guarded.execute_candidate_finally(
                robot=_PostPlaceRobot(),
                image_client=_PostPlaceImg(),
                trace_session=None,
                sent_chunks=6,
                last_install=10.0,
                last_duration=2.0,
                switch_delay=0.05,
                exit_reason="normal_completion",
                policy="normal_if_safe_and_confirmed",
                confirm_exit_home=True,
                holding_object=False,
                min_safe_home_z_m=0.175,
                placement_completed=True,
                unsafe_exit_action="confirm_before_disable",
                hold_confirm_fn=_confirm_cb,
                monotonic_fn=lambda: 15.0,
                sleep_fn=lambda _s: None,
            )
            return {"scenario": name, "exception": None, **c, "audit": audit}

        s13 = _run_post_placement_lift_case("post_placement_low_z_lifts_then_homes", fail_ik=False)
        s13["passed"] = (
            s13["home_calls"] == 1
            and s13["lift_calls"] == 1
            and s13["plan_calls"] == 1
            and s13["hold_confirm_calls"] == 0
            and s13["disconnect_calls"] == 1
            and s13["audit"]["post_placement_lift_attempted"] is True
            and s13["audit"]["post_placement_lift_succeeded"] is True
            and s13["audit"]["measured_ee_z_m"] >= 0.200
            and s13["audit"]["home_decision"]["home_permitted"] is True
        )
        scenarios.append(s13)

        s14 = _run_post_placement_lift_case("post_placement_lift_failed_stays_held", fail_ik=True)
        s14["passed"] = (
            s14["home_calls"] == 0
            and s14["lift_calls"] == 0
            and s14["hold_confirm_calls"] == 1
            and s14["disconnect_at_hold_confirm"] == 0
            and s14["disconnect_calls"] == 1
            and s14["audit"]["post_placement_lift_attempted"] is True
            and s14["audit"]["post_placement_lift_succeeded"] is False
            and s14["audit"]["post_placement_lift_error"] is not None
            and "near_table_contact_low_z" in s14["audit"]["home_decision"]["suppress_reasons"]
        )
        scenarios.append(s14)
    finally:
        srv.shutdown()
        srv.server_close()

    return {
        "scenarios_total": len(scenarios),
        "scenarios_passed": sum(1 for s in scenarios if s["passed"]),
        "all_passed": all(s["passed"] for s in scenarios),
        "robot_safe_hold_hardware_verified": False,
        "scenarios": scenarios,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--baseline-manifest", type=Path, default=DEFAULT_BASELINE_MANIFEST)
    ap.add_argument("--smolvla-root", type=Path, default=DEFAULT_SMOLVLA_ROOT)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--server-script", type=Path, default=g3.CANDIDATE_SERVER_DEFAULT)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=THIS_DIR.parent / "results/g4_execution_adapter",
    )
    return ap.parse_args(argv)


def _fast_cycle_runtime_args() -> SimpleNamespace:
    ns = _new9_runtime_args()
    ns.takeoff_speed = 0.30
    ns.aerial_approach_speed = 0.36
    ns.approach_speed = 0.24
    ns.grasp_duration_s = 2.0
    ns.transport_speed = 0.30
    ns.place_speed = 0.28
    ns.settle_margin_s = 0.10
    ns.settle_drift_sleep_s = 0.10
    ns.home_takeoff_hover = True
    ns.fast_grid_approach = True
    ns.trim_idle_tail = True
    ns.direct_place_transport = True
    return ns


def _evaluate_fast_cycle_case_host(
    *,
    name: str,
    phase: str,
    loop_index: int,
    start_pose7: np.ndarray,
    segment: np.ndarray,
    duration_s: float,
    trimmed_points: int = 0,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rejects: list[str] = []
    if len(segment) == 0:
        rejects.append("empty_segment")
        max_xyz = 0.0
        entry_mm = 0.0
        max_grip = 0.0
    else:
        if not np.isfinite(segment).all():
            rejects.append("nonfinite_segment")
        xyz_steps_mm = 1000.0 * np.linalg.norm(
            np.diff(np.vstack([start_pose7, segment])[:, :3], axis=0), axis=1
        )
        grip_steps_mm = 1000.0 * np.abs(np.diff(np.r_[start_pose7[6], segment[:, 6]]))
        entry_mm = float(xyz_steps_mm[0])
        max_xyz = float(xyz_steps_mm.max())
        max_grip = float(grip_steps_mm.max())
        step_cap_mm = 25.0 if loop_index == 0 else 15.0
        if entry_mm > (25.0 if loop_index == 0 else 15.0):
            rejects.append(f"entry_gap_gt_{25 if loop_index == 0 else 15}mm:{entry_mm:.2f}")
        if max_xyz > step_cap_mm:
            rejects.append(f"cartesian_step_gt_{int(step_cap_mm)}mm:{max_xyz:.2f}")
        if max_grip > 15.0:
            rejects.append(f"gripper_step_gt_15mm:{max_grip:.2f}")
        if np.any(segment[:, 6] < 0.0) or np.any(segment[:, 6] > guarded.GRIPPER_MAX_M):
            rejects.append("gripper_out_of_range")
    out = {
        "name": name,
        "phase": phase,
        "loop_index": int(loop_index),
        "segment_len": int(len(segment)),
        "duration_s": round(float(duration_s), 4),
        "trimmed_points": int(trimmed_points),
        "entry_gap_mm": round(float(entry_mm), 3),
        "max_xyz_step_mm": round(float(max_xyz), 3),
        "max_grip_step_mm": round(float(max_grip), 3),
        "guard_rejects": rejects,
        "passed_host_guards": len(rejects) == 0,
    }
    if extra:
        out.update(extra)
    return out


def build_and_validate_fast_cycle_sdk_bundle(
    *,
    smolvla_root: Path,
    manifest: dict[str, Any],
    checkpoint: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build 13-case SDK dry-run bundle for Option A Home->Hover, R2 aerial approach, R8 trimmed release, and R8/R2 direct transport."""
    fast_args = _fast_cycle_runtime_args()
    win0 = manifest["windows"][0]
    home_q = np.asarray(win0["state_values"][:6], dtype=float)
    home_grip = float(win0["state_values"][6])
    home_pose7 = np.r_[
        np.asarray(win0["right_ee_position_m"], dtype=float).reshape(3),
        np.asarray(win0["right_ee_euler_rad"], dtype=float).reshape(3),
        home_grip,
    ]

    cases_sdk: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []

    for cell, hover_pose in sorted(grasp_cases.GRID_HOVER_POSES.items()):
        takeoff_seg = grasp_cases.condition_home_takeoff_actions(
            home_pose7,
            hover_pose=hover_pose,
            hover_z_m=fast_args.hover_z_m,
            open_grip=home_grip,
        )
        prefix = guarded.make_initial_prefix(home_pose7, takeoff_seg[0], fast_args.initial_bridge_steps)
        seg = np.concatenate([prefix, takeoff_seg], axis=0)[: fast_args.exec_steps]
        _, dur_s = guarded.get_phase_speed_and_duration("takeoff", len(seg), fast_args)
        case_name = f"v070_fast_takeoff_Home_to_{cell}"
        case_summaries.append(
            _evaluate_fast_cycle_case_host(
                name=case_name,
                phase="takeoff",
                loop_index=0,
                start_pose7=home_pose7,
                segment=seg,
                duration_s=dur_s,
                extra={"cell": cell, "takeoff_raw_len": int(len(takeoff_seg))},
            )
        )
        cases_sdk.append({
            "name": case_name,
            "source": f"option_a_home_takeoff_{cell}_from_w01_home",
            "window_id": f"fast_takeoff_{cell}",
            "loop_index": 0,
            "phase": "takeoff",
            "duration_s": float(dur_s),
            "margin_rad": fast_args.margin,
            "smooth_waypoints": fast_args.smooth_waypoints,
            "waypoint_smooth_alpha": fast_args.waypoint_smooth_alpha,
            "waypoint_smooth_passes": fast_args.waypoint_smooth_passes,
            "max_interior_joint_step_rad": 0.25,
            "requires_initial_prefix_selection": True,
            "initial_prefix_candidates": prefix.tolist(),
            "seed": home_q.tolist(),
            "start_action": home_pose7.tolist(),
            "actions": seg.tolist(),
            "raw_actions": takeoff_seg.tolist(),
        })

    # Case 10: R2 122628 loop_01 far-grid shortcut + idle-tail trim
    r2_dir = smolvla_root / "diagnostics/20261113_live/right7_R2_yellow_pick_place_20261113_122628_snapshot"
    r2_st = json.loads((r2_dir / "loop_01_state.json").read_text(encoding="utf-8"))
    r2_state = np.asarray(r2_st["state"], dtype=float).reshape(7)
    r2_pose7 = np.r_[r2_st["right_ee_position_m"], r2_st["right_ee_euler_rad"], float(r2_state[6])]
    r2_raw = np.asarray(np.load(r2_dir / "loop_01_raw_actions.npy"), dtype=float)
    r2_hi = grasp_cases.detect_high_cam_grid_target((r2_dir / "loop_00_cam_high.jpg").read_bytes(), target_color="yellow")
    r2_obb = grasp_cases.detect_wrist_block_obb((r2_dir / "loop_01_cam_right_wrist.jpg").read_bytes(), target_color="yellow")
    r2_seg, _, r2_reason = grasp_cases.condition_approach_actions(
        r2_pose7,
        r2_raw,
        max_position_m=fast_args.settled_bridge_max_mm / 1000.0,
        wrist_obb=r2_obb,
        hover_z_m=fast_args.hover_z_m,
        grid_hover_pose=None if r2_hi is None else r2_hi["hover_pose"],
        prefer_grid_hover_when_far=True,
        adaptive_steps=True,
    )
    r2_seg, r2_trimmed = grasp_cases.trim_segment_idle_tail(r2_seg)
    r2_aerial_floor = max(0.176, float(getattr(fast_args, "hover_z_m", 0.182)) - 0.012)
    r2_phase = "aerial_approach" if float(np.min(r2_seg[:, 2])) >= r2_aerial_floor else "approach"
    _, r2_dur_s = guarded.get_phase_speed_and_duration(r2_phase, len(r2_seg), fast_args)
    case_summaries.append(
        _evaluate_fast_cycle_case_host(
            name="v070_fast_grid_R2_122628_loop01",
            phase=r2_phase,
            loop_index=1,
            start_pose7=r2_pose7,
            segment=r2_seg,
            duration_s=r2_dur_s,
            trimmed_points=r2_trimmed,
            extra={
                "high_cam_cell": None if r2_hi is None else r2_hi.get("cell"),
                "approach_reason": str(r2_reason),
            },
        )
    )
    cases_sdk.append({
        "name": "v070_fast_grid_R2_122628_loop01",
        "source": "diagnostics/20261113_live/right7_R2_yellow_pick_place_20261113_122628_snapshot/loop_01_raw_actions.npy",
        "window_id": "fast_grid_R2_122628_loop01",
        "loop_index": 1,
        "phase": r2_phase,
        "duration_s": float(r2_dur_s),
        "margin_rad": fast_args.margin,
        "smooth_waypoints": fast_args.smooth_waypoints,
        "waypoint_smooth_alpha": fast_args.waypoint_smooth_alpha,
        "waypoint_smooth_passes": fast_args.waypoint_smooth_passes,
        "max_interior_joint_step_rad": 0.07,
        "requires_initial_prefix_selection": False,
        "initial_prefix_candidates": None,
        "seed": r2_state[:6].tolist(),
        "start_action": r2_pose7.tolist(),
        "actions": r2_seg.tolist(),
        "raw_actions": r2_raw.tolist(),
    })

    # Case 11: R8 122503 loop_05 placement + idle-tail trim
    r8_dir = smolvla_root / "diagnostics/20261113_live/right7_R8_yellow_pick_place_20261113_122503_snapshot"
    r8_st = json.loads((r8_dir / "loop_05_state.json").read_text(encoding="utf-8"))
    r8_state = np.asarray(r8_st["state"], dtype=float).reshape(7)
    r8_pose7 = np.r_[r8_st["right_ee_position_m"], r8_st["right_ee_euler_rad"], float(r8_state[6])]
    r8_raw = np.asarray(np.load(r8_dir / "loop_05_raw_actions.npy"), dtype=float)
    r8_tobb = grasp_cases.detect_wrist_target_obb((r8_dir / "loop_05_cam_right_wrist.jpg").read_bytes())
    r8_held = _find_run_held_grip(r8_dir, 5, float(r8_state[6]))
    r8_seg, _, r8_reason = grasp_cases.condition_placement_actions(
        r8_pose7,
        r8_raw,
        held_grip=r8_held,
        max_position_m=fast_args.placement_bridge_max_mm / 1000.0,
        target_obb=r8_tobb,
        trim_release_tail=True,
    )
    r8_seg, r8_trimmed = grasp_cases.trim_segment_idle_tail(r8_seg, keep_tail_hold=4, min_points=14)
    _, r8_dur_s = guarded.get_phase_speed_and_duration("place", len(r8_seg), fast_args)
    case_summaries.append(
        _evaluate_fast_cycle_case_host(
            name="v070_fast_trimmed_R8_122503_loop05",
            phase="place",
            loop_index=5,
            start_pose7=r8_pose7,
            segment=r8_seg,
            duration_s=r8_dur_s,
            trimmed_points=r8_trimmed,
            extra={
                "target_obb_present": r8_tobb is not None,
                "placement_reason": str(r8_reason),
            },
        )
    )
    cases_sdk.append({
        "name": "v070_fast_trimmed_R8_122503_loop05",
        "source": "diagnostics/20261113_live/right7_R8_yellow_pick_place_20261113_122503_snapshot/loop_05_raw_actions.npy",
        "window_id": "fast_trimmed_R8_122503_loop05",
        "loop_index": 5,
        "phase": "place",
        "duration_s": float(r8_dur_s),
        "margin_rad": fast_args.margin,
        "smooth_waypoints": fast_args.smooth_waypoints,
        "waypoint_smooth_alpha": fast_args.waypoint_smooth_alpha,
        "waypoint_smooth_passes": fast_args.waypoint_smooth_passes,
        "max_interior_joint_step_rad": 0.07,
        "requires_initial_prefix_selection": False,
        "initial_prefix_candidates": None,
        "seed": r8_state[:6].tolist(),
        "start_action": r8_pose7.tolist(),
        "actions": r8_seg.tolist(),
        "raw_actions": r8_raw.tolist(),
    })

    # Cases 12-13: Post-grasp 2-stage direct geometric transport from 20261113_193004/193254 snapshots
    live_fast_root = THIS_DIR.parent.parent / "results" / "live_fast_20261113"
    direct_transport_specs = (
        (
            "v070_fast_direct_transport_R8_193004_loop04",
            "right7_v070_fast_R8_yellow_20261113_193004_snapshot",
            "fast_direct_transport_R8_193004_loop04",
        ),
        (
            "v070_fast_direct_transport_R2_193254_loop04",
            "right7_v070_fast_R2_yellow_20261113_193254_snapshot",
            "fast_direct_transport_R2_193254_loop04",
        ),
    )
    for case_name, snap_name, window_id in direct_transport_specs:
        dt_st_path = live_fast_root / snap_name / "loop_04_state.json"
        dt_st = json.loads(dt_st_path.read_text(encoding="utf-8"))
        dt_state = np.asarray(dt_st["state"], dtype=float).reshape(7)
        dt_pose7 = np.r_[dt_st["right_ee_position_m"], dt_st["right_ee_euler_rad"], float(dt_state[6])]
        dt_held = float(dt_state[6])
        dt_seg = grasp_cases.condition_direct_transport_actions(
            dt_pose7,
            held_grip=dt_held,
            target_pose=grasp_cases.BROWN_TARGET_HOVER_POSE6,
        )
        _, dt_dur_s = guarded.get_phase_speed_and_duration("transport", len(dt_seg), fast_args)
        case_summaries.append(
            _evaluate_fast_cycle_case_host(
                name=case_name,
                phase="transport",
                loop_index=4,
                start_pose7=dt_pose7,
                segment=dt_seg,
                duration_s=dt_dur_s,
                extra={
                    "held_grip_m": round(dt_held, 4),
                    "target_hover_pose": grasp_cases.BROWN_TARGET_HOVER_POSE6.tolist(),
                },
            )
        )
        cases_sdk.append({
            "name": case_name,
            "source": f"results/live_fast_20261113/{snap_name}/loop_04_state.json",
            "window_id": window_id,
            "loop_index": 4,
            "phase": "transport",
            "duration_s": float(dt_dur_s),
            "margin_rad": fast_args.margin,
            "smooth_waypoints": fast_args.smooth_waypoints,
            "waypoint_smooth_alpha": fast_args.waypoint_smooth_alpha,
            "waypoint_smooth_passes": fast_args.waypoint_smooth_passes,
            "max_interior_joint_step_rad": 0.07,
            "requires_initial_prefix_selection": False,
            "initial_prefix_candidates": None,
            "seed": dt_state[:6].tolist(),
            "start_action": dt_pose7.tolist(),
            "actions": dt_seg.tolist(),
            "raw_actions": dt_seg.tolist(),
        })

    bundle = {
        "schema": g4_sdk.BUNDLE_SCHEMA,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": str(checkpoint),
        "manifest_id": f"{manifest['manifest_id']}__fast_cycle_v070",
        "hardware_connected": False,
        "cases": cases_sdk,
    }
    audit = {
        "cases_total": len(cases_sdk),
        "cases_passed_host_guards": sum(1 for c in case_summaries if c["passed_host_guards"]),
        "all_passed": all(c["passed_host_guards"] for c in case_summaries),
        "cases": case_summaries,
    }
    return audit, bundle


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.port == g3.RESERVED_PRODUCTION_PORT:
        raise SystemExit(f"Refusing reserved production port {args.port}")

    env_identity = verify_env_identity(
        expected_prefix=Path(os.environ["EXPECTED_CONDA_PREFIX"]) if os.environ.get("EXPECTED_CONDA_PREFIX") else None,
        expected_lerobot_version="0.7.0",
        require_site_packages=True,
    )
    archive_audit = audit_archived_new9_vendor_trajectories(args.smolvla_root)
    manifest = load_replay_manifest(args.manifest)
    g0_replay, sdk_bundle = replay_g0_through_candidate_adapter(
        smolvla_root=args.smolvla_root,
        manifest_path=args.manifest,
        checkpoint=args.checkpoint,
        server_script=args.server_script,
        host=args.host,
        port=args.port,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    bundle_path = args.output_dir / "g4_sdk_handoff_bundle.json"
    if bundle_path.exists():
        try:
            old_bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            if (
                old_bundle.get("cases") == sdk_bundle.get("cases")
                and old_bundle.get("baseline_reference_cases") == sdk_bundle.get("baseline_reference_cases")
                and old_bundle.get("created_at_utc")
            ):
                sdk_bundle["created_at_utc"] = old_bundle["created_at_utc"]
        except Exception:
            pass
    bundle_path.write_text(json.dumps(sdk_bundle, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    bundle_sha256 = sha256_file(bundle_path)

    fast_audit, fast_bundle = build_and_validate_fast_cycle_sdk_bundle(
        smolvla_root=args.smolvla_root,
        manifest=manifest,
        checkpoint=args.checkpoint,
    )
    fast_bundle_path = args.output_dir / "g4_fast_cycle_sdk_bundle.json"
    if fast_bundle_path.exists():
        try:
            old_fast = json.loads(fast_bundle_path.read_text(encoding="utf-8"))
            if old_fast.get("cases") == fast_bundle.get("cases") and old_fast.get("created_at_utc"):
                fast_bundle["created_at_utc"] = old_fast["created_at_utc"]
        except Exception:
            pass
    fast_bundle_path.write_text(json.dumps(fast_bundle, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fast_bundle_sha256 = sha256_file(fast_bundle_path)

    lifecycle_audit = run_lifecycle_audit_suite()

    robot_report_path = args.output_dir / "g4_sdk_dryrun_report_robot.json"
    sdk_verified = False
    if robot_report_path.is_file():
        try:
            robot_report = json.loads(robot_report_path.read_text(encoding="utf-8"))
            sdk_verified = bool(
                robot_report.get("all_passed") is True
                and robot_report.get("cleanup_returned") is True
                and robot_report.get("input_sha256") == bundle_sha256
            )
        except Exception:
            sdk_verified = False

    fast_robot_report_path = args.output_dir / "g4_fast_cycle_sdk_dryrun_robot.json"
    fast_sdk_verified = False
    if fast_robot_report_path.is_file():
        try:
            fast_robot_report = json.loads(fast_robot_report_path.read_text(encoding="utf-8"))
            fast_sdk_verified = bool(
                fast_robot_report.get("all_passed") is True
                and fast_robot_report.get("cleanup_returned") is True
                and fast_robot_report.get("input_sha256") == fast_bundle_sha256
            )
        except Exception:
            fast_sdk_verified = False

    host_checks_passed = (
        archive_audit["passed"]
        and g0_replay["all_passed_host_guards_v070"]
        and fast_audit["all_passed"]
        and lifecycle_audit["all_passed"]
    )
    if not host_checks_passed:
        status = "HOST_GATES_FAILED"
    elif sdk_verified and fast_sdk_verified:
        status = "G4_SDK_DRYRUN_PASSED_SAFE_HOLD_PENDING"
    elif sdk_verified:
        status = "HOST_GATES_PASSED_FAST_CYCLE_SDK_DRYRUN_PENDING"
    else:
        status = "HOST_GATES_PASSED_ROBOT_SDK_DRYRUN_PENDING"

    report = {
        "gate": "G4_execution_adapter_and_replay",
        "status": status,
        "hardware_connected": False,
        "vendor_sdk_present_on_host": False,
        "robot_sdk_ik_planner_verified_for_v070_outputs": sdk_verified,
        "fast_cycle_sdk_dryrun_verified_on_robot": fast_sdk_verified,
        "robot_safe_hold_verified_on_hardware": False,
        "env_identity": env_identity,
        "candidate_guarded_sha256": sha256_file(CANDIDATE_DIR / "station_lite_smolvla_right7_guarded.py"),
        "candidate_build_grasp_replay_cases_sha256": sha256_file(CANDIDATE_DIR / "build_grasp_replay_cases.py"),
        "candidate_server_sha256": sha256_file(args.server_script),
        "g4_sdk_handoff_bundle": str(bundle_path),
        "g4_sdk_handoff_bundle_sha256": bundle_sha256,
        "g4_fast_cycle_sdk_bundle": str(fast_bundle_path),
        "g4_fast_cycle_sdk_bundle_sha256": fast_bundle_sha256,
        "host_checks_passed": bool(host_checks_passed),
        "archived_new9_vendor_trajectory_audit": archive_audit,
        "live_g0_candidate_adapter_replay": g0_replay,
        "fast_cycle_bundle_audit": fast_audit,
        "lifecycle_fault_injection_audit": lifecycle_audit,
        "next_robot_command_unshare_net": (
            "sudo unshare --net $ROBOT_PYTHON "
            "g4_sdk_dryrun_gate.py --input g4_sdk_handoff_bundle.json "
            "--output g4_sdk_dryrun_report_robot.json"
        ),
        "next_fast_cycle_robot_command_unshare_net": (
            "sudo unshare --net $ROBOT_PYTHON "
            "g4_sdk_dryrun_gate.py --input g4_fast_cycle_sdk_bundle.json "
            "--output g4_fast_cycle_sdk_dryrun_robot.json"
        ),
    }
    report_path = args.output_dir / "g4_execution_adapter_report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "gate": report["gate"],
        "status": report["status"],
        "host_checks_passed": report["host_checks_passed"],
        "archive_dry_runs": archive_audit["controller_dry_run_plans_checked"],
        "archive_waypoints": archive_audit["aligned_waypoints_checked"],
        "g0_windows_passed_v070": f"{sum(w['passed_host_guards_v070'] for w in g0_replay['windows'])}/{g0_replay['windows_evaluated']}",
        "fast_cycle_cases_passed": f"{fast_audit['cases_passed_host_guards']}/{fast_audit['cases_total']}",
        "lifecycle_scenarios": f"{lifecycle_audit['scenarios_passed']}/{lifecycle_audit['scenarios_total']}",
        "report_path": str(report_path),
    }, indent=2), flush=True)
    return 0 if host_checks_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
