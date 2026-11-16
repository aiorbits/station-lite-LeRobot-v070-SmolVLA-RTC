"""Stage D2 guarded recovery & inter-object trajectories + SDK dry-run bundle builder."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .candidate_import import import_build_grasp_replay_cases, import_right7_guarded
from .perception_adapter import compute_brown_pad_slot_pose

_grasp_cases = import_build_grasp_replay_cases()
_guarded = import_right7_guarded()

GRID_HOVER_POSES: dict[str, np.ndarray] = _grasp_cases.GRID_HOVER_POSES
BROWN_TARGET_HOVER_POSE6: np.ndarray = np.asarray(_grasp_cases.BROWN_TARGET_HOVER_POSE6, dtype=float)
# South-of-grid high observation pose clear of R1..R9 sightlines and >= min_safe_home_z_m (0.175 m).
CLEAR_OBS_POSE6: np.ndarray = np.array([0.200, 0.015, 0.205, -0.3317, 1.0418, 1.0361], dtype=float)
GRIPPER_MAX_M = float(_guarded.GRIPPER_MAX_M)
SDK_BUNDLE_SCHEMA = "v070_g4_sdk_handoff_bundle_v1"

# Calibrated robot-measured states recorded from live right7 snapshots (FK-exact seeds).
ANCHOR_SEED_R8_GRASP = {
    "source": "diagnostics/20261113_live/right7_R8_yellow_pick_place_20261113_122503_snapshot/loop_04_state.json",
    "q": [0.5495155260547807, 1.4051651789120323, -0.7043945983062488, 0.39311055161364156, 0.23823147936217343, -0.7600900282291906],
    "pose6": [0.2249, 0.1448, 0.1622, -0.3317, 1.0418, 1.0361],
    "cell": "R8",
}

ANCHOR_SEED_R2_GRASP = {
    "source": "diagnostics/20261113_live/right7_R2_yellow_pick_place_20261113_122628_snapshot/loop_07_state.json",
    "q": [0.41752498664835713, 1.6783016708628973, -0.9721904325932701, 0.39272907606622415, 0.2866788738841844, -0.9081025406271453],
    "pose6": [0.3063, 0.1438, 0.1631, -0.4022, 1.0243, 0.9928],
    "cell": "R2",
}

ANCHOR_SEED_BROWN_LOW = {
    "source": "diagnostics/20261113_live/right7_R2_yellow_pick_place_20261113_122628_snapshot/loop_08_state.json",
    "q": [0.5140383001449617, 1.8617914091706724, -1.177424277103839, 0.17376211184862989, 0.25654230563820907, -0.8035782406347752],
    "pose6": [0.3534, 0.2070, 0.1718, -0.5180, 0.8210, 0.8956],
}

ANCHOR_SEED_BROWN_HOVER = {
    "source": "diagnostics/20261113_live/right7_R8_yellow_pick_place_20261113_122503_snapshot/loop_05_state.json",
    "q": [0.586137178606852, 1.7027161058976115, -1.0473411154344996, 0.10090028229190473, 0.14019226367589788, -0.6013962005035484],
    "pose6": [0.3145, 0.2131, 0.1943, -0.4703, 0.7471, 0.7778],
}


def validate_cartesian_segment(
    start_pose7,
    segment,
    *,
    loop_index: int = 1,
    max_step_m: float = 0.015,
    max_first_step_m: float | None = None,
    max_grip_step_m: float = 0.015,
    max_rot_step_rad: float = np.radians(8.0),
) -> dict[str, Any]:
    """Verify host Cartesian/gripper safety invariants matching g4_sdk_dryrun_gate."""
    start = np.asarray(start_pose7, dtype=float).reshape(7)
    arr = np.asarray(segment, dtype=float)
    if not np.isfinite(start).all():
        raise ValueError("Non-finite start_pose7")
    if arr.ndim != 2 or len(arr) < 2 or arr.shape[1] != 7 or not np.isfinite(arr).all():
        raise ValueError(f"Invalid segment shape or non-finite values: {arr.shape}")
    if np.any(arr[:, 6] < 0.0) or np.any(arr[:, 6] > GRIPPER_MAX_M):
        raise ValueError("Gripper target out of calibrated bounds [0.0, 0.085]")

    chain = np.vstack([start, arr])
    xyz_steps = np.linalg.norm(np.diff(chain[:, :3], axis=0), axis=1)
    grip_steps = np.abs(np.diff(chain[:, 6]))
    rot_diff = (np.diff(chain[:, 3:6], axis=0) + np.pi) % (2.0 * np.pi) - np.pi
    max_rot = float(np.max(np.abs(rot_diff)))

    first_limit = max_first_step_m if max_first_step_m is not None else (0.025 if loop_index == 0 else max_step_m)
    if float(xyz_steps[0]) > first_limit + 1e-6:
        raise ValueError(f"First XYZ gap {xyz_steps[0]*1000:.2f} mm > {first_limit*1000:.1f} mm")
    if len(xyz_steps) > 1 and float(np.max(xyz_steps[1:])) > max_step_m + 1e-6:
        raise ValueError(f"Interior XYZ step {np.max(xyz_steps[1:])*1000:.2f} mm > {max_step_m*1000:.1f} mm")
    if float(np.max(grip_steps)) > max_grip_step_m + 1e-6:
        raise ValueError(f"Gripper step {np.max(grip_steps)*1000:.2f} mm > {max_grip_step_m*1000:.1f} mm")
    if max_rot > max_rot_step_rad + 1e-6:
        raise ValueError(f"Rotation step {math.degrees(max_rot):.2f} deg > {math.degrees(max_rot_step_rad):.1f} deg")

    return {
        "points": int(len(arr)),
        "first_xyz_gap_mm": round(float(xyz_steps[0] * 1000.0), 3),
        "max_xyz_step_mm": round(float(np.max(xyz_steps) * 1000.0), 3),
        "max_grip_step_mm": round(float(np.max(grip_steps) * 1000.0), 3),
        "max_rot_step_deg": round(float(math.degrees(max_rot)), 3),
        "min_z_m": round(float(np.min(arr[:, 2])), 4),
        "max_z_m": round(float(np.max(arr[:, 2])), 4),
    }


def _interpolate_vertical_lift_then_transit(
    start_pose7,
    *,
    goal_pose6,
    target_grip: float,
    clear_z_m: float,
    ramp_grip_during_lift: bool,
    hold_tail_steps: int = 3,
    max_xyz_step_m: float = 0.009,
    max_grip_step_m: float = 0.010,
    max_rot_step_rad: float = np.radians(3.5),
    max_steps: int = 64,
) -> np.ndarray:
    """Shared 2-stage collision-clearing trajectory helper: vertical lift first, then aerial cosine transit."""
    start = np.asarray(start_pose7, dtype=float).reshape(7)
    goal6 = np.asarray(goal_pose6, dtype=float).reshape(6)
    if not np.isfinite(start).all() or not np.isfinite(goal6).all():
        raise ValueError("Invalid non-finite pose inputs")

    eff_grip = float(np.clip(target_grip, 0.0, 0.080))
    eff_clear_z = max(float(clear_z_m), float(start[2]), float(goal6[2]))
    points: list[np.ndarray] = []

    # Stage 1: pure vertical lift keeping XY and wrist orientation locked.
    lift_end = start.copy()
    lift_end[2] = eff_clear_z
    lift_end[6] = eff_grip if ramp_grip_during_lift else float(start[6])
    lift_dz = max(0.0, float(lift_end[2]) - float(start[2]))
    grip_delta = abs(float(lift_end[6]) - float(start[6]))

    if lift_dz > 1e-4 or (ramp_grip_during_lift and grip_delta > 1e-4):
        n_lift = max(
            4,
            math.ceil(1.62 * max(lift_dz, 0.012) / max(1e-4, float(max_xyz_step_m))),
            math.ceil(grip_delta / max(1e-4, float(max_grip_step_m))) if ramp_grip_during_lift else 1,
        )
        for k in range(1, n_lift + 1):
            u = float(k) / float(n_lift)
            w = 0.5 * (1.0 - math.cos(math.pi * u))
            pt = start.copy()
            pt[2] = float(start[2]) + w * lift_dz
            if ramp_grip_during_lift:
                pt[6] = float(start[6]) + u * (eff_grip - float(start[6]))
            points.append(pt)

    # Stage 2: lateral + orientation transit staying >= eff_clear_z over initial 55% of motion.
    d_xy = goal6[:2] - lift_end[:2]
    d_xy_norm = float(np.linalg.norm(d_xy))
    d_rot = (goal6[3:6] - lift_end[3:6] + np.pi) % (2.0 * np.pi) - np.pi
    d_rot_max = float(np.max(np.abs(d_rot)))
    dz_to_goal = abs(float(goal6[2]) - float(lift_end[2]))
    rem_grip = abs(eff_grip - float(lift_end[6]))

    n_transit = max(
        4,
        math.ceil(1.62 * d_xy_norm / max(1e-4, float(max_xyz_step_m))),
        math.ceil(2.95 * dz_to_goal / max(1e-4, float(max_xyz_step_m))),
        math.ceil(1.62 * d_rot_max / max(1e-4, float(max_rot_step_rad))),
        math.ceil(rem_grip / max(1e-4, float(max_grip_step_m))),
    )
    for k in range(1, n_transit + 1):
        u = float(k) / float(n_transit)
        w_xy = 0.5 * (1.0 - math.cos(math.pi * u))
        pt = np.empty(7, dtype=float)
        pt[:2] = lift_end[:2] + w_xy * d_xy
        if u <= 0.55:
            pt[2] = float(lift_end[2])
        else:
            u_desc = (u - 0.55) / 0.45
            w_desc = 0.5 * (1.0 - math.cos(math.pi * u_desc))
            pt[2] = float(lift_end[2]) + w_desc * (float(goal6[2]) - float(lift_end[2]))
        pt[3:6] = lift_end[3:6] + w_xy * d_rot
        pt[6] = float(lift_end[6]) + min(1.0, float(k) / max(1, min(4, n_transit))) * (eff_grip - float(lift_end[6]))
        points.append(pt)

    goal7 = np.r_[goal6, eff_grip]
    for _ in range(max(1, int(hold_tail_steps)) - 1):
        points.append(goal7.copy())

    arr = np.asarray(points, dtype=float)
    if len(arr) > int(max_steps):
        raise ValueError(f"Generated segment length {len(arr)} exceeds max_steps={max_steps}")
    validate_cartesian_segment(start, arr, loop_index=1)
    return arr


def plan_empty_grasp_retreat(
    start_pose7,
    *,
    target_cell: str,
    clear_z_m: float = 0.206,
    hover_z_m: float = 0.200,
    open_grip_m: float = 0.075,
) -> np.ndarray:
    """Verified safe retract after EMPTY_GRASP_CONFIRMED: lift cleanly + open, then hover above cell."""
    if target_cell not in GRID_HOVER_POSES:
        raise ValueError(f"Unknown target grid cell: {target_cell}")
    goal = GRID_HOVER_POSES[target_cell].copy()
    goal[2] = max(float(hover_z_m), float(goal[2]))
    return _interpolate_vertical_lift_then_transit(
        start_pose7,
        goal_pose6=goal,
        target_grip=open_grip_m,
        clear_z_m=clear_z_m,
        ramp_grip_during_lift=True,
    )


def plan_vertical_lift_in_place(
    start_pose7,
    *,
    target_z_m: float = 0.200,
    target_grip_m: float | None = None,
    max_xyz_step_m: float = 0.008,
    hold_tail_steps: int = 2,
) -> np.ndarray:
    """Lift vertically in place keeping XY and wrist orientation locked for check/clear steps."""
    start = np.asarray(start_pose7, dtype=float).reshape(7)
    if not np.isfinite(start).all():
        raise ValueError("Invalid non-finite start_pose7 for vertical lift")
    eff_z = max(float(target_z_m), float(start[2]))
    eff_grip = float(np.clip(start[6] if target_grip_m is None else target_grip_m, 0.0, 0.080))
    dz = max(0.0, eff_z - float(start[2]))
    n_lift = max(6, math.ceil(1.62 * max(dz, 0.015) / max(1e-4, float(max_xyz_step_m))))
    points: list[np.ndarray] = []
    for k in range(1, n_lift + 1):
        u = float(k) / float(n_lift)
        w = 0.5 * (1.0 - math.cos(math.pi * u))
        pt = start.copy()
        pt[2] = float(start[2]) + w * dz
        pt[6] = float(start[6]) + u * (eff_grip - float(start[6]))
        points.append(pt)
    for _ in range(max(0, int(hold_tail_steps) - 1)):
        points.append(points[-1].copy())
    arr = np.asarray(points, dtype=float)
    validate_cartesian_segment(start, arr, loop_index=1)
    return arr


def _plan_grid_hover_transit(
    start_pose7,
    *,
    target_cell: str,
    clear_z_m: float,
    hover_z_m: float,
    open_grip_m: float,
) -> np.ndarray:
    if target_cell not in GRID_HOVER_POSES:
        raise ValueError(f"Unknown destination grid cell: {target_cell}")
    goal = GRID_HOVER_POSES[target_cell].copy()
    goal[2] = max(float(hover_z_m), float(goal[2]))
    return _interpolate_vertical_lift_then_transit(
        start_pose7,
        goal_pose6=goal,
        target_grip=open_grip_m,
        clear_z_m=clear_z_m,
        ramp_grip_during_lift=True,
    )


def plan_brown_to_next_grid_hover(
    start_pose7,
    *,
    next_cell: str,
    clear_z_m: float = 0.210,
    hover_z_m: float = 0.200,
    open_grip_m: float = 0.072,
) -> np.ndarray:
    """Verified inter-object transit: vertical lift clear of placed blocks, then fly to next grid hover."""
    return _plan_grid_hover_transit(
        start_pose7,
        target_cell=next_cell,
        clear_z_m=clear_z_m,
        hover_z_m=hover_z_m,
        open_grip_m=open_grip_m,
    )


def plan_brown_to_clear_obs(
    start_pose7,
    *,
    clear_z_m: float = 0.210,
    open_grip_m: float = 0.068,
) -> np.ndarray:
    """Lift vertically from brown pad and retract south of R1..R9 so cam_high sees all grid cells."""
    return _interpolate_vertical_lift_then_transit(
        start_pose7,
        goal_pose6=CLEAR_OBS_POSE6,
        target_grip=open_grip_m,
        clear_z_m=clear_z_m,
        ramp_grip_during_lift=True,
        max_steps=68,
    )


def plan_clear_obs_to_grid_hover(
    start_pose7,
    *,
    target_cell: str,
    clear_z_m: float = 0.208,
    hover_z_m: float = 0.200,
    open_grip_m: float = 0.072,
) -> np.ndarray:
    """Transit from clear observation pose above south edge to next grid cell hover."""
    return _plan_grid_hover_transit(
        start_pose7,
        target_cell=target_cell,
        clear_z_m=clear_z_m,
        hover_z_m=hover_z_m,
        open_grip_m=open_grip_m,
    )


def plan_post_grasp_slot_transport(
    start_pose7,
    *,
    held_grip_m: float,
    slot_index: int = 0,
    preload_m: float = 0.0125,
) -> np.ndarray:
    """Route grasped woodblock to a non-overlapping brown-pad slot using Scheme 1 direct transport."""
    slot_pose = compute_brown_pad_slot_pose(slot_index)
    seg = _grasp_cases.condition_direct_transport_actions(
        start_pose7,
        held_grip=held_grip_m,
        target_pose=slot_pose,
        preload_m=float(preload_m),
    )
    validate_cartesian_segment(start_pose7, seg, loop_index=4)
    return seg


def _make_sdk_case(
    *,
    name: str,
    source: str,
    loop_index: int,
    phase: str,
    seed: list[float],
    start_action: np.ndarray,
    segment: np.ndarray,
) -> dict[str, Any]:
    """Serialize one validated host segment into a g4_sdk_dryrun_gate bundle entry."""
    dur_s = round(len(segment) / (30.0 * 0.30), 4)
    return {
        "name": name,
        "source": source,
        "window_id": name,
        "loop_index": loop_index,
        "phase": phase,
        "duration_s": float(dur_s),
        "margin_rad": 0.02,
        "smooth_waypoints": True,
        "waypoint_smooth_alpha": 0.20,
        "waypoint_smooth_passes": 1,
        "max_interior_joint_step_rad": 0.07,
        "requires_initial_prefix_selection": False,
        "initial_prefix_candidates": None,
        "seed": list(seed),
        "start_action": start_action.tolist(),
        "actions": segment.tolist(),
        "raw_actions": segment.tolist(),
    }


def build_demo_d2_sdk_dryrun_bundle(output_path: Path | None = None) -> dict[str, Any]:
    """Build v070_g4_sdk_handoff_bundle_v1 covering D2 recovery and multi-object transitions."""
    cases_sdk: list[dict[str, Any]] = []

    # 1. Empty-grasp recovery cases from measured low tabletop poses
    for anchor in (ANCHOR_SEED_R8_GRASP, ANCHOR_SEED_R2_GRASP):
        start7 = np.r_[anchor["pose6"], 0.008]  # Empty closed gripper after failed grasp
        seg = plan_empty_grasp_retreat(start7, target_cell=anchor["cell"])
        cases_sdk.append(
            _make_sdk_case(
                name=f"d2_empty_grasp_retreat_{anchor['cell']}",
                source=anchor["source"],
                loop_index=3,
                phase="takeoff",
                seed=anchor["q"],
                start_action=start7,
                segment=seg,
            )
        )

    # 2. Multi-object inter-task transit: Brown Zone -> R1, R2, R4, R7, R8, R9
    brown_start7 = np.r_[ANCHOR_SEED_BROWN_LOW["pose6"], 0.0686]
    for cell in ("R1", "R2", "R4", "R7", "R8", "R9"):
        seg = plan_brown_to_next_grid_hover(brown_start7, next_cell=cell)
        cases_sdk.append(
            _make_sdk_case(
                name=f"d2_brown_to_{cell}_hover",
                source=ANCHOR_SEED_BROWN_LOW["source"],
                loop_index=6,
                phase="transport",
                seed=ANCHOR_SEED_BROWN_LOW["q"],
                start_action=brown_start7,
                segment=seg,
            )
        )

    # 3. Direct transport to multi-block brown pad slots 1 & 2 (non-overlapping offsets)
    r8_held7 = np.r_[ANCHOR_SEED_R8_GRASP["pose6"], 0.0280]
    for slot_idx in (1, 2):
        seg = plan_post_grasp_slot_transport(r8_held7, held_grip_m=0.0280, slot_index=slot_idx)
        cases_sdk.append(
            _make_sdk_case(
                name=f"d2_r8_to_brown_slot_{slot_idx}",
                source=ANCHOR_SEED_R8_GRASP["source"],
                loop_index=4,
                phase="transport",
                seed=ANCHOR_SEED_R8_GRASP["q"],
                start_action=r8_held7,
                segment=seg,
            )
        )

    bundle = {
        "schema": SDK_BUNDLE_SCHEMA,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifest_id": "demo_d2_recovery_and_multiobject_transitions_v1",
        "hardware_connected": False,
        "cases": cases_sdk,
    }
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(bundle, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return bundle


def build_demo_d3_sdk_dryrun_bundle(output_path: Path | None = None) -> dict[str, Any]:
    """Build v070_g4_sdk_handoff_bundle_v1 covering D3 vertical lifts, R3/R5/R6, and clear_obs."""
    cases_sdk: list[dict[str, Any]] = []

    # 1. Vertical post-grasp check lifts holding block (R8, R2) + post-release vertical lift (Brown)
    for anchor, grip_m in (
        (ANCHOR_SEED_R8_GRASP, 0.0280),
        (ANCHOR_SEED_R2_GRASP, 0.0280),
    ):
        start7 = np.r_[anchor["pose6"], grip_m]
        seg = plan_vertical_lift_in_place(start7, target_z_m=0.200, target_grip_m=grip_m)
        cases_sdk.append(
            _make_sdk_case(
                name=f"d3_post_grasp_lift_{anchor['cell']}",
                source=anchor["source"],
                loop_index=3,
                phase="transport",
                seed=anchor["q"],
                start_action=start7,
                segment=seg,
            )
        )

    brown_start7 = np.r_[ANCHOR_SEED_BROWN_LOW["pose6"], 0.0686]
    lift_brown = plan_vertical_lift_in_place(brown_start7, target_z_m=0.205, target_grip_m=0.0686)
    cases_sdk.append(
        _make_sdk_case(
            name="d3_post_release_lift_brown",
            source=ANCHOR_SEED_BROWN_LOW["source"],
            loop_index=6,
            phase="takeoff",
            seed=ANCHOR_SEED_BROWN_LOW["q"],
            start_action=brown_start7,
            segment=lift_brown,
        )
    )

    # 2. Remaining white-grid cells (R3, R5, R6) so R1..R9 are 100% SDK dry-run verified
    for cell in ("R3", "R5", "R6"):
        seg = plan_brown_to_next_grid_hover(brown_start7, next_cell=cell)
        cases_sdk.append(
            _make_sdk_case(
                name=f"d3_brown_to_{cell}_hover",
                source=ANCHOR_SEED_BROWN_LOW["source"],
                loop_index=6,
                phase="transport",
                seed=ANCHOR_SEED_BROWN_LOW["q"],
                start_action=brown_start7,
                segment=seg,
            )
        )

    # 3. Retract from brown pad south to clear observation pose (zero arm occlusion over R1..R9)
    seg_clear = plan_brown_to_clear_obs(brown_start7)
    cases_sdk.append(
        _make_sdk_case(
            name="d3_brown_to_clear_obs",
            source=ANCHOR_SEED_BROWN_LOW["source"],
            loop_index=6,
            phase="takeoff",
            seed=ANCHOR_SEED_BROWN_LOW["q"],
            start_action=brown_start7,
            segment=seg_clear,
        )
    )

    bundle = {
        "schema": SDK_BUNDLE_SCHEMA,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifest_id": "demo_d3_lifts_remaining_grid_and_clear_obs_v1",
        "hardware_connected": False,
        "cases": cases_sdk,
    }
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(bundle, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return bundle


def main(argv: list[str] | None = None) -> int:
    import argparse
    import hashlib

    ap = argparse.ArgumentParser(description="Generate Stage D2/D3 CAN-free SDK dry-run bundle.")
    ap.add_argument("--stage", choices=("d2", "d3"), default="d2", help="Which bundle stage to emit")
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output bundle JSON path",
    )
    args = ap.parse_args(argv)
    out_path = args.output or Path(f"bundles/g4_demo_{args.stage}_sdk_bundle.json")
    bundle = (
        build_demo_d2_sdk_dryrun_bundle(out_path)
        if args.stage == "d2"
        else build_demo_d3_sdk_dryrun_bundle(out_path)
    )
    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print(f"[{args.stage.upper()} BUNDLE] wrote {len(bundle['cases'])} cases -> {out_path} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
