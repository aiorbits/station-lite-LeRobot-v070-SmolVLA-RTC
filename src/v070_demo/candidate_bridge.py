"""Stage D2 candidate execution bridge connecting SessionService to v070_candidate guards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .candidate_import import import_build_grasp_replay_cases, import_right7_guarded
from .contracts import (
    ExecutionPhase,
    HardwareAdmissionState,
    ObjectRecord,
)
from .evidence_verifier import (
    verify_live_grasp_outcome,
    verify_live_placement_outcome,
)
from .executor_bridge import ExecutionBridge, PhaseStepResult
from .perception_adapter import (
    BROWN_TARGET_HOVER_POSE6,
    GRID_HOVER_POSES,
    compute_brown_pad_slot_pose,
    detect_all_high_cam_blocks,
)
from .trajectory_planner import (
    plan_brown_to_next_grid_hover,
    plan_empty_grasp_retreat,
    plan_post_grasp_slot_transport,
    validate_cartesian_segment,
)

_grasp_cases = import_build_grasp_replay_cases()
_guarded = import_right7_guarded()

D2_BUNDLE_SHA256 = "bf0bdc76ef776d0dbb56f5d27b1b5d2b623a232e997c7e1fe6f94f6e70475c0e"


def verify_sdk_dryrun_report(
    report_path: str | Path,
    *,
    expected_bundle_sha256: str | None = D2_BUNDLE_SHA256,
    min_cases: int = 1,
) -> dict[str, Any]:
    """Fail-closed cryptographic & schema verification of a robot-host CAN-isolated SDK dry-run report."""
    path = Path(report_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"SDK dry-run report not found: {path}")
    raw_bytes = path.read_bytes()
    report_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    data = json.loads(raw_bytes.decode("utf-8"))

    if data.get("schema") != "v070_g4_sdk_dryrun_report_v1":
        raise PermissionError(f"Unexpected dry-run report schema: {data.get('schema')!r}")
    if data.get("bundle_schema") != "v070_g4_sdk_handoff_bundle_v1":
        raise PermissionError(f"Unexpected dry-run bundle_schema: {data.get('bundle_schema')!r}")
    if data.get("hardware_connected") is not False:
        raise PermissionError(f"SDK dry-run report must assert hardware_connected=False, got {data.get('hardware_connected')!r}")
    if data.get("cleanup_returned") is not True:
        raise PermissionError(f"SDK dry-run report cleanup_returned must be True, got {data.get('cleanup_returned')!r}")
    if data.get("all_passed") is not True or int(data.get("cases_failed", -1)) != 0:
        raise PermissionError(
            f"SDK dry-run report did not pass cleanly: all_passed={data.get('all_passed')} cases_failed={data.get('cases_failed')}"
        )

    cases_total = int(data.get("cases_total", 0))
    cases_passed = int(data.get("cases_passed", 0))
    cases = data.get("cases")
    if (
        cases_total < int(min_cases)
        or cases_passed != cases_total
        or not isinstance(cases, list)
        or len(cases) != cases_total
    ):
        raise PermissionError(
            f"SDK dry-run report case count invalid: total={cases_total} passed={cases_passed} actual={len(cases) if isinstance(cases, list) else None}"
        )
    for item in cases:
        if not isinstance(item, dict) or item.get("passed") is not True or item.get("reject"):
            name = item.get("name") if isinstance(item, dict) else "<invalid>"
            raise PermissionError(f"SDK dry-run case {name!r} failed or has reject reasons")

    input_sha256 = str(data.get("input_sha256", "")).strip()
    if expected_bundle_sha256 is not None and input_sha256 != expected_bundle_sha256:
        raise PermissionError(
            f"SDK dry-run input_sha256 mismatch: expected={expected_bundle_sha256}, got={input_sha256}"
        )

    return {
        "report_path": str(path),
        "report_sha256": report_sha256,
        "input_sha256": input_sha256,
        "cases_total": cases_total,
        "cases_passed": cases_passed,
        "sdk_version": data.get("sdk_version"),
        "cleanup_returned": True,
    }


@dataclass
class GuardedBridgeConfig:
    """Explicit admission and safety configuration (never defaults to live hardware)."""

    dry_run_report_verified: bool = False
    confirm_startup_home: bool = False
    confirm_exit_home: bool = False
    operator_admitted: bool = False
    max_observation_age_ms: float = 800.0
    hover_z_m: float = 0.200
    step_delay_s: float = 0.0
    max_brown_slots: int = 3


class CandidateRuntimeState:
    """Tracks single-owner physical pose estimation, placed slots, and planned segments."""

    def __init__(self) -> None:
        self.ee_pose7 = np.array([0.1272, 0.0002, 0.1776, 0.0, 0.0, 0.0, 0.072], dtype=float)
        self.held_grip_m: float | None = None
        self.placed_slots_used = 0
        self.planned_segments: list[dict[str, Any]] = []
        self.cancelled_generations: list[tuple[int, str]] = []
        self.hold_log: list[tuple[ExecutionPhase, str]] = []
        self.recovery_segments_executed: int = 0
        self.inter_object_transitions_executed: int = 0


class GuardedCandidateBridge(ExecutionBridge):
    """Single-owner ExecutionBridge validating candidate paths and dual evidence in Stage D2.

    In D2 host/replay mode, runs against synthetic camera/image providers and validates
    real v070_candidate geometry + condition + step guards without touching robot hardware.
    Hardware admission requires explicit `admit_hardware(...)` after a passed SDK dry-run.
    """

    def __init__(
        self,
        *,
        config: GuardedBridgeConfig | None = None,
        frame_provider: Callable[[str, set[str], int], dict[str, Any]] | None = None,
        clock_fn: Callable[[], float] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self.config = config or GuardedBridgeConfig()
        self._frame_provider = frame_provider
        self._clock = clock_fn or time.monotonic
        self.state = CandidateRuntimeState()
        self._admitted = False
        self._obs_seq = 0
        self._placed_object_ids: set[str] = set()
        self.phase_hook: Callable[[ObjectRecord, ExecutionPhase], None] | None = None
        self._pending_empty_grasp_ids: set[str] = set()

    def admit_hardware(self, *, dry_run_report_sha256: str) -> None:
        """Fail-closed admission gate blocking live execution unless D2 prerequisites pass."""
        if not dry_run_report_sha256.strip():
            raise PermissionError("Hardware admission requires a non-empty D2 SDK dry-run report SHA256")
        if not self.config.dry_run_report_verified:
            raise PermissionError("Hardware admission blocked: dry_run_report_verified must be True")
        if not self.config.operator_admitted:
            raise PermissionError("Hardware admission blocked: operator_admitted must be True")
        if not (self.config.confirm_startup_home and self.config.confirm_exit_home):
            raise PermissionError(
                "Hardware admission blocked: workspace clear home gates (--confirm-startup-home/--confirm-exit-home) required"
            )
        self._admitted = True

    def admit_hardware_from_report(
        self,
        report_path: str | Path,
        *,
        expected_bundle_sha256: str | None = D2_BUNDLE_SHA256,
        min_cases: int = 10,
    ) -> dict[str, Any]:
        """Verify robot-host SDK dry-run JSON cryptographically before unlocking hardware admission."""
        verified = verify_sdk_dryrun_report(
            report_path,
            expected_bundle_sha256=expected_bundle_sha256,
            min_cases=min_cases,
        )
        self.config.dry_run_report_verified = True
        self.admit_hardware(dry_run_report_sha256=verified["input_sha256"])
        return verified

    @property
    def admission_state(self) -> HardwareAdmissionState:
        if self._admitted:
            return HardwareAdmissionState.ADMITTED_SUPERVISED
        return HardwareAdmissionState.UNADMITTED

    @property
    def publisher_count(self) -> int:
        return 1

    def begin_new_batch(self) -> None:
        """Reset per-batch software slot counters without resetting physical arm state."""
        with self._lock:
            self.state.placed_slots_used = 0
            self.state.held_grip_m = None
            self._placed_object_ids.clear()

    def set_empty_grasp_once(self, object_id: str) -> None:
        """Inject a single simulated physical empty grasp for dual-evidence verification."""
        with self._lock:
            self._pending_empty_grasp_ids.add(object_id)

    def observe_scene(self, scene_id: str, now_mono: float) -> tuple[str, list[ObjectRecord]]:
        with self._lock:
            self._obs_seq += 1
            obs_id = f"cand_obs_{self._obs_seq:04d}"
            if self._frame_provider is None:
                raise RuntimeError("No frame_provider configured for GuardedCandidateBridge")

            frame_bundle = self._frame_provider(scene_id, set(self._placed_object_ids), self._obs_seq)
            obs_ts = float(frame_bundle.get("observation_ts_mono", now_mono))
            cam_high_bgr = frame_bundle["cam_high_bgr"]
            detected = detect_all_high_cam_blocks(
                cam_high_bgr,
                observation_id=obs_id,
                observation_ts_mono=obs_ts,
            )
            return obs_id, detected

    def run_phase_step(
        self,
        *,
        obj: ObjectRecord,
        phase: ExecutionPhase,
        task_generation: int,
        attempt_number: int,
        now_mono: float,
    ) -> PhaseStepResult:
        if self.phase_hook is not None:
            self.phase_hook(obj, phase)
        if self.config.step_delay_s > 0.0:
            time.sleep(self.config.step_delay_s)

        with self._lock:
            obs_age_ms = (self._clock() - obj.observation_ts_mono) * 1000.0
            if obs_age_ms > self.config.max_observation_age_ms:
                return PhaseStepResult(
                    ok=False,
                    phase=phase,
                    task_generation=task_generation,
                    observation_id=obj.observation_id,
                    observation_ts_mono=obj.observation_ts_mono,
                    gripper_width_m=float(self.state.ee_pose7[6]),
                    ee_pose=tuple(float(v) for v in self.state.ee_pose7[:6]),
                    error_code="stale_observation",
                    error_detail=f"obs_age={obs_age_ms:.1f}ms>{self.config.max_observation_age_ms:.1f}ms",
                )

            cell = obj.grid_cell or "R8"
            hover_pose = GRID_HOVER_POSES.get(cell, GRID_HOVER_POSES["R8"])

            if phase == ExecutionPhase.APPROACHING_EMPTY:
                # Choose either Option A Home->Hover (first object from Home) or Brown->GridHover (subsequent object!)
                at_home = bool(np.linalg.norm(self.state.ee_pose7[:3] - np.array([0.1272, 0.0002, 0.1776])) < 0.03)
                if at_home:
                    seg = _grasp_cases.condition_home_takeoff_actions(
                        self.state.ee_pose7,
                        hover_pose=hover_pose,
                        hover_z_m=self.config.hover_z_m,
                    )
                    stats = validate_cartesian_segment(self.state.ee_pose7, seg, loop_index=0)
                    seg_kind = "home_takeoff_hover"
                else:
                    seg = plan_brown_to_next_grid_hover(
                        self.state.ee_pose7,
                        next_cell=cell,
                        hover_z_m=self.config.hover_z_m,
                    )
                    stats = validate_cartesian_segment(self.state.ee_pose7, seg, loop_index=1)
                    self.state.inter_object_transitions_executed += 1
                    seg_kind = "brown_to_next_grid_hover"
                self.state.planned_segments.append({"object_id": obj.object_id, "kind": seg_kind, **stats})
                self.state.ee_pose7 = seg[-1].copy()

            elif phase == ExecutionPhase.DESCENDING_CONTACT:
                # Descend vertically from hover to grasp tabletop approach height (~0.162 m)
                goal = np.array([hover_pose[0], hover_pose[1], 0.162, hover_pose[3], hover_pose[4], hover_pose[5]], dtype=float)
                steps = 10
                seg = np.empty((steps, 7), dtype=float)
                for k in range(1, steps + 1):
                    alpha = float(k) / float(steps)
                    seg[k - 1, :6] = self.state.ee_pose7[:6] + alpha * (goal - self.state.ee_pose7[:6])
                    seg[k - 1, 6] = self.state.ee_pose7[6]
                stats = validate_cartesian_segment(self.state.ee_pose7, seg, loop_index=1)
                self.state.planned_segments.append({"object_id": obj.object_id, "kind": "vertical_descent", **stats})
                self.state.ee_pose7 = seg[-1].copy()

            elif phase == ExecutionPhase.GRASPING:
                empty_injected = obj.object_id in self._pending_empty_grasp_ids and attempt_number == 1
                self.state.ee_pose7[6] = 0.008 if empty_injected else 0.0278

            elif phase == ExecutionPhase.LIFTING_CHECK:
                assert self._frame_provider is not None
                measured_grip = float(self.state.ee_pose7[6])
                empty_injected = obj.object_id in self._pending_empty_grasp_ids and attempt_number == 1
                if empty_injected:
                    self._pending_empty_grasp_ids.discard(obj.object_id)
                post_lift_frames = self._frame_provider(
                    f"post_lift:{obj.object_id}:{'empty' if empty_injected else 'held'}",
                    set(self._placed_object_ids),
                    self._obs_seq,
                )
                verdict = verify_live_grasp_outcome(
                    measured_grip,
                    cam_high_post_lift=post_lift_frames.get("cam_high_bgr"),
                    wrist_post_lift=post_lift_frames.get("wrist_bgr"),
                    source_cell=cell,
                    target_color=obj.color,
                )
                if verdict.verdict == "HELD_CONFIRMED":
                    self.state.held_grip_m = measured_grip
                return PhaseStepResult(
                    ok=True,
                    phase=phase,
                    task_generation=task_generation,
                    observation_id=obj.observation_id,
                    observation_ts_mono=self._clock(),
                    gripper_width_m=measured_grip,
                    ee_pose=tuple(float(v) for v in self.state.ee_pose7[:6]),
                    visual_block_on_table=verdict.evidence.get("visual_block_still_at_cell"),
                )

            elif phase == ExecutionPhase.TRANSPORTING:
                held_grip = float(self.state.held_grip_m or self.state.ee_pose7[6])
                slot_idx = self.state.placed_slots_used
                if slot_idx >= self.config.max_brown_slots:
                    return PhaseStepResult(
                        ok=False,
                        phase=phase,
                        task_generation=task_generation,
                        observation_id=obj.observation_id,
                        observation_ts_mono=self._clock(),
                        gripper_width_m=held_grip,
                        ee_pose=tuple(float(v) for v in self.state.ee_pose7[:6]),
                        error_code="brown_pad_capacity_exceeded",
                        error_detail=f"Already placed {slot_idx} objects on brown target",
                    )
                obj.destination_slot = slot_idx
                seg = plan_post_grasp_slot_transport(
                    self.state.ee_pose7,
                    held_grip_m=held_grip,
                    slot_index=slot_idx,
                )
                stats = validate_cartesian_segment(self.state.ee_pose7, seg, loop_index=4)
                self.state.planned_segments.append({"object_id": obj.object_id, "kind": f"direct_transport_slot_{slot_idx}", **stats})
                self.state.ee_pose7 = seg[-1].copy()
                return PhaseStepResult(
                    ok=True,
                    phase=phase,
                    task_generation=task_generation,
                    observation_id=obj.observation_id,
                    observation_ts_mono=self._clock(),
                    gripper_width_m=float(self.state.ee_pose7[6]),
                    ee_pose=tuple(float(v) for v in self.state.ee_pose7[:6]),
                    destination_slot=slot_idx,
                )

            elif phase == ExecutionPhase.PLACING:
                # Lower gently to placement contact (~0.153 m) and release smoothly -> 0.068 m
                slot_pose = compute_brown_pad_slot_pose(obj.destination_slot or 0)
                place_end = np.array([slot_pose[0], slot_pose[1], 0.153, slot_pose[3], slot_pose[4], slot_pose[5], 0.068], dtype=float)
                steps = 10
                seg = np.empty((steps, 7), dtype=float)
                for k in range(1, steps + 1):
                    alpha = float(k) / float(steps)
                    seg[k - 1, :6] = self.state.ee_pose7[:6] + alpha * (place_end[:6] - self.state.ee_pose7[:6])
                    seg[k - 1, 6] = self.state.ee_pose7[6] + alpha * (place_end[6] - self.state.ee_pose7[6])
                stats = validate_cartesian_segment(self.state.ee_pose7, seg, loop_index=5)
                self.state.planned_segments.append({"object_id": obj.object_id, "kind": "slot_place_release", **stats})
                self.state.ee_pose7 = seg[-1].copy()

            elif phase == ExecutionPhase.RELEASING_CHECK:
                assert self._frame_provider is not None
                held_grip = float(self.state.held_grip_m or 0.0278)
                post_release = self._frame_provider(
                    f"post_release:{obj.object_id}",
                    set(self._placed_object_ids) | {obj.object_id},
                    self._obs_seq,
                )
                verdict = verify_live_placement_outcome(
                    float(self.state.ee_pose7[6]),
                    held_grip_m=held_grip,
                    wrist_post_release=post_release.get("wrist_bgr"),
                    target_color=obj.color,
                )
                used_slot = obj.destination_slot if obj.destination_slot is not None else self.state.placed_slots_used
                if verdict.confirmed:
                    self._placed_object_ids.add(obj.object_id)
                    self.state.placed_slots_used += 1
                    self.state.held_grip_m = None
                return PhaseStepResult(
                    ok=True,
                    phase=phase,
                    task_generation=task_generation,
                    observation_id=obj.observation_id,
                    observation_ts_mono=self._clock(),
                    gripper_width_m=float(self.state.ee_pose7[6]),
                    ee_pose=tuple(float(v) for v in self.state.ee_pose7[:6]),
                    visual_block_in_brown_zone=bool(verdict.evidence.get("visual_confirmed", False) and verdict.confirmed),
                    visual_block_at_gripper=bool(verdict.evidence.get("still_clutched_near_fingers", False)),
                    destination_slot=used_slot,
                )

            return PhaseStepResult(
                ok=True,
                phase=phase,
                task_generation=task_generation,
                observation_id=obj.observation_id,
                observation_ts_mono=self._clock(),
                gripper_width_m=float(self.state.ee_pose7[6]),
                ee_pose=tuple(float(v) for v in self.state.ee_pose7[:6]),
            )

    def run_empty_grasp_recovery(
        self,
        *,
        obj: ObjectRecord,
        task_generation: int,
        now_mono: float,
    ) -> PhaseStepResult:
        """Execute verified vertical lift + gripper open back to hover cell on EMPTY_GRASP_CONFIRMED."""
        with self._lock:
            cell = obj.grid_cell or "R8"
            seg = plan_empty_grasp_retreat(self.state.ee_pose7, target_cell=cell)
            stats = validate_cartesian_segment(self.state.ee_pose7, seg, loop_index=3)
            self.state.planned_segments.append({"object_id": obj.object_id, "kind": "empty_grasp_recovery_retreat", **stats})
            self.state.ee_pose7 = seg[-1].copy()
            self.state.held_grip_m = None
            self.state.recovery_segments_executed += 1
            return PhaseStepResult(
                ok=True,
                phase=ExecutionPhase.RECOVERING,
                task_generation=task_generation,
                observation_id=obj.observation_id,
                observation_ts_mono=self._clock(),
                gripper_width_m=float(self.state.ee_pose7[6]),
                ee_pose=tuple(float(v) for v in self.state.ee_pose7[:6]),
            )

    def request_phase_hold(self, *, current_phase: ExecutionPhase, reason: str) -> bool:
        with self._lock:
            self.state.hold_log.append((current_phase, reason))
            return True

    def cancel_future_queue(self, *, generation: int, reason: str) -> int:
        with self._lock:
            self.state.cancelled_generations.append((generation, reason))
            return 1
