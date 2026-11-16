"""Single-owner execution bridge, dual-evidence evaluation, and mock publisher."""

from __future__ import annotations

import copy
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from .contracts import (
    ContactHeldState,
    ExecutionPhase,
    HardwareAdmissionState,
    ObjectRecord,
)
from .task_parser import format_single_target_task


@dataclass
class PhaseStepResult:
    ok: bool
    phase: ExecutionPhase
    task_generation: int
    observation_id: str
    observation_ts_mono: float
    gripper_width_m: float
    ee_pose: tuple[float, float, float, float, float, float]
    visual_block_on_table: bool | None = None
    visual_block_in_brown_zone: bool | None = None
    visual_block_at_gripper: bool | None = None
    error_code: str | None = None
    error_detail: str = ""
    latency_ms: float = 0.0
    destination_slot: int | None = None


@dataclass
class ScenePreset:
    scene_id: str
    description: str
    objects: list[dict[str, Any]]


SCENE_PRESETS: dict[str, ScenePreset] = {
    "scene_1_standard": ScenePreset(
        scene_id="scene_1_standard",
        description="Three separated woodblocks (red near, yellow mid, blue far) with high clearance.",
        objects=[
            {
                "object_id": "obj_red_1",
                "color": "red",
                "pose_estimate": {"x": 0.25, "y": -0.06, "z": 0.03, "yaw_deg": 5.0},
                "confidence": 0.96,
                "occlusion_score": 0.02,
                "clearance_m": 0.095,
                "estimated_duration_s": 21.0,
            },
            {
                "object_id": "obj_yellow_1",
                "color": "yellow",
                "pose_estimate": {"x": 0.27, "y": 0.03, "z": 0.03, "yaw_deg": 18.0},
                "confidence": 0.95,
                "occlusion_score": 0.03,
                "clearance_m": 0.090,
                "estimated_duration_s": 22.0,
            },
            {
                "object_id": "obj_blue_1",
                "color": "blue",
                "pose_estimate": {"x": 0.21, "y": 0.09, "z": 0.03, "yaw_deg": 32.0},
                "confidence": 0.91,
                "occlusion_score": 0.06,
                "clearance_m": 0.075,
                "estimated_duration_s": 25.0,
            },
        ],
    ),
    "scene_2_mixed_distance": ScenePreset(
        scene_id="scene_2_mixed_distance",
        description="Yellow easiest near cell, red far angled, blue mid-right.",
        objects=[
            {
                "object_id": "obj_red_2",
                "color": "red",
                "pose_estimate": {"x": 0.19, "y": -0.09, "z": 0.03, "yaw_deg": 45.0},
                "confidence": 0.86,
                "occlusion_score": 0.10,
                "clearance_m": 0.060,
                "estimated_duration_s": 27.0,
            },
            {
                "object_id": "obj_yellow_2",
                "color": "yellow",
                "pose_estimate": {"x": 0.28, "y": 0.00, "z": 0.03, "yaw_deg": 0.0},
                "confidence": 0.97,
                "occlusion_score": 0.01,
                "clearance_m": 0.110,
                "estimated_duration_s": 19.5,
            },
            {
                "object_id": "obj_blue_2",
                "color": "blue",
                "pose_estimate": {"x": 0.24, "y": 0.08, "z": 0.03, "yaw_deg": 20.0},
                "confidence": 0.90,
                "occlusion_score": 0.07,
                "clearance_m": 0.080,
                "estimated_duration_s": 23.0,
            },
        ],
    ),
    "scene_3_empty_grasp_recovery": ScenePreset(
        scene_id="scene_3_empty_grasp_recovery",
        description="Tight clearance scene where blue attempt #1 triggers verified empty-grasp recovery.",
        objects=[
            {
                "object_id": "obj_blue_3",
                "color": "blue",
                "pose_estimate": {"x": 0.26, "y": 0.04, "z": 0.03, "yaw_deg": 12.0},
                "confidence": 0.95,
                "occlusion_score": 0.04,
                "clearance_m": 0.085,
                "estimated_duration_s": 21.5,
            },
            {
                "object_id": "obj_red_3",
                "color": "red",
                "pose_estimate": {"x": 0.23, "y": -0.05, "z": 0.03, "yaw_deg": 35.0},
                "confidence": 0.92,
                "occlusion_score": 0.06,
                "clearance_m": 0.070,
                "estimated_duration_s": 24.0,
            },
            {
                "object_id": "obj_yellow_3",
                "color": "yellow",
                "pose_estimate": {"x": 0.20, "y": 0.09, "z": 0.03, "yaw_deg": 10.0},
                "confidence": 0.89,
                "occlusion_score": 0.08,
                "clearance_m": 0.065,
                "estimated_duration_s": 25.5,
            },
        ],
    ),
    "scene_ambiguous_duplicate": ScenePreset(
        scene_id="scene_ambiguous_duplicate",
        description="Fault scenario containing two red blocks to verify clarification rejection.",
        objects=[
            {
                "object_id": "obj_red_a",
                "color": "red",
                "pose_estimate": {"x": 0.26, "y": -0.04, "z": 0.03, "yaw_deg": 0.0},
                "confidence": 0.94,
                "occlusion_score": 0.02,
                "clearance_m": 0.080,
                "estimated_duration_s": 22.0,
            },
            {
                "object_id": "obj_red_b",
                "color": "red",
                "pose_estimate": {"x": 0.22, "y": -0.08, "z": 0.03, "yaw_deg": 15.0},
                "confidence": 0.91,
                "occlusion_score": 0.05,
                "clearance_m": 0.070,
                "estimated_duration_s": 23.5,
            },
            {
                "object_id": "obj_yellow_a",
                "color": "yellow",
                "pose_estimate": {"x": 0.25, "y": 0.06, "z": 0.03, "yaw_deg": 0.0},
                "confidence": 0.93,
                "occlusion_score": 0.03,
                "clearance_m": 0.085,
                "estimated_duration_s": 22.5,
            },
        ],
    ),
    "scene_clear_table": ScenePreset(
        scene_id="scene_clear_table",
        description="Clear the table: continuously detect and sort white-grid blocks until empty.",
        objects=[
            {
                "object_id": "obj_red_clear_1",
                "color": "red",
                "pose_estimate": {"x": 0.25, "y": -0.06, "z": 0.03, "yaw_deg": 5.0},
                "confidence": 0.96,
                "occlusion_score": 0.02,
                "clearance_m": 0.095,
                "estimated_duration_s": 21.0,
            },
            {
                "object_id": "obj_yellow_clear_1",
                "color": "yellow",
                "pose_estimate": {"x": 0.27, "y": 0.03, "z": 0.03, "yaw_deg": 18.0},
                "confidence": 0.95,
                "occlusion_score": 0.03,
                "clearance_m": 0.090,
                "estimated_duration_s": 22.0,
            },
            {
                "object_id": "obj_blue_clear_1",
                "color": "blue",
                "pose_estimate": {"x": 0.21, "y": 0.09, "z": 0.03, "yaw_deg": 32.0},
                "confidence": 0.91,
                "occlusion_score": 0.06,
                "clearance_m": 0.075,
                "estimated_duration_s": 25.0,
            },
        ],
    ),
}


def evaluate_grasp_evidence(
    *,
    gripper_width_m: float,
    visual_block_on_table: bool | None,
    min_block_width_m: float = 0.018,
    max_block_width_m: float = 0.045,
) -> tuple[str, ContactHeldState, dict[str, Any]]:
    """Combine calibrated gripper width AND post-lift vision per Section 6.2."""
    gripper_shows_block = min_block_width_m <= gripper_width_m <= max_block_width_m
    gripper_empty_closed = gripper_width_m < min_block_width_m

    evidence = {
        "gripper_width_m": round(gripper_width_m, 4),
        "gripper_shows_block": gripper_shows_block,
        "gripper_empty_closed": gripper_empty_closed,
        "visual_block_on_table": visual_block_on_table,
    }

    if visual_block_on_table is None:
        return "HELD_UNCERTAIN", ContactHeldState.HELD_UNCERTAIN, evidence

    if gripper_shows_block and visual_block_on_table is False:
        return "HELD_CONFIRMED", ContactHeldState.HELD_CONFIRMED, evidence

    if gripper_empty_closed and visual_block_on_table is True:
        return "EMPTY_GRASP_CONFIRMED", ContactHeldState.EMPTY_CONFIRMED, evidence

    return "HELD_UNCERTAIN", ContactHeldState.HELD_UNCERTAIN, evidence


def evaluate_placement_evidence(
    *,
    gripper_width_m: float,
    visual_block_in_brown_zone: bool | None,
    visual_block_at_gripper: bool | None,
    min_released_width_m: float = 0.055,
) -> tuple[bool, ContactHeldState, dict[str, Any]]:
    """Require released gripper AND target-zone visual presence without sticky grip."""
    gripper_open = gripper_width_m >= min_released_width_m
    evidence = {
        "gripper_width_m": round(gripper_width_m, 4),
        "gripper_open": gripper_open,
        "visual_block_in_brown_zone": visual_block_in_brown_zone,
        "visual_block_at_gripper": visual_block_at_gripper,
    }

    if (
        gripper_open
        and visual_block_in_brown_zone is True
        and visual_block_at_gripper is False
    ):
        return True, ContactHeldState.EMPTY_CONFIRMED, evidence

    if visual_block_at_gripper is True:
        return False, ContactHeldState.HELD_UNCERTAIN, evidence

    return False, ContactHeldState.UNKNOWN, evidence


class ExecutionBridge(ABC):
    """Single-owner boundary between SessionService and robot/mock executor."""

    @property
    @abstractmethod
    def admission_state(self) -> HardwareAdmissionState:
        ...

    @property
    @abstractmethod
    def publisher_count(self) -> int:
        ...

    @abstractmethod
    def observe_scene(self, scene_id: str, now_mono: float) -> tuple[str, list[ObjectRecord]]:
        ...

    @abstractmethod
    def run_phase_step(
        self,
        *,
        obj: ObjectRecord,
        phase: ExecutionPhase,
        task_generation: int,
        attempt_number: int,
        now_mono: float,
    ) -> PhaseStepResult:
        ...

    @abstractmethod
    def run_empty_grasp_recovery(
        self,
        *,
        obj: ObjectRecord,
        task_generation: int,
        now_mono: float,
    ) -> PhaseStepResult:
        ...

    @abstractmethod
    def request_phase_hold(self, *, current_phase: ExecutionPhase, reason: str) -> bool:
        ...

    @abstractmethod
    def cancel_future_queue(self, *, generation: int, reason: str) -> int:
        ...

    def begin_new_batch(self, *, allow_slot_recycling: bool = False) -> None:
        """Reset per-batch executor tracking if supported by backend."""
        return

    def sync_active_grid_colors(self, colors: set[str]) -> None:
        """Inform executor which object colors are currently visible on the white workspace grid."""
        return

    def return_home_on_completion(self) -> tuple[bool, str | None, str | None]:
        """Optional guarded return-to-home after safe empty-handed batch completion."""
        return False, None, None


@dataclass
class FaultConfig:
    """Configurable fault injection knobs for deterministic D1 testing."""

    empty_grasp_attempts: dict[tuple[str, int], bool] = field(default_factory=dict)
    uncertain_held_objects: set[str] = field(default_factory=set)
    placement_unconfirmed_objects: set[str] = field(default_factory=set)
    timeout_on: tuple[str, ExecutionPhase] | None = None
    guard_reject_on: tuple[str, ExecutionPhase] | None = None
    stale_obs_age_ms: float | None = None
    stale_result_generation_once: int | None = None
    infeasible_objects_next_rounds: dict[str, int] = field(default_factory=dict)
    hide_color_on_reobserve: set[str] = field(default_factory=set)
    reject_hold_request: bool = False


class MockRobotPublisher(ExecutionBridge):
    """Pure-software simulated execution backend with zero CAN/SDK side effects."""

    def __init__(
        self,
        *,
        step_delay_s: float = 0.25,
        faults: FaultConfig | None = None,
        clock_fn: Callable[[], float] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._step_delay_s = float(step_delay_s)
        self.faults = faults or FaultConfig()
        self._clock_fn = clock_fn or time.monotonic
        self._obs_seq = 0
        self._placed_ids: set[str] = set()
        self._recovery_count = 0
        self.cancelled_generations: list[tuple[int, str]] = []
        self.hold_requests: list[tuple[ExecutionPhase, str]] = []
        self.robot_connect_calls = 0
        self.home_calls = 0
        self.placed_history: list[str] = []
        self.phase_hook: Callable[[ObjectRecord, ExecutionPhase], None] | None = None

    @property
    def admission_state(self) -> HardwareAdmissionState:
        return HardwareAdmissionState.SIMULATION_MOCK

    @property
    def publisher_count(self) -> int:
        return 1

    def reset_scene_tracking(self) -> None:
        with self._lock:
            self._placed_ids.clear()
            self.cancelled_generations.clear()
            self.hold_requests.clear()
            self.placed_history.clear()

    def observe_scene(self, scene_id: str, now_mono: float) -> tuple[str, list[ObjectRecord]]:
        with self._lock:
            preset = SCENE_PRESETS.get(scene_id)
            if preset is None:
                raise KeyError(f"Unknown scene preset: {scene_id}")
            self._obs_seq += 1
            obs_id = f"mock_obs_{self._obs_seq:04d}"
            obs_ts = now_mono
            if self.faults.stale_obs_age_ms is not None:
                obs_ts = now_mono - (self.faults.stale_obs_age_ms / 1000.0)

            out: list[ObjectRecord] = []
            for raw in preset.objects:
                oid = str(raw["object_id"])
                color = str(raw["color"])
                if oid in self._placed_ids:
                    continue
                if color in self.faults.hide_color_on_reobserve and self._obs_seq > 1:
                    continue

                rem_infeasible = self.faults.infeasible_objects_next_rounds.get(oid, 0)
                path_ok = True
                if rem_infeasible > 0:
                    path_ok = False
                    self.faults.infeasible_objects_next_rounds[oid] = rem_infeasible - 1

                out.append(
                    ObjectRecord(
                        object_id=oid,
                        category="woodblock",
                        color=color,
                        pose_estimate=copy.deepcopy(raw["pose_estimate"]),
                        confidence=float(raw["confidence"]),
                        observation_id=obs_id,
                        observation_ts_mono=obs_ts,
                        destination_id="brown_target",
                        occlusion_score=float(raw["occlusion_score"]),
                        clearance_m=float(raw["clearance_m"]),
                        grasp_pose_valid=True,
                        path_checked=path_ok,
                        estimated_duration_s=float(raw["estimated_duration_s"]),
                    )
                )
            return obs_id, out

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
        if self._step_delay_s > 0.0:
            time.sleep(self._step_delay_s)

        with self._lock:
            current_now = self._clock_fn()
            self._obs_seq += 1
            obs_id = f"mock_step_obs_{self._obs_seq:04d}"
            task_prompt = format_single_target_task(obj)
            _ = task_prompt  # Ensures single-target prompt formatting is exercised.

            reported_gen = task_generation
            if self.faults.stale_result_generation_once is not None:
                reported_gen = self.faults.stale_result_generation_once
                self.faults.stale_result_generation_once = None

            if self.faults.timeout_on == (obj.object_id, phase):
                return PhaseStepResult(
                    ok=False,
                    phase=phase,
                    task_generation=reported_gen,
                    observation_id=obs_id,
                    observation_ts_mono=current_now,
                    gripper_width_m=0.070,
                    ee_pose=(0.25, 0.0, 0.20, -0.45, 0.90, 0.94),
                    error_code="inference_timeout",
                    error_detail=f"Simulated inference timeout on {obj.object_id} during {phase.value}",
                )

            if self.faults.guard_reject_on == (obj.object_id, phase):
                return PhaseStepResult(
                    ok=False,
                    phase=phase,
                    task_generation=reported_gen,
                    observation_id=obs_id,
                    observation_ts_mono=current_now,
                    gripper_width_m=0.070,
                    ee_pose=(0.25, 0.0, 0.20, -0.45, 0.90, 0.94),
                    error_code="candidate_guard_rejected",
                    error_detail=f"Simulated guard rejection on {obj.object_id} during {phase.value}",
                )

            if phase == ExecutionPhase.LIFTING_CHECK:
                is_empty = self.faults.empty_grasp_attempts.get((obj.object_id, attempt_number), False)
                if not is_empty and obj.color == "blue" and obj.object_id == "obj_blue_3" and attempt_number == 1:
                    is_empty = True

                if obj.object_id in self.faults.uncertain_held_objects:
                    # Conflicting: gripper partially closed as if holding, yet vision still sees block on table!
                    return PhaseStepResult(
                        ok=True,
                        phase=phase,
                        task_generation=reported_gen,
                        observation_id=obs_id,
                        observation_ts_mono=current_now,
                        gripper_width_m=0.028,
                        ee_pose=(obj.pose_estimate["x"], obj.pose_estimate["y"], 0.20, -0.45, 0.90, 0.94),
                        visual_block_on_table=True,
                    )
                if is_empty:
                    return PhaseStepResult(
                        ok=True,
                        phase=phase,
                        task_generation=reported_gen,
                        observation_id=obs_id,
                        observation_ts_mono=current_now,
                        gripper_width_m=0.008,
                        ee_pose=(obj.pose_estimate["x"], obj.pose_estimate["y"], 0.20, -0.45, 0.90, 0.94),
                        visual_block_on_table=True,
                    )
                return PhaseStepResult(
                    ok=True,
                    phase=phase,
                    task_generation=reported_gen,
                    observation_id=obs_id,
                    observation_ts_mono=current_now,
                    gripper_width_m=0.028,
                    ee_pose=(obj.pose_estimate["x"], obj.pose_estimate["y"], 0.208, -0.45, 0.90, 0.94),
                    visual_block_on_table=False,
                )

            if phase == ExecutionPhase.RELEASING_CHECK:
                if obj.object_id in self.faults.placement_unconfirmed_objects:
                    return PhaseStepResult(
                        ok=True,
                        phase=phase,
                        task_generation=reported_gen,
                        observation_id=obs_id,
                        observation_ts_mono=current_now,
                        gripper_width_m=0.072,
                        ee_pose=(0.348, 0.216, 0.193, -0.45, 0.90, 0.94),
                        visual_block_in_brown_zone=False,
                        visual_block_at_gripper=True,
                    )
                self._placed_ids.add(obj.object_id)
                self.placed_history.append(obj.object_id)
                return PhaseStepResult(
                    ok=True,
                    phase=phase,
                    task_generation=reported_gen,
                    observation_id=obs_id,
                    observation_ts_mono=current_now,
                    gripper_width_m=0.072,
                    ee_pose=(0.348, 0.216, 0.193, -0.45, 0.90, 0.94),
                    visual_block_in_brown_zone=True,
                    visual_block_at_gripper=False,
                )

            return PhaseStepResult(
                ok=True,
                phase=phase,
                task_generation=reported_gen,
                observation_id=obs_id,
                observation_ts_mono=current_now,
                gripper_width_m=0.028 if phase in (ExecutionPhase.GRASPING, ExecutionPhase.TRANSPORTING, ExecutionPhase.PLACING) else 0.072,
                ee_pose=(0.348, 0.216, 0.193, -0.45, 0.90, 0.94),
            )

    def run_empty_grasp_recovery(
        self,
        *,
        obj: ObjectRecord,
        task_generation: int,
        now_mono: float,
    ) -> PhaseStepResult:
        if self._step_delay_s > 0.0:
            time.sleep(self._step_delay_s)
        with self._lock:
            self._recovery_count += 1
            self._obs_seq += 1
            return PhaseStepResult(
                ok=True,
                phase=ExecutionPhase.RECOVERING,
                task_generation=task_generation,
                observation_id=f"mock_recover_{self._obs_seq:04d}",
                observation_ts_mono=self._clock_fn(),
                gripper_width_m=0.075,
                ee_pose=(obj.pose_estimate["x"], obj.pose_estimate["y"], 0.200, -0.45, 0.90, 0.94),
            )

    def request_phase_hold(self, *, current_phase: ExecutionPhase, reason: str) -> bool:
        with self._lock:
            self.hold_requests.append((current_phase, reason))
            return not self.faults.reject_hold_request

    def cancel_future_queue(self, *, generation: int, reason: str) -> int:
        with self._lock:
            self.cancelled_generations.append((generation, reason))
            return 1

