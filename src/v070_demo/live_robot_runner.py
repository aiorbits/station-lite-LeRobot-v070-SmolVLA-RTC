"""Stage D3 supervised hardware execution bridge connecting SessionService to live right7."""

from __future__ import annotations

import base64
import concurrent.futures
import json
import math
import threading
import time
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from .candidate_bridge import D2_BUNDLE_SHA256, verify_sdk_dryrun_report
from .candidate_import import import_build_grasp_replay_cases, import_right7_guarded
from .contracts import (
    ExecutionPhase,
    HardwareAdmissionState,
    ObjectRecord,
)
from .evidence_verifier import (
    EMPTY_CLOSED_MAX_M,
    HELD_GRIP_MAX_M,
    HELD_GRIP_MIN_M,
    verify_live_grasp_outcome,
    verify_live_placement_outcome,
)
from .executor_bridge import ExecutionBridge, PhaseStepResult
from .perception_adapter import (
    GRID_HOVER_POSES,
    detect_all_high_cam_blocks,
)
from .task_parser import format_single_target_task
from .trajectory_planner import (
    plan_brown_to_clear_obs,
    plan_brown_to_next_grid_hover,
    plan_clear_obs_to_grid_hover,
    plan_empty_grasp_retreat,
    plan_post_grasp_slot_transport,
    plan_vertical_lift_in_place,
    validate_cartesian_segment,
)

_grasp_cases = import_build_grasp_replay_cases()
_guarded = import_right7_guarded()

_HOME_XYZ = np.array([0.1272, 0.0002, 0.1776], dtype=float)

# Calibrated brown cork pad interior release separation (meters around visual pad center ~[0.364, 0.239]).
# Arrival transport uses calibrated slot_index=0 ([0.348, 0.216]) instead of [0.323, 0.216], because
# negative offsets from [0.348, 0.216] land outside the cork border onto the white 'SP' table sticker.
_PAD_CENTER_CLAMP_XY_LOW = np.array([0.360, 0.235], dtype=float)
_PAD_CENTER_CLAMP_XY_HIGH = np.array([0.368, 0.243], dtype=float)
_PAD_INTERIOR_XY_LOW = np.array([0.348, 0.224], dtype=float)
_PAD_INTERIOR_XY_HIGH = np.array([0.380, 0.256], dtype=float)
_PAD_INTERIOR_SLOT_OFFSETS_XY_M: tuple[tuple[float, float], ...] = (
    (-0.011, -0.008),  # Slot 0: ~[0.353, 0.231]
    (+0.015, -0.008),  # Slot 1: ~[0.379, 0.231] (26 mm X separation)
    (+0.002, +0.015),  # Slot 2: ~[0.366, 0.254] (26.4 mm triangle separation)
)


def _validate_candidate_server_url(server_url: str) -> str:
    raw = str(server_url or "").strip().rstrip("/")
    parsed = urllib.parse.urlparse(raw)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or any(ord(ch) > 127 for ch in raw)
        or "<" in raw
        or ">" in raw
    ):
        raise ValueError(f"Invalid isolated candidate server URL: {server_url!r}")
    if parsed.port == 8010:
        raise PermissionError("Port 8010 is reserved for production baseline; use isolated v0.7.0 server (e.g. 8011)")
    return raw


@dataclass
class LiveHardwareConfig:
    """Strict Stage D3 supervised hardware parameters (fail-closed defaults)."""

    server_url: str = "http://127.0.0.1:8011"
    img_host: str = "127.0.0.1"
    dry_run_report_path: Path | None = None
    expected_bundle_sha256: str | None = D2_BUNDLE_SHA256
    expected_checkpoint: str = str(_guarded.EXPECTED_CHECKPOINT)
    expected_seed_mode: str = "fixed"
    operator_admitted: bool = False
    confirm_startup_home: bool = False
    confirm_exit_home: bool = False
    exit_home_policy: str = "safe_exit_only"
    unsafe_exit_action: str = "confirm_before_disable"
    min_safe_home_z_m: float = 0.175
    max_observation_age_ms: float = 800.0
    hover_z_m: float = 0.200
    margin_rad: float = 0.02
    max_soft_clip_rad: float = 0.0
    rate_hz: float = 30.0
    speed: float = 0.30
    takeoff_speed: float = 0.30
    aerial_approach_speed: float = 0.36
    approach_speed: float = 0.24
    grasp_duration_s: float = 2.5
    clamp_preload_m: float = 0.0125
    transport_speed: float = 0.30
    place_speed: float = 0.28
    switch_delay_s: float = 0.04
    settle_margin_s: float = 0.10
    settle_drift_sleep_s: float = 0.10
    max_initial_gap_mm: float = 25.0
    max_rotation_gap_deg: float = 30.0
    max_gripper_gap_mm: float = 15.0
    settled_bridge_max_mm: float = 40.0
    placement_bridge_max_mm: float = 50.0
    wrist_align_tol_px: float = 20.0
    initial_bridge_steps: int = 6
    max_initial_retreat_targets: int = 2
    max_open_gripper_overshoot_mm: float = 18.0
    max_close_gripper_undershoot_mm: float = 12.0
    max_brown_slots: int = 3
    use_clear_obs_retreat: bool = False
    return_home_on_batch_complete: bool = False
    predict_timeout_s: float = 15.0
    robot_init_settle_s: float = 1.5
    chunk_wait_slice_s: float = 0.05
    snapshot_dir: Path | None = None


class LiveRight7HardwareBridge(ExecutionBridge):
    """Single-owner supervised live right7 ExecutionBridge for Stage D3.

    Preserves strict D3 safety guarantees:
    1. Does NOT import or construct DualTouchRobot/ImageClient on construction, web load, or start_session.
    2. Requires verified dry-run report + explicit admission + live /health check on isolated non-8010 server.
    3. Attaches hardware lazily on the first batch observe_scene and runs one non-acting JIT warmup /predict.
    4. Never executes implicit go_home() mid-batch, on pause, cancel, fault, or held-uncertain exit.
    """

    def __init__(
        self,
        *,
        config: LiveHardwareConfig,
        robot_cls: Callable[[], Any] | None = None,
        image_client_cls: Callable[..., Any] | None = None,
        hold_confirm_fn: Callable[[dict[str, Any]], None] | None = None,
        clock_fn: Callable[[], float] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self._exec_lock = threading.RLock()
        self._state_lock = threading.Lock()
        self.config = config
        self._server_url = _validate_candidate_server_url(config.server_url)
        if config.max_observation_age_ms <= 0.0 or config.max_observation_age_ms > 800.0:
            raise ValueError("max_observation_age_ms must be in (0, 800]")
        if float(config.max_soft_clip_rad) != 0.0:
            raise ValueError("max_soft_clip_rad must remain 0.0")
        if not (30.0 <= float(config.settled_bridge_max_mm) <= 50.0):
            raise ValueError("settled_bridge_max_mm must remain in [30.0, 50.0] mm")
        if not (30.0 <= float(config.placement_bridge_max_mm) <= 60.0):
            raise ValueError("placement_bridge_max_mm must remain in [30.0, 60.0] mm")
        if not (0.005 <= float(config.clamp_preload_m) <= 0.016):
            raise ValueError("clamp_preload_m must remain in [0.005, 0.016] m")

        self._robot_cls = robot_cls
        self._image_client_cls = image_client_cls
        self._hold_confirm_fn = hold_confirm_fn
        self._clock = clock_fn or time.monotonic
        self._sleep = sleep_fn or time.sleep

        self._admitted = False
        self._dryrun_info: dict[str, Any] | None = None
        self._health_info: dict[str, Any] | None = None

        self._robot: Any = None
        self._image_client: Any = None
        self._hardware_attached = False
        self._warmup_infer_ms: float | None = None

        self._session_tag = "demo-d3-" + uuid.uuid4().hex[:10]
        self._obs_seq = 0
        self._predict_seq = 0
        self._sent_chunks = 0
        self._last_install_mono: float | None = None
        self._last_duration_s: float = 0.0
        self._placed_slots_used = 0
        self._allow_slot_recycling = False
        self._occupied_pad_slots: dict[int, str] = {}
        self._returned_home_after_batch = False
        self._held_grip_m: float | None = None
        self._holding_object = False
        self._placement_completed_any = False
        self._at_clear_obs = False
        self._last_place_target_obb: dict[str, Any] | None = None
        self._hold_event = threading.Event()
        self._pregrasp_preempt_event = threading.Event()
        self._cancelled_generations: list[tuple[int, str]] = []
        self._latest_jpegs: dict[str, bytes] = {}
        self._latest_frame_mono: dict[str, float] = {}
        self._shutdown_done = False

        self._speed_args = SimpleNamespace(
            phase_speeds=True,
            rate=float(config.rate_hz),
            speed=float(config.speed),
            takeoff_speed=float(config.takeoff_speed),
            aerial_approach_speed=float(config.aerial_approach_speed),
            approach_speed=float(config.approach_speed),
            hold_grasp_pose=True,
            grasp_duration_s=float(config.grasp_duration_s),
            transport_speed=float(config.transport_speed),
            place_speed=float(config.place_speed),
        )

    @property
    def admission_state(self) -> HardwareAdmissionState:
        return HardwareAdmissionState.ADMITTED_SUPERVISED if self._admitted else HardwareAdmissionState.UNADMITTED

    @property
    def publisher_count(self) -> int:
        return 1

    @property
    def hardware_attached(self) -> bool:
        return self._hardware_attached

    @property
    def sent_chunks(self) -> int:
        return self._sent_chunks

    def runtime_status(self) -> dict[str, Any]:
        """Return safe read-only runtime telemetry without touching hardware or blocking on motion."""
        with self._state_lock:
            return {
                "server_url": self._server_url,
                "hardware_attached": self._hardware_attached,
                "sent_chunks": self._sent_chunks,
                "placed_slots_used": self._placed_slots_used,
                "warmup_infer_ms": self._warmup_infer_ms,
                "dryrun_input_sha256": (self._dryrun_info or {}).get("input_sha256"),
                "dryrun_cases_passed": (self._dryrun_info or {}).get("cases_passed"),
                "checkpoint": (self._health_info or {}).get("checkpoint"),
                "seed_base": (self._health_info or {}).get("seed_base"),
            }

    def latest_jpeg(self, cam_name: str) -> bytes | None:
        with self._state_lock:
            return self._latest_jpegs.get(cam_name)

    def admit_hardware(self) -> dict[str, Any]:
        """Verify dry-run SHA256, workspace flags, and isolated CUDA /health without constructing robot."""
        with self._state_lock:
            if self.config.dry_run_report_path is None:
                raise PermissionError("Stage D3 requires --dry-run-report path from CAN-isolated SDK dry-run")
            if not self.config.operator_admitted:
                raise PermissionError("Stage D3 blocked: operator_admitted (--admit-hardware) required")
            if not (self.config.confirm_startup_home and self.config.confirm_exit_home):
                raise PermissionError(
                    "Stage D3 blocked: workspace clear confirmations (--confirm-startup-home & --confirm-exit-home) required"
                )
            dryrun = verify_sdk_dryrun_report(
                self.config.dry_run_report_path,
                expected_bundle_sha256=self.config.expected_bundle_sha256,
                min_cases=7,
            )
            health = _guarded.validate_right7_health(self._server_url, self.config.expected_checkpoint)
            if self.config.expected_seed_mode and health.get("seed_mode") != self.config.expected_seed_mode:
                raise PermissionError(
                    f"Unexpected server seed_mode={health.get('seed_mode')!r}; expected {self.config.expected_seed_mode!r}"
                )
            self._dryrun_info = dryrun
            self._health_info = health
            self._admitted = True
            return {"dryrun": dryrun, "health": health}

    def _ensure_hardware_attached(self) -> None:
        if not self._admitted:
            raise PermissionError("Live hardware execution blocked: admit_hardware() has not succeeded")
        if self._hardware_attached:
            return

        robot_cls = self._robot_cls
        image_client_cls = self._image_client_cls
        if image_client_cls is None:
            from teleimager.image_client import ImageClient as image_client_cls  # noqa: E402
        if robot_cls is None:
            from xr_teleoperate2.robot_control.robot_dual_touch import DualTouchRobot as robot_cls  # noqa: E402

        print("[D3 HARDWARE ATTACH] Instantiating DualTouchRobot (homes BOTH arms)...", flush=True)
        robot = robot_cls()
        self._sleep(float(self.config.robot_init_settle_s))
        image_client = image_client_cls(host=self.config.img_host, auto_subscribe=False)
        image_client.start_subscriptions()

        self._robot = robot
        self._image_client = image_client
        with self._state_lock:
            self._hardware_attached = True

        # Run one non-acting JIT warmup /predict so chunk 0 does not exceed 800 ms on cold compile.
        t0 = self._clock()
        payload, _raw = self._read_live_observation(task=str(_guarded.TASK))
        result = self._post_predict(payload)
        _guarded.validate_predict_response(
            result,
            expected_session_id=self._session_tag,
            expected_seq=payload["seq"],
            expected_seed_mode=self.config.expected_seed_mode,
            expected_seed_base=(self._health_info or {}).get("seed_base"),
        )
        warmup_ms = round((self._clock() - t0) * 1000.0, 1)
        with self._state_lock:
            self._warmup_infer_ms = warmup_ms
        print(f"[D3 WARMUP] Non-acting /predict warmup OK infer_total_ms={warmup_ms:.1f}", flush=True)

    def _read_live_observation(self, *, task: str) -> tuple[dict[str, Any], dict[str, Any]]:
        seq = self._predict_seq
        self._predict_seq += 1
        timing: dict[str, Any] = {}
        # Single monotonic clock domain (Section 6.4): capture earliest bound in self._clock before frame wait.
        first_cam_mono = float(self._clock())
        payload = _guarded.read_observation(
            self._robot,
            self._image_client,
            _guarded.wait_frame,
            _guarded.enc_jpeg_b64,
            _guarded.REQUEST_SCHEMA,
            self._session_tag,
            seq,
            timing,
            task=task,
        )
        cam_done_mono = float(self._clock())
        high_b64 = payload["images_jpeg_base64"].get("observation.images.cam_high", "")
        wrist_b64 = payload["images_jpeg_base64"].get("observation.images.cam_right_wrist", "")
        high_bytes = base64.b64decode(high_b64) if high_b64 else b""
        wrist_bytes = base64.b64decode(wrist_b64) if wrist_b64 else b""
        with self._state_lock:
            if high_bytes:
                self._latest_jpegs["cam_high"] = high_bytes
                self._latest_frame_mono["cam_high"] = first_cam_mono
            if wrist_bytes:
                self._latest_jpegs["cam_right_wrist"] = wrist_bytes
                self._latest_frame_mono["cam_right_wrist"] = cam_done_mono
        raw = {
            "timing": timing,
            "first_camera_mono": first_cam_mono,
            "high_bytes": high_bytes,
            "wrist_bytes": wrist_bytes,
        }
        return payload, raw

    def _post_predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            f"{self._server_url}/predict",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=float(self.config.predict_timeout_s)) as resp:
            return json.load(resp)

    def _query_policy_actions(self, *, task: str) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
        payload, raw = self._read_live_observation(task=task)
        try:
            result = self._post_predict(payload)
        except (TimeoutError, concurrent.futures.TimeoutError, urllib.error.URLError) as exc:
            raise TimeoutError(f"inference_timeout:{exc}") from exc
        actions = _guarded.validate_predict_response(
            result,
            expected_session_id=self._session_tag,
            expected_seq=payload["seq"],
            expected_seed_mode=self.config.expected_seed_mode,
            expected_seed_base=(self._health_info or {}).get("seed_base"),
        )
        return payload, raw, np.asarray(actions, dtype=float)

    def _measured_pose7(self) -> tuple[np.ndarray, np.ndarray]:
        arm = self._robot.right_arm
        q = np.asarray(arm.get_joint_positions(), dtype=float).reshape(-1)
        grip = float(arm.get_gripper_distance())
        pos, euler = arm.get_ee_pose_euler()
        pos = np.asarray(pos, dtype=float).reshape(3)
        euler = np.asarray(euler, dtype=float).reshape(3)
        if q.shape != (6,) or not np.isfinite(q).all() or not np.isfinite(grip):
            raise ValueError("Invalid right-arm joint or gripper state")
        if not np.isfinite(pos).all() or not np.isfinite(euler).all():
            raise ValueError("Invalid right-arm end-effector pose")
        return q, np.r_[pos, euler, grip]

    @staticmethod
    def _action_diagnostics(pose7: np.ndarray, actions: np.ndarray) -> dict[str, float]:
        delta = actions[0] - pose7
        delta[3:6] = (delta[3:6] + math.pi) % (2.0 * math.pi) - math.pi
        return {
            "raw0_gap_mm": float(np.linalg.norm(delta[:3]) * 1000.0),
            "max_rot_deg": float(np.degrees(np.max(np.abs(delta[3:6])))),
            "start_grip": float(pose7[6]),
            "grip_min": float(np.min(actions[:, 6])),
            "grip_max": float(np.max(actions[:, 6])),
        }

    def _save_snapshot(
        self,
        label: str,
        payload: dict[str, Any],
        segment: np.ndarray | None = None,
        *,
        raw_actions: np.ndarray | None = None,
        pose7: np.ndarray | None = None,
    ) -> None:
        if self.config.snapshot_dir is None:
            return
        root = Path(self.config.snapshot_dir)
        root.mkdir(parents=True, exist_ok=True)
        if segment is not None:
            np.save(root / f"{label}_segment.npy", np.asarray(segment, dtype=float))
        if raw_actions is not None:
            np.save(root / f"{label}_raw_actions.npy", np.asarray(raw_actions, dtype=float))
        for key, encoded in payload.get("images_jpeg_base64", {}).items():
            cam = key.rsplit(".", 1)[-1]
            (root / f"{label}_{cam}.jpg").write_bytes(base64.b64decode(encoded))
        state_obj: dict[str, Any] = {"state": payload.get("state"), "task": payload.get("task")}
        if pose7 is not None:
            state_obj["pose7"] = [float(x) for x in pose7]
        (root / f"{label}_state.json").write_text(
            json.dumps(state_obj, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _evaluate_post_plan_guard(
        *,
        loop_index: int,
        q_start: np.ndarray,
        q_fresh: np.ndarray,
        wp0: np.ndarray,
        seg0_grip: float,
        grip_fresh: float,
        obs_age_ms: float,
        max_gripper_gap_mm: float,
        max_observation_age_ms: float,
    ) -> list[str]:
        """Match guarded Home takeoff tolerances (0.02/0.25 rad) at chunk 0 and settled guard for chunk > 0."""
        if loop_index > 0:
            return list(
                _guarded.evaluate_settled_post_plan_guard(
                    q_start,
                    q_fresh,
                    wp0,
                    seg0_grip,
                    grip_fresh,
                    obs_age_ms,
                    max_gripper_gap_mm=max_gripper_gap_mm,
                    max_observation_age_ms=max_observation_age_ms,
                )
            )
        reasons: list[str] = []
        if (not np.all(np.isfinite(q_fresh))) or float(np.max(np.abs(q_fresh - q_start))) > 0.02:
            reasons.append("post_plan_state_drift")
        if (not np.all(np.isfinite(wp0))) or float(np.max(np.abs(wp0 - q_fresh))) > 0.25:
            reasons.append("first_waypoint_gap")
        if not math.isfinite(grip_fresh):
            reasons.append("invalid_gripper_feedback")
        elif abs(float(seg0_grip) - grip_fresh) * 1000.0 > float(max_gripper_gap_mm):
            reasons.append("gripper_gap")
        if float(obs_age_ms) > float(max_observation_age_ms):
            reasons.append("observation_expired")
        return reasons

    def _plan_and_send_segment(
        self,
        *,
        segment: np.ndarray,
        phase_name: str,
        first_camera_mono: float | None = None,
        refine: bool = True,
        loop_index: int = 1,
        max_joint_step_rad: float | None = None,
        max_first_step_m: float | None = None,
        payload_for_snapshot: dict[str, Any] | None = None,
        snapshot_label: str | None = None,
    ) -> tuple[bool, str | None, str | None]:
        """Unified IK, smoothing, controller dry-run, 800ms freshness gate, and chunk execution."""
        arm = self._robot.right_arm
        q_start, pose7_start = self._measured_pose7()
        seg = np.asarray(segment, dtype=float)
        if payload_for_snapshot is not None and snapshot_label:
            self._save_snapshot(snapshot_label, payload_for_snapshot, seg)

        # Fail fast on malformed arrays or out-of-range grippers before IK/refinement.
        if seg.ndim != 2 or len(seg) < 2 or seg.shape[1] != 7 or not np.isfinite(seg).all():
            raise ValueError(f"Invalid segment shape or non-finite values: {seg.shape}")
        if np.any(seg[:, 6] < 0.0) or np.any(seg[:, 6] > float(_guarded.GRIPPER_MAX_M)):
            raise ValueError("Gripper target out of calibrated bounds [0.0, 0.085]")

        eff_first_step_m = (
            float(max_first_step_m)
            if max_first_step_m is not None
            else (self.config.max_initial_gap_mm / 1000.0 if loop_index == 0 else 0.015)
        )

        def _check_ik(cand: np.ndarray):
            return _guarded.check_right_targets(
                arm,
                cand,
                float(self.config.margin_rad),
                _guarded.solve_with_retreat,
                _guarded.JOINT_LOW,
                _guarded.JOINT_UP,
                _guarded.GRIPPER_MAX_M,
                float(self.config.max_soft_clip_rad),
                max_initial_retreat_targets=(self.config.max_initial_retreat_targets if loop_index == 0 else 0),
                max_open_gripper_overshoot_m=float(self.config.max_open_gripper_overshoot_mm) / 1000.0,
                max_close_gripper_undershoot_m=float(self.config.max_close_gripper_undershoot_mm) / 1000.0,
            )

        if refine and loop_index > 0:
            seg, waypoints, grippers, _ins = _guarded.refine_joint_steps(seg, _check_ik, limit_rad=0.07)
        else:
            waypoints, grippers = _check_ik(seg)

        # Verify Cartesian step bounds on refined segment so refine_joint_steps handles >15mm interior gaps first.
        validate_cartesian_segment(
            pose7_start,
            seg,
            loop_index=loop_index,
            max_first_step_m=eff_first_step_m,
        )

        raw_wp = np.asarray(waypoints, dtype=float)
        smoothed_wp = _guarded.smooth_joint_waypoints(waypoints, alpha=0.20, passes=1)
        step_lim = (
            max_joint_step_rad
            if max_joint_step_rad is not None
            else (0.25 if loop_index == 0 else (0.07 if refine else 0.08))
        )
        _guarded.validate_post_smoothing_waypoints(
            smoothed_wp,
            raw_waypoints=raw_wp,
            margin=float(self.config.margin_rad),
            max_soft_clip_rad=float(self.config.max_soft_clip_rad),
            max_joint_step_rad=float(step_lim),
        )

        wp_list = [list(map(float, row)) for row in smoothed_wp]
        grip_list = [float(g) for g in grippers]
        first_jump = float(np.max(np.abs(np.asarray(wp_list[0], dtype=float) - q_start)))
        jump_limit = 0.25 if loop_index == 0 else 0.07
        if first_jump > jump_limit:
            return False, "ik_first_waypoint_gap", f"jump={first_jump:.4f}rad > {jump_limit:.2f}rad"

        speed_val, dur_s = _guarded.get_phase_speed_and_duration(phase_name, len(wp_list), self._speed_args)
        inner_arm = getattr(arm, "arm", None)
        planner = getattr(inner_arm, "plan_joint_waypoints_with_gripper", None)
        if callable(planner):
            planned = np.asarray(planner(q_start.tolist(), wp_list, grip_list, time_sec=float(dur_s)), dtype=float)
            if (
                planned.ndim != 2
                or len(planned) < 2
                or planned.shape[1] < 15
                or not np.isfinite(planned).all()
                or np.any(np.diff(planned[:, 0]) <= 0)
                or np.any(planned[:, 1:7] < _guarded.JOINT_LOW)
                or np.any(planned[:, 1:7] > _guarded.JOINT_UP)
            ):
                return False, "sdk_dryrun_failed", "Controller dry-run failed soft/kinematic checks"

        # Freshness and hold checks right at install boundary (Section 6 / T3 <= 800 ms).
        if first_camera_mono is not None:
            age_ms = (self._clock() - float(first_camera_mono)) * 1000.0
            if age_ms > float(self.config.max_observation_age_ms):
                return False, "stale_observation", f"obs_age={age_ms:.1f}ms>{self.config.max_observation_age_ms:.1f}ms"

        q_fresh_arr = np.asarray(arm.get_joint_positions(), dtype=float).reshape(6)
        grip_fresh = float(arm.get_gripper_distance())
        obs_age_ms = (self._clock() - float(first_camera_mono)) * 1000.0 if first_camera_mono is not None else 0.0
        post_reasons = self._evaluate_post_plan_guard(
            loop_index=loop_index,
            q_start=q_start,
            q_fresh=q_fresh_arr,
            wp0=np.asarray(wp_list[0], dtype=float).reshape(6),
            seg0_grip=float(seg[0, 6]),
            grip_fresh=grip_fresh,
            obs_age_ms=obs_age_ms,
            max_gripper_gap_mm=float(self.config.max_gripper_gap_mm),
            max_observation_age_ms=float(self.config.max_observation_age_ms),
        )
        if post_reasons:
            code = "stale_observation" if "observation_expired" in post_reasons else "settled_guard_rejected"
            return False, code, ",".join(post_reasons)

        if self._hold_event.is_set():
            return False, "hold_requested_before_send", "Execution hold active; chunk not sent"

        print(
            f"[D3 SEND {self._sent_chunks}] phase={phase_name} waypoints={len(wp_list)} "
            f"dur={dur_s:.3f}s speed={speed_val:.2f}",
            flush=True,
        )
        arm.update_joint_waypoint_chunk_with_gripper(
            wp_list,
            grip_list,
            time_sec=float(dur_s),
            switch_delay_sec=float(self.config.switch_delay_s),
        )
        with self._state_lock:
            self._last_install_mono = self._clock()
            self._last_duration_s = float(dur_s)
            self._sent_chunks += 1

        wait_total = float(dur_s) + float(self.config.switch_delay_s) + float(self.config.settle_margin_s)
        deadline = self._clock() + wait_total
        while True:
            rem = deadline - self._clock()
            if rem <= 0.0:
                break
            self._sleep(min(float(self.config.chunk_wait_slice_s), rem))

        q_before_drift = np.asarray(arm.get_joint_positions(), dtype=float).reshape(6)
        self._sleep(float(self.config.settle_drift_sleep_s))
        q_after_drift = np.asarray(arm.get_joint_positions(), dtype=float).reshape(6)
        drift = _guarded.initial_state_drift(q_before_drift, q_after_drift)
        if drift > 0.025:
            return False, "arm_not_settled", f"post_chunk_drift={drift:.4f}rad>0.025rad"
        return True, None, None

    def observe_scene(self, scene_id: str, now_mono: float) -> tuple[str, list[ObjectRecord]]:
        with self._exec_lock:
            self._ensure_hardware_attached()
            self._obs_seq += 1
            obs_id = f"d3_obs_{self._obs_seq:04d}"
            frame = _guarded.wait_frame(self._image_client.get_head_frame)
            obs_ts = self._clock()
            high_bytes = base64.b64decode(_guarded.enc_jpeg_b64(frame))
            with self._state_lock:
                self._latest_jpegs["cam_high"] = high_bytes
                self._latest_frame_mono["cam_high"] = obs_ts
            records = detect_all_high_cam_blocks(
                high_bytes,
                observation_id=obs_id,
                observation_ts_mono=obs_ts,
            )
            return obs_id, records

    def _run_approaching_empty(self, obj: ObjectRecord) -> tuple[bool, str | None, str | None]:
        if self._next_available_slot() is None:
            return (
                False,
                "brown_pad_capacity_exceeded",
                f"slots_used={self._placed_slots_used},occupied={self._occupied_pad_slots}",
            )
        _q, pose7 = self._measured_pose7()
        cell = obj.grid_cell or "R8"
        hover_pose = GRID_HOVER_POSES.get(cell, GRID_HOVER_POSES["R8"])
        at_home = bool(np.linalg.norm(pose7[:3] - _HOME_XYZ) < 0.035)
        if at_home:
            takeoff_actions = _grasp_cases.condition_home_takeoff_actions(
                pose7,
                hover_pose=hover_pose,
                hover_z_m=float(self.config.hover_z_m),
                open_grip=max(float(pose7[6]), 0.072),
                max_steps=44,
            )
            if self.config.initial_bridge_steps > 0:
                prefix = _guarded.make_initial_prefix(pose7, takeoff_actions[0], int(self.config.initial_bridge_steps))
                prefix, _omitted = _guarded.select_initial_prefix(
                    self._robot.right_arm,
                    prefix,
                    float(self.config.margin_rad),
                )
                seg = np.concatenate([prefix, takeoff_actions], axis=0)[:50]
            else:
                seg = takeoff_actions[:50]
            self._at_clear_obs = False
            return self._plan_and_send_segment(
                segment=seg,
                phase_name="takeoff",
                refine=False,
                loop_index=0,
                max_joint_step_rad=0.25,
            )

        if self._at_clear_obs:
            seg = plan_clear_obs_to_grid_hover(pose7, target_cell=cell, hover_z_m=float(self.config.hover_z_m))
        else:
            seg = plan_brown_to_next_grid_hover(pose7, next_cell=cell, hover_z_m=float(self.config.hover_z_m))
        self._at_clear_obs = False
        return self._plan_and_send_segment(segment=seg, phase_name="transport", refine=True, loop_index=1)

    @staticmethod
    def _plan_aligned_hover_descent(pose7: np.ndarray, *, wrist_obb: dict[str, Any], target_z_m: float = 0.162) -> np.ndarray:
        """Build a bounded wrist-aligned vertical descent segment when policy omits <0.04 closure at hover."""
        start = np.asarray(pose7, dtype=float).reshape(7)
        cx, cy = float(wrist_obb["cx"]), float(wrist_obb["cy"])
        base_cx, base_cy = _grasp_cases._wrist_obb_base_targets(wrist_obb)
        corr_x_m, corr_y_m, _ = _grasp_cases.solve_wrist_xy_correction(
            start[:6],
            cx,
            cy,
            end_rpy_rad=start[3:6],
            cx_target=base_cx,
            cy_target=base_cy,
            table_grasp_z_m=0.160,
            deadband_u_px=4.0,
            deadband_v_px=4.0,
            max_dx_range=(-0.015, 0.015),
            max_dy_range=(-0.015, 0.015),
        )
        end_pose = start[:6].copy()
        end_pose[0] += float(corr_x_m)
        end_pose[1] += float(corr_y_m)
        end_pose[2] = float(np.clip(target_z_m, 0.158, float(start[2])))
        open_grip = float(min(max(float(start[6]), 0.072), 0.080))
        n_total = 32
        horiz_steps = 10
        seg = np.empty((n_total, 7), dtype=float)
        for idx in range(n_total):
            u_xy = min(1.0, float(idx + 1) / float(horiz_steps))
            w_xy = 0.5 * (1.0 - math.cos(math.pi * u_xy))
            seg[idx, 0] = float(start[0]) + w_xy * float(corr_x_m)
            seg[idx, 1] = float(start[1]) + w_xy * float(corr_y_m)
            seg[idx, 3:6] = start[3:6]
            if idx + 1 <= horiz_steps:
                seg[idx, 2] = float(start[2])
            else:
                u_z = float(idx + 1 - horiz_steps) / float(n_total - horiz_steps)
                w_z = 0.5 * (1.0 - math.cos(math.pi * u_z))
                seg[idx, 2] = float(start[2]) + w_z * (float(end_pose[2]) - float(start[2]))
            seg[idx, 6] = open_grip
        validate_cartesian_segment(start, seg, loop_index=1)
        return seg

    def _run_descending_contact(self, obj: ObjectRecord) -> tuple[bool, str | None, str | None]:
        task = format_single_target_task(obj)
        cell = obj.grid_cell or "R8"
        hover_pose = GRID_HOVER_POSES.get(cell, GRID_HOVER_POSES["R8"])

        for step_idx in range(4):
            if self._hold_event.is_set():
                return False, "hold_requested", "Pause or cancel requested"
            if self._pregrasp_preempt_event.is_set():
                return True, None, "preempted_before_grasp"
            payload, raw, actions = self._query_policy_actions(task=task)
            _q, pose7 = self._measured_pose7()
            wrist_obb = _grasp_cases.detect_wrist_block_obb(raw["wrist_bytes"], target_color=obj.color)
            dist_xy = float(np.linalg.norm(pose7[:2] - np.asarray(hover_pose[:2], dtype=float)))
            # Only apply coarse grid fallback when wrist OBB is absent; never pull an already
            # wrist-visible / wrist-aligned EE back toward nominal grid coordinates (>45mm offset).
            grid_pose = hover_pose if (wrist_obb is None and dist_xy > 0.045) else None

            diag = self._action_diagnostics(pose7, actions)
            obb_desc = "none" if wrist_obb is None else f"({wrist_obb['cx']:.1f},{wrist_obb['cy']:.1f})"
            print(
                f"[D3 APPROACH step={step_idx}] obj={obj.object_id} z={pose7[2]:.4f} "
                f"raw0_gap_mm={diag['raw0_gap_mm']:.1f} rot={diag['max_rot_deg']:.1f}deg "
                f"grip=[{diag['grip_min']:.4f},{diag['grip_max']:.4f}] start_grip={diag['start_grip']:.4f} "
                f"dist_hover_mm={dist_xy*1000.0:.1f} wrist_obb={obb_desc}",
                flush=True,
            )

            align = None
            if wrist_obb is not None:
                align = _grasp_cases.check_wrist_grasp_alignment(
                    wrist_obb,
                    ee_z_m=float(pose7[2]),
                    ee_pose6=pose7[:6],
                    tol_u_px=float(self.config.wrist_align_tol_px),
                    tol_v_px=float(self.config.wrist_align_tol_px),
                )
                if align["aligned"] and float(pose7[2]) <= 0.168 and np.any(actions[:, 6] < 0.04):
                    return True, None, None

            try:
                seg, _held_open, _anchored = _grasp_cases.condition_approach_actions(
                    pose7,
                    actions,
                    max_position_m=float(self.config.settled_bridge_max_mm) / 1000.0,
                    wrist_obb=wrist_obb,
                    hover_z_m=float(self.config.hover_z_m),
                    grid_hover_pose=grid_pose,
                    prefer_grid_hover_when_far=True,
                    adaptive_steps=True,
                )
            except ValueError:
                self._save_snapshot(
                    f"{obj.object_id}_approach_{step_idx}_rejected",
                    payload,
                    raw_actions=actions,
                    pose7=pose7,
                )
                raise
            seg, _trimmed = _grasp_cases.trim_segment_idle_tail(seg, keep_tail_hold=2, min_points=12)

            # Edge cells / out-of-distribution non-red blocks can stay tightly aligned at hover without
            # SmolVLA predicting <0.04 closure at z=0.200m. Once aligned after >=1 hover step, descend
            # smoothly to z=0.162m under verified wrist alignment instead of looping at hover height.
            if (
                step_idx >= 1
                and float(pose7[2]) > 0.175
                and float(np.min(seg[:, 2])) > 0.175
                and wrist_obb is not None
                and align is not None
                and align["aligned"]
                and abs(float(align["du"])) <= 15.0
                and abs(float(align["dv"])) <= 15.0
            ):
                seg = self._plan_aligned_hover_descent(pose7, wrist_obb=wrist_obb, target_z_m=0.162)
                print(
                    f"[D3 APPROACH DESCENT FALLBACK step={step_idx}] obj={obj.object_id} "
                    f"du={align['du']:.1f} dv={align['dv']:.1f} descending {pose7[2]:.4f}->{seg[-1, 2]:.4f}m",
                    flush=True,
                )

            phase_name = "aerial_approach" if float(np.min(seg[:, 2])) >= 0.176 else "approach"
            ok, err_c, err_d = self._plan_and_send_segment(
                segment=seg,
                phase_name=phase_name,
                first_camera_mono=raw["first_camera_mono"],
                refine=True,
                loop_index=1 + step_idx,
                max_first_step_m=float(self.config.settled_bridge_max_mm) / 1000.0,
                payload_for_snapshot=payload,
                snapshot_label=f"{obj.object_id}_approach_{step_idx}",
            )
            if not ok:
                return False, err_c, err_d
            _q_after, pose7_after = self._measured_pose7()
            if float(pose7_after[2]) <= 0.168 and wrist_obb is not None and align is not None and align["aligned"]:
                return True, None, None
        return True, None, None

    def _run_grasping(self, obj: ObjectRecord) -> tuple[bool, str | None, str | None]:
        task = format_single_target_task(obj)
        for attempt_sub in range(2):
            if self._hold_event.is_set():
                return False, "hold_requested", "Pause or cancel requested"
            payload, raw, actions = self._query_policy_actions(task=task)
            _q, pose7 = self._measured_pose7()
            wrist_obb = _grasp_cases.detect_wrist_block_obb(raw["wrist_bytes"], target_color=obj.color)
            align = _grasp_cases.check_wrist_grasp_alignment(
                wrist_obb,
                ee_z_m=float(pose7[2]),
                ee_pose6=pose7[:6],
                tol_u_px=float(self.config.wrist_align_tol_px),
                tol_v_px=float(self.config.wrist_align_tol_px),
            )
            has_closure = bool(np.any(actions[:, 6] < 0.04))
            diag = self._action_diagnostics(pose7, actions)
            print(
                f"[D3 GRASP step={attempt_sub}] obj={obj.object_id} z={pose7[2]:.4f} "
                f"raw0_gap_mm={diag['raw0_gap_mm']:.1f} rot={diag['max_rot_deg']:.1f}deg "
                f"grip=[{diag['grip_min']:.4f},{diag['grip_max']:.4f}] start_grip={diag['start_grip']:.4f} "
                f"has_closure={has_closure} du={align.get('du')} dv={align.get('dv')}",
                flush=True,
            )
            gross_misaligned = (
                (align["du"] is not None and abs(align["du"]) > 30.0)
                or (align["dv"] is not None and abs(align["dv"]) > 30.0)
            )
            tight_aligned = bool(
                wrist_obb is not None
                and align["aligned"]
                and align["du"] is not None
                and align["dv"] is not None
                and abs(float(align["du"])) <= 15.0
                and abs(float(align["dv"])) <= 15.0
            )
            use_wrist_closure_fallback = bool(
                not has_closure
                and float(pose7[2]) <= 0.166
                and tight_aligned
                and (attempt_sub == 1 or float(pose7[2]) <= 0.164)
            )
            if (has_closure and not gross_misaligned) or use_wrist_closure_fallback:
                cond_actions = actions
                if use_wrist_closure_fallback:
                    cond_actions = np.repeat(pose7[None, :], 50, axis=0)
                    cond_actions[15:, 6] = 0.0
                    print(
                        f"[D3 GRASP WRIST FALLBACK] obj={obj.object_id} z={pose7[2]:.4f} "
                        f"du={align['du']:.1f} dv={align['dv']:.1f}: triggering calibrated wrist-OBB closure",
                        flush=True,
                    )
                try:
                    seg, _clip, _splits = _grasp_cases.condition_grasp_actions(
                        pose7,
                        cond_actions,
                        max_position_m=float(self.config.settled_bridge_max_mm) / 1000.0,
                        hold_pose=True,
                        wrist_obb=wrist_obb,
                    )
                except ValueError:
                    self._save_snapshot(
                        f"{obj.object_id}_grasp_{attempt_sub}_rejected",
                        payload,
                        raw_actions=actions,
                        pose7=pose7,
                    )
                    raise
                return self._plan_and_send_segment(
                    segment=seg,
                    phase_name="grasp",
                    first_camera_mono=raw["first_camera_mono"],
                    refine=True,
                    loop_index=3,
                    max_first_step_m=float(self.config.settled_bridge_max_mm) / 1000.0,
                    payload_for_snapshot=payload,
                    snapshot_label=f"{obj.object_id}_grasp",
                )
            if attempt_sub == 0 and not gross_misaligned:
                if float(pose7[2]) > 0.175 and tight_aligned:
                    seg = self._plan_aligned_hover_descent(pose7, wrist_obb=wrist_obb, target_z_m=0.162)
                else:
                    # One final closed-loop fine approach micro-chunk before rejecting.
                    try:
                        seg, _ho, _pa = _grasp_cases.condition_approach_actions(
                            pose7,
                            actions,
                            max_position_m=float(self.config.settled_bridge_max_mm) / 1000.0,
                            wrist_obb=wrist_obb,
                            hover_z_m=float(self.config.hover_z_m),
                            prefer_grid_hover_when_far=False,
                            adaptive_steps=True,
                        )
                    except ValueError:
                        self._save_snapshot(
                            f"{obj.object_id}_grasp_fine_{attempt_sub}_rejected",
                            payload,
                            raw_actions=actions,
                            pose7=pose7,
                        )
                        raise
                ok, err_c, err_d = self._plan_and_send_segment(
                    segment=seg,
                    phase_name="approach",
                    first_camera_mono=raw["first_camera_mono"],
                    refine=True,
                    loop_index=3,
                    max_first_step_m=float(self.config.settled_bridge_max_mm) / 1000.0,
                )
                if not ok:
                    return False, err_c, err_d
                continue
            return (
                False,
                "grasp_guard_rejected",
                f"has_closure={has_closure} gross_misaligned={gross_misaligned} reason={align.get('reason')}",
            )
        return False, "grasp_guard_rejected", "Unreachable grasp exit"

    def _verify_and_log_lift_check(
        self,
        obj: ObjectRecord,
        *,
        pose7: np.ndarray,
        label_suffix: str,
        log_prefix: str,
    ):
        payload, raw = self._read_live_observation(task=format_single_target_task(obj))
        grip_val = float(pose7[6])
        verdict = verify_live_grasp_outcome(
            grip_val,
            cam_high_post_lift=raw["high_bytes"],
            wrist_post_lift=raw["wrist_bytes"],
            source_cell=obj.grid_cell,
            target_color=obj.color,
        )
        print(
            f"[{log_prefix}] obj={obj.object_id} grip={grip_val:.4f} verdict={verdict.verdict} "
            f"on_table={verdict.evidence.get('visual_block_still_at_cell')} "
            f"wrist_obb={verdict.evidence.get('wrist_post_lift_obb') is not None} "
            f"reason={verdict.evidence.get('reason', 'ok')}",
            flush=True,
        )
        self._save_snapshot(f"{obj.object_id}_{label_suffix}", payload, pose7=pose7)
        return verdict, grip_val

    def _run_lifting_check(self, obj: ObjectRecord) -> PhaseStepResult:
        _q, pose7_before = self._measured_pose7()
        grip_measured = float(pose7_before[6])
        if grip_measured <= EMPTY_CLOSED_MAX_M:
            lift_target_grip_m = min(grip_measured, 0.005)
        else:
            lift_preload_m = float(
                _grasp_cases.compute_adaptive_preload_m(grip_measured, float(self.config.clamp_preload_m))
            )
            lift_target_grip_m = max(0.005, grip_measured - lift_preload_m)
        lift_seg = plan_vertical_lift_in_place(
            pose7_before,
            target_z_m=float(self.config.hover_z_m),
            target_grip_m=lift_target_grip_m,
        )
        ok, err_c, err_d = self._plan_and_send_segment(
            segment=lift_seg,
            phase_name="transport",
            refine=True,
            loop_index=4,
        )
        _q, pose7_after = self._measured_pose7()
        if not ok:
            return PhaseStepResult(
                ok=False,
                phase=ExecutionPhase.LIFTING_CHECK,
                task_generation=0,
                observation_id=obj.observation_id,
                observation_ts_mono=obj.observation_ts_mono,
                gripper_width_m=float(pose7_after[6]),
                ee_pose=tuple(float(v) for v in pose7_after[:6]),
                error_code=err_c,
                error_detail=err_d,
            )

        # If the block slipped out during the preloaded vertical lift and the jaws stopped inside the
        # (EMPTY_CLOSED_MAX_M, HELD_GRIP_MIN_M) ambiguity band near lift_target_grip_m, close in-place
        # to 0.005 m so an empty gripper cleanly reaches <= EMPTY_CLOSED_MAX_M for auto-retry.
        if grip_measured > EMPTY_CLOSED_MAX_M and EMPTY_CLOSED_MAX_M < float(pose7_after[6]) < HELD_GRIP_MIN_M:
            squeeze_seg = plan_vertical_lift_in_place(
                pose7_after,
                target_z_m=float(self.config.hover_z_m),
                target_grip_m=0.005,
            )
            ok_sq, _, _ = self._plan_and_send_segment(
                segment=squeeze_seg,
                phase_name="transport",
                refine=True,
                loop_index=4,
            )
            if ok_sq:
                _q, pose7_after = self._measured_pose7()

        # Read fresh post-lift overhead + wrist images for dual-evidence fusion.
        verdict, grip_final = self._verify_and_log_lift_check(
            obj, pose7=pose7_after, label_suffix="lift_check", log_prefix="D3 LIFT CHECK"
        )

        # If cam_high is occluded (empty) OR a held block hovering directly over source_cell photobombed cam_high,
        # safely move aside to CLEAR_OBS_POSE6 (keeping clamped preload if holding) and take a clear cam_high frame.
        need_empty_clear = (
            verdict.verdict == "HELD_UNCERTAIN"
            and grip_final <= EMPTY_CLOSED_MAX_M
            and verdict.evidence.get("visual_block_still_at_cell") is False
        )
        need_held_clear = (
            verdict.verdict == "HELD_UNCERTAIN"
            and HELD_GRIP_MIN_M <= grip_final <= HELD_GRIP_MAX_M
            and verdict.evidence.get("visual_block_still_at_cell") is True
        )
        if self.config.use_clear_obs_retreat and (need_empty_clear or need_held_clear):
            if need_held_clear:
                preload_m = float(
                    _grasp_cases.compute_adaptive_preload_m(grip_final, float(self.config.clamp_preload_m))
                )
                target_clear_grip = max(0.005, grip_final - preload_m)
            else:
                target_clear_grip = 0.068
            clear_seg = plan_brown_to_clear_obs(pose7_after, open_grip_m=target_clear_grip)
            ok_clr, _, _ = self._plan_and_send_segment(segment=clear_seg, phase_name="takeoff", refine=True, loop_index=4)
            if ok_clr:
                _q_clr, pose7_clr = self._measured_pose7()
                pose7_after = pose7_clr
                if float(pose7_clr[6]) <= EMPTY_CLOSED_MAX_M:
                    self._at_clear_obs = True
                verdict, grip_final = self._verify_and_log_lift_check(
                    obj, pose7=pose7_clr, label_suffix="lift_check_clear_obs", log_prefix="D3 LIFT CHECK CLEAR_OBS"
                )

        if verdict.verdict == "HELD_CONFIRMED":
            self._held_grip_m = grip_final
            self._holding_object = True
        elif verdict.verdict == "EMPTY_GRASP_CONFIRMED":
            self._held_grip_m = None
            self._holding_object = False
        else:
            # Uncertain physical contact: retain conservative held flag so no unsafe go_home runs!
            self._holding_object = True

        return PhaseStepResult(
            ok=True,
            phase=ExecutionPhase.LIFTING_CHECK,
            task_generation=0,
            observation_id=obj.observation_id,
            observation_ts_mono=self._clock(),
            gripper_width_m=grip_final,
            ee_pose=tuple(float(v) for v in pose7_after[:6]),
            visual_block_on_table=verdict.evidence.get("visual_block_still_at_cell"),
        )

    def _next_available_slot(self) -> int | None:
        cap = int(self.config.max_brown_slots)
        if not self._allow_slot_recycling:
            return self._placed_slots_used if self._placed_slots_used < cap else None
        with self._state_lock:
            for idx in range(cap):
                if idx not in self._occupied_pad_slots:
                    return idx
        return None

    def sync_active_grid_colors(self, colors: set[str]) -> None:
        """Free recycled pad slots whose block colors are freshly observed back on the white grid."""
        if not self._allow_slot_recycling:
            return
        norm = {str(c).lower().strip() for c in colors}
        with self._state_lock:
            for idx in [k for k, col in self._occupied_pad_slots.items() if col in norm]:
                self._occupied_pad_slots.pop(idx, None)

    @staticmethod
    def _is_block_at_wrist_fingers(wrist_bytes: bytes | None, *, target_color: str) -> bool:
        """Return True when target block OBB is centered between the gripper fingertips in cam_right_wrist."""
        if wrist_bytes is None:
            return False
        obb = _grasp_cases.detect_wrist_block_obb(wrist_bytes, target_color=target_color)
        if obb is None:
            return False
        return bool(
            290.0 <= float(obb["cx"]) <= 360.0
            and 290.0 <= float(obb["cy"]) <= 355.0
            and min(float(obb.get("major_px", 0.0)), float(obb.get("minor_px", 0.0))) >= 42.0
        )

    def _handle_potential_slippage(
        self,
        obj: ObjectRecord,
        *,
        pose7: np.ndarray,
        held: float,
        wrist_bytes: bytes | None = None,
    ) -> tuple[bool, float, str | None, str | None]:
        """Verify whether a >4mm grip drop is tighter pad settling (still held) or an empty drop to recover."""
        measured = float(pose7[6])
        if measured >= held - 0.004:
            return True, held, None, None
        eff_preload = float(
            _grasp_cases.compute_adaptive_preload_m(held, float(self.config.clamp_preload_m))
        )
        clamp_target = max(0.005, held - eff_preload)
        if wrist_bytes is None:
            try:
                _payload, raw_chk = self._read_live_observation(task=format_single_target_task(obj))
                wrist_bytes = raw_chk["wrist_bytes"]
            except Exception:
                wrist_bytes = None
        if (
            measured >= max(HELD_GRIP_MIN_M, 0.024)
            and measured >= clamp_target + 0.0045
            and self._is_block_at_wrist_fingers(wrist_bytes, target_color=obj.color)
        ):
            print(
                f"[D3 CLAMP SETTLED] obj={obj.object_id} grip settled {held:.4f}->{measured:.4f}m "
                f"(clamp_target={clamp_target:.4f}m) with block centered in fingertips; continuing",
                flush=True,
            )
            self._held_grip_m = measured
            return True, measured, None, None

        print(
            f"[D3 SLIP RECOVER] obj={obj.object_id} measured={measured:.4f}<held-4mm ({held:.4f}); "
            "block absent from fingertips -> retracting open to CLEAR_OBS for re-observe",
            flush=True,
        )
        clear_seg = plan_brown_to_clear_obs(pose7, open_grip_m=0.068)
        ok_rec, err_c, err_d = self._plan_and_send_segment(
            segment=clear_seg,
            phase_name="takeoff",
            refine=True,
            loop_index=6,
        )
        if ok_rec:
            self._at_clear_obs = True
            self._held_grip_m = None
            self._holding_object = False
            return (
                False,
                measured,
                "block_slipped_recovered_empty",
                f"measured={measured:.4f}<held-4mm;retracted_to_clear_obs",
            )
        return False, measured, "block_slippage_detected", f"measured={measured:.4f}<held-4mm"

    def _run_transporting(self, obj: ObjectRecord) -> tuple[bool, int | None, str | None, str | None]:
        _q, pose7 = self._measured_pose7()
        held = float(self._held_grip_m if self._held_grip_m is not None else pose7[6])
        keep_going, held, slip_c, slip_d = self._handle_potential_slippage(obj, pose7=pose7, held=held)
        if not keep_going:
            return False, None, slip_c, slip_d
        slot_idx = self._next_available_slot()
        if slot_idx is None:
            return (
                False,
                None,
                "brown_pad_capacity_exceeded",
                f"slots_used={self._placed_slots_used},occupied={self._occupied_pad_slots}",
            )
        # Fly every block to SDK-verified slot_index=0 hover ([0.348, 0.216]) where the entire cork pad
        # sits cleanly in cam_right_wrist, then separate slots inside the pad during _run_placing().
        seg = plan_post_grasp_slot_transport(
            pose7,
            held_grip_m=held,
            slot_index=0,
            preload_m=float(self.config.clamp_preload_m),
        )
        ok, err_c, err_d = self._plan_and_send_segment(segment=seg, phase_name="transport", refine=True, loop_index=4)
        return ok, (slot_idx if ok else None), err_c, err_d

    @staticmethod
    def _compute_pad_slot_release_xy(visual_end_xy, *, slot_index: int) -> np.ndarray:
        """Compute a bounded, non-overlapping release XY strictly inside the brown cork pad."""
        if slot_index < 0 or slot_index >= len(_PAD_INTERIOR_SLOT_OFFSETS_XY_M):
            raise ValueError(
                f"Brown pad slot_index={slot_index} exceeds interior capacity ({len(_PAD_INTERIOR_SLOT_OFFSETS_XY_M)})"
            )
        center_xy = np.clip(np.asarray(visual_end_xy, dtype=float)[:2], _PAD_CENTER_CLAMP_XY_LOW, _PAD_CENTER_CLAMP_XY_HIGH)
        offset = np.asarray(_PAD_INTERIOR_SLOT_OFFSETS_XY_M[slot_index], dtype=float)
        return np.clip(center_xy + offset, _PAD_INTERIOR_XY_LOW, _PAD_INTERIOR_XY_HIGH)

    def _run_placing(self, obj: ObjectRecord) -> tuple[bool, str | None, str | None]:
        task = format_single_target_task(obj)
        slot_idx = int(obj.destination_slot if obj.destination_slot is not None else self._placed_slots_used)
        held = float(self._held_grip_m if self._held_grip_m is not None else 0.028)
        release_threshold = max(0.040, min(0.070, held + 0.010))

        for chunk_i in range(2):
            if self._hold_event.is_set():
                return False, "hold_requested", "Pause or cancel requested"
            payload, raw, actions = self._query_policy_actions(task=task)
            _q, pose7 = self._measured_pose7()
            keep_going, held, slip_c, slip_d = self._handle_potential_slippage(
                obj,
                pose7=pose7,
                held=held,
                wrist_bytes=raw["wrist_bytes"],
            )
            if not keep_going:
                return False, slip_c, slip_d
            release_threshold = max(0.040, min(0.070, held + 0.010))

            target_obb = _grasp_cases.detect_wrist_target_obb(raw["wrist_bytes"])
            if target_obb is not None:
                self._last_place_target_obb = dict(target_obb)
            diag = self._action_diagnostics(pose7, actions)
            print(
                f"[D3 PLACE step={chunk_i}] obj={obj.object_id} slot={slot_idx} z={pose7[2]:.4f} "
                f"raw0_gap_mm={diag['raw0_gap_mm']:.1f} rot={diag['max_rot_deg']:.1f}deg "
                f"grip=[{diag['grip_min']:.4f},{diag['grip_max']:.4f}] held={held:.4f}",
                flush=True,
            )
            try:
                seg, _clip, _splits = _grasp_cases.condition_placement_actions(
                    pose7,
                    actions,
                    held_grip=held,
                    max_position_m=float(self.config.placement_bridge_max_mm) / 1000.0,
                    preload_m=float(self.config.clamp_preload_m),
                    target_obb=target_obb,
                    trim_release_tail=True,
                )
            except ValueError:
                self._save_snapshot(
                    f"{obj.object_id}_place_{chunk_i}_rejected",
                    payload,
                    raw_actions=actions,
                    pose7=pose7,
                )
                raise

            released = bool(np.any(seg[:, 6] >= release_threshold))
            # Smoothly offset ONLY on release chunks around the visually centered brown pad target
            # (0.0 at chunk[0] -> 1.0 at tail) so multi-block slots stay inside the cork pad.
            if released and len(seg) > 1:
                target_xy = self._compute_pad_slot_release_xy(seg[-1, :2], slot_index=slot_idx)
                tail_delta_xy = np.clip(target_xy - seg[-1, :2], -0.035, 0.035)
                ramp = np.linspace(0.0, 1.0, len(seg), dtype=float)[:, None]
                seg[:, :2] = seg[:, :2] + ramp * tail_delta_xy[None, :]
                print(
                    f"[D3 PLACE SLOT] obj={obj.object_id} slot={slot_idx} "
                    f"target_xy=[{target_xy[0]:.4f},{target_xy[1]:.4f}] z_end={seg[-1, 2]:.4f}",
                    flush=True,
                )

            seg, _trimmed = _grasp_cases.trim_segment_idle_tail(
                seg,
                keep_tail_hold=(4 if released else 2),
                min_points=14,
            )
            phase_name = "place" if released else "transport"
            ok, err_c, err_d = self._plan_and_send_segment(
                segment=seg,
                phase_name=phase_name,
                first_camera_mono=raw["first_camera_mono"],
                refine=True,
                loop_index=5 + chunk_i,
                max_first_step_m=float(self.config.placement_bridge_max_mm) / 1000.0,
                payload_for_snapshot=payload,
                snapshot_label=f"{obj.object_id}_place_{chunk_i}",
            )
            if not ok:
                return False, err_c, err_d
            if released:
                return True, None, None
        return False, "placement_no_release_predicted", "Policy did not issue gripper open within placement chunks"

    def _run_releasing_check(self, obj: ObjectRecord) -> PhaseStepResult:
        _q, pose7 = self._measured_pose7()
        held = float(self._held_grip_m if self._held_grip_m is not None else 0.028)
        payload_chk, raw = self._read_live_observation(task=format_single_target_task(obj))
        verdict = verify_live_placement_outcome(
            float(pose7[6]),
            held_grip_m=held,
            wrist_post_release=raw["wrist_bytes"],
            target_color=obj.color,
            require_target_color=True,
            fallback_target_obb=self._last_place_target_obb,
        )
        for _retry in range(2):
            ev_r = verdict.evidence
            if (
                not verdict.confirmed
                and ev_r.get("reason") == "missing_brown_target_or_block_obb_after_release"
                and bool(ev_r.get("gripper_opened_ok", False))
                and not bool(ev_r.get("still_clutched_near_fingers", False))
            ):
                self._sleep(0.6)
                payload_chk, raw = self._read_live_observation(task=format_single_target_task(obj))
                _q, pose7 = self._measured_pose7()
                verdict = verify_live_placement_outcome(
                    float(pose7[6]),
                    held_grip_m=held,
                    wrist_post_release=raw["wrist_bytes"],
                    target_color=obj.color,
                    require_target_color=True,
                    fallback_target_obb=self._last_place_target_obb,
                )
            else:
                break
        self._save_snapshot(f"{obj.object_id}_release_check", payload_chk, None, pose7=pose7)
        used_slot = int(obj.destination_slot if obj.destination_slot is not None else self._placed_slots_used)
        ev = verdict.evidence
        blk = ev.get("block_obb") or {}
        tgt = ev.get("target_obb") or {}
        print(
            f"[D3 RELEASE CHECK] obj={obj.object_id} slot={used_slot} grip={float(pose7[6]):.4f} "
            f"confirmed={verdict.confirmed} at_grip={bool(ev.get('still_clutched_near_fingers', False))} "
            f"blk=({blk.get('cx')},{blk.get('cy')}) tgt=({tgt.get('cx')},{tgt.get('cy')}) "
            f"dist_px={ev.get('dist_px')} max_px={ev.get('max_allowed_dist_px')} reason={ev.get('reason', 'ok')}",
            flush=True,
        )
        if not verdict.confirmed:
            self._holding_object = True
            return PhaseStepResult(
                ok=True,
                phase=ExecutionPhase.RELEASING_CHECK,
                task_generation=0,
                observation_id=obj.observation_id,
                observation_ts_mono=self._clock(),
                gripper_width_m=float(pose7[6]),
                ee_pose=tuple(float(v) for v in pose7[:6]),
                visual_block_in_brown_zone=False,
                visual_block_at_gripper=bool(verdict.evidence.get("still_clutched_near_fingers", False)),
                destination_slot=used_slot,
            )

        # Confirmed placed & empty! Safely lift clear above brown pad before reporting checkpoint settle.
        self._placed_slots_used += 1
        with self._state_lock:
            self._occupied_pad_slots[used_slot] = obj.color.lower()
        self._held_grip_m = None
        self._holding_object = False
        self._last_place_target_obb = None
        self._placement_completed_any = True

        lift_seg = plan_vertical_lift_in_place(
            pose7,
            target_z_m=max(float(self.config.hover_z_m), float(self.config.min_safe_home_z_m) + 0.025),
            target_grip_m=float(pose7[6]),
        )
        ok_lift, err_c, err_d = self._plan_and_send_segment(segment=lift_seg, phase_name="takeoff", refine=True, loop_index=6)
        if ok_lift and self.config.use_clear_obs_retreat:
            _q2, pose7_lifted = self._measured_pose7()
            clear_seg = plan_brown_to_clear_obs(pose7_lifted, open_grip_m=max(float(pose7_lifted[6]), 0.068))
            ok_clr, err_c2, err_d2 = self._plan_and_send_segment(segment=clear_seg, phase_name="takeoff", refine=True, loop_index=6)
            if ok_clr:
                self._at_clear_obs = True
            else:
                ok_lift, err_c, err_d = False, err_c2, err_d2

        _q3, pose7_end = self._measured_pose7()
        return PhaseStepResult(
            ok=ok_lift,
            phase=ExecutionPhase.RELEASING_CHECK,
            task_generation=0,
            observation_id=obj.observation_id,
            observation_ts_mono=self._clock(),
            gripper_width_m=float(pose7_end[6]),
            ee_pose=tuple(float(v) for v in pose7_end[:6]),
            visual_block_in_brown_zone=True,
            visual_block_at_gripper=bool(verdict.evidence.get("still_clutched_near_fingers", False)),
            destination_slot=used_slot,
            error_code=err_c,
            error_detail=err_d,
        )

    def run_phase_step(
        self,
        *,
        obj: ObjectRecord,
        phase: ExecutionPhase,
        task_generation: int,
        attempt_number: int,
        now_mono: float,
    ) -> PhaseStepResult:
        with self._exec_lock:
            self._hold_event.clear()
            self._ensure_hardware_attached()
            try:
                if phase == ExecutionPhase.APPROACHING_EMPTY:
                    ok, err_c, err_d = self._run_approaching_empty(obj)
                elif phase == ExecutionPhase.DESCENDING_CONTACT:
                    ok, err_c, err_d = self._run_descending_contact(obj)
                elif phase == ExecutionPhase.GRASPING:
                    ok, err_c, err_d = self._run_grasping(obj)
                elif phase == ExecutionPhase.LIFTING_CHECK:
                    res = self._run_lifting_check(obj)
                    return PhaseStepResult(**{**res.__dict__, "task_generation": task_generation})
                elif phase == ExecutionPhase.TRANSPORTING:
                    ok, slot_idx, err_c, err_d = self._run_transporting(obj)
                    _q, pose7 = self._measured_pose7()
                    return PhaseStepResult(
                        ok=ok,
                        phase=phase,
                        task_generation=task_generation,
                        observation_id=obj.observation_id,
                        observation_ts_mono=self._clock(),
                        gripper_width_m=float(pose7[6]),
                        ee_pose=tuple(float(v) for v in pose7[:6]),
                        destination_slot=slot_idx,
                        error_code=err_c,
                        error_detail=err_d,
                    )
                elif phase == ExecutionPhase.PLACING:
                    ok, err_c, err_d = self._run_placing(obj)
                elif phase == ExecutionPhase.RELEASING_CHECK:
                    res = self._run_releasing_check(obj)
                    return PhaseStepResult(**{**res.__dict__, "task_generation": task_generation})
                else:
                    ok, err_c, err_d = True, None, None
            except TimeoutError as exc:
                _q, pose7 = self._measured_pose7()
                return PhaseStepResult(
                    ok=False,
                    phase=phase,
                    task_generation=task_generation,
                    observation_id=obj.observation_id,
                    observation_ts_mono=self._clock(),
                    gripper_width_m=float(pose7[6]),
                    ee_pose=tuple(float(v) for v in pose7[:6]),
                    error_code="inference_timeout",
                    error_detail=str(exc),
                )
            except ValueError as exc:
                print(f"[D3 GUARD REJECT] phase={phase.value} obj={obj.object_id} detail={exc}", flush=True)
                _q, pose7 = self._measured_pose7()
                return PhaseStepResult(
                    ok=False,
                    phase=phase,
                    task_generation=task_generation,
                    observation_id=obj.observation_id,
                    observation_ts_mono=self._clock(),
                    gripper_width_m=float(pose7[6]),
                    ee_pose=tuple(float(v) for v in pose7[:6]),
                    error_code="guard_rejected",
                    error_detail=str(exc),
                )

            _q, pose7 = self._measured_pose7()
            return PhaseStepResult(
                ok=ok,
                phase=phase,
                task_generation=task_generation,
                observation_id=obj.observation_id,
                observation_ts_mono=self._clock(),
                gripper_width_m=float(pose7[6]),
                ee_pose=tuple(float(v) for v in pose7[:6]),
                error_code=err_c,
                error_detail=err_d,
            )

    def run_empty_grasp_recovery(
        self,
        *,
        obj: ObjectRecord,
        task_generation: int,
        now_mono: float,
    ) -> PhaseStepResult:
        """Retract safely vertically + open gripper back to hover cell on EMPTY_GRASP_CONFIRMED."""
        with self._exec_lock:
            self._hold_event.clear()
            self._ensure_hardware_attached()
            cell = obj.grid_cell or "R8"
            _q, pose7 = self._measured_pose7()
            if self._at_clear_obs:
                seg = plan_clear_obs_to_grid_hover(pose7, target_cell=cell, hover_z_m=float(self.config.hover_z_m))
            else:
                seg = plan_empty_grasp_retreat(pose7, target_cell=cell, hover_z_m=float(self.config.hover_z_m))
            ok, err_c, err_d = self._plan_and_send_segment(segment=seg, phase_name="takeoff", refine=True, loop_index=3)
            self._at_clear_obs = False
            self._held_grip_m = None
            self._holding_object = False
            _q2, pose7_after = self._measured_pose7()
            return PhaseStepResult(
                ok=ok,
                phase=ExecutionPhase.RECOVERING,
                task_generation=task_generation,
                observation_id=obj.observation_id,
                observation_ts_mono=self._clock(),
                gripper_width_m=float(pose7_after[6]),
                ee_pose=tuple(float(v) for v in pose7_after[:6]),
                error_code=err_c,
                error_detail=err_d,
            )

    def begin_new_batch(self, *, allow_slot_recycling: bool = False) -> None:
        """Reset per-batch software slot/held bookkeeping without homing or detaching hardware."""
        with self._state_lock:
            self._placed_slots_used = 0
            self._allow_slot_recycling = bool(allow_slot_recycling)
            self._occupied_pad_slots.clear()
            self._returned_home_after_batch = False
            self._held_grip_m = None
            self._holding_object = False
            self._last_place_target_obb = None
            self._hold_event.clear()
            self._pregrasp_preempt_event.clear()

    def _execute_safe_home_motion_locked(self, pose7: np.ndarray, *, log_prefix: str) -> tuple[bool, str | None, str | None]:
        """Lift above min_safe_home_z_m if low, then invoke go_home() and settle under _exec_lock."""
        if float(pose7[2]) < float(self.config.min_safe_home_z_m):
            lift_seg = plan_vertical_lift_in_place(
                pose7,
                target_z_m=max(float(self.config.hover_z_m), float(self.config.min_safe_home_z_m) + 0.025),
                target_grip_m=float(pose7[6]),
            )
            ok_lift, err_c, err_d = self._plan_and_send_segment(
                segment=lift_seg, phase_name="takeoff", refine=True, loop_index=6
            )
            if not ok_lift:
                return False, err_c, err_d
            _q, pose7 = self._measured_pose7()
            if float(pose7[2]) < float(self.config.min_safe_home_z_m):
                return False, "unsafe_home_suppressed", f"EE z={pose7[2]:.4f}m < min_safe_home_z_m"
        print(log_prefix, flush=True)
        self._robot.go_home()
        self._sleep(float(self.config.robot_init_settle_s))
        self._at_clear_obs = False
        self._returned_home_after_batch = True
        return True, None, None

    def return_home_on_completion(self) -> tuple[bool, str | None, str | None]:
        """Return BOTH arms safely to calibrated Home ONLY after verified empty-handed batch completion."""
        if not (self.config.return_home_on_batch_complete and self.config.confirm_exit_home):
            return False, None, None
        with self._exec_lock:
            if self._robot is None or self._shutdown_done:
                return False, None, None
            if self._holding_object or self._held_grip_m is not None:
                return False, "unsafe_home_suppressed", "Cannot return home while holding object"
            _q, pose7 = self._measured_pose7()
            if float(pose7[6]) < 0.045:
                return False, "unsafe_home_suppressed", f"Gripper width {pose7[6]:.4f}m < 0.045m"
            if bool(np.linalg.norm(pose7[:3] - _HOME_XYZ) < 0.035):
                self._at_clear_obs = False
                self._returned_home_after_batch = True
                return True, None, None
            return self._execute_safe_home_motion_locked(
                pose7,
                log_prefix="[D3 BATCH HOME] Safe normal completion: returning BOTH arms to Home",
            )

    def manual_return_home(self) -> tuple[bool, str | None, str | None]:
        """Operator-confirmed explicit Home button: requires idle batch + verified open gripper (>= 45 mm)."""
        with self._exec_lock:
            if self._shutdown_done:
                return False, "hardware_shutdown", "Robot hardware session is already shut down"
            if self._robot is None:
                return True, None, "Robot hardware not attached yet; already idle"
            if not self.config.confirm_exit_home:
                return False, "exit_home_not_confirmed", "Restart with --confirm-exit-home to enable Return Home"
            _q, pose7 = self._measured_pose7()
            grip_m = float(pose7[6])
            if grip_m < 0.045:
                return (
                    False,
                    "unsafe_home_gripper_closed",
                    f"Gripper width {grip_m:.4f}m < 0.045m; release or clear block before Home",
                )
            self._holding_object = False
            self._held_grip_m = None
            if bool(np.linalg.norm(pose7[:3] - _HOME_XYZ) < 0.035):
                self._at_clear_obs = False
                self._returned_home_after_batch = True
                return True, None, "Both arms already at Home."
            ok, err_c, err_d = self._execute_safe_home_motion_locked(
                pose7,
                log_prefix="[D3 OPERATOR HOME] Open gripper verified; returning BOTH arms to Home",
            )
            if not ok:
                return False, err_c, err_d
            return True, None, "Both arms returned safely to Home."

    def request_pregrasp_preempt(self) -> None:
        """Signal between empty-hand pre-grasp chunks before Chunk 3 GRASPING starts."""
        self._pregrasp_preempt_event.set()

    def clear_pregrasp_preempt(self) -> None:
        """Clear pending pre-grasp preempt signal."""
        self._pregrasp_preempt_event.clear()

    def preempt_to_safe_hover(self) -> tuple[bool, str | None, str | None]:
        """Lift open empty gripper back to hover_z_m before pre-grasp target switch."""
        with self._exec_lock:
            self._pregrasp_preempt_event.clear()
            if self._robot is None or self._shutdown_done:
                return True, None, None
            if self._holding_object or self._held_grip_m is not None:
                return False, "unsafe_preempt_holding_object", "Cannot preempt target while holding object"
            _q, pose7 = self._measured_pose7()
            if float(pose7[6]) < 0.045:
                return False, "unsafe_preempt_gripper_not_open", f"Gripper width {pose7[6]:.4f}m < 0.045m"
            hover_z = float(self.config.hover_z_m)
            if float(pose7[2]) < hover_z - 0.015:
                lift_seg = plan_vertical_lift_in_place(
                    pose7,
                    target_z_m=hover_z,
                    target_grip_m=float(max(pose7[6], 0.072)),
                )
                ok_lift, err_c, err_d = self._plan_and_send_segment(
                    segment=lift_seg, phase_name="takeoff", refine=True, loop_index=3
                )
                if not ok_lift:
                    return False, err_c, err_d
            self._at_clear_obs = False
            return True, None, None

    def request_phase_hold(self, *, current_phase: ExecutionPhase, reason: str) -> bool:
        """Keep motors energized holding current pose without closing or homing hardware."""
        self._hold_event.set()
        return True

    def cancel_future_queue(self, *, generation: int, reason: str) -> int:
        with self._state_lock:
            self._cancelled_generations.append((generation, reason))
        self._hold_event.set()
        return 1

    def shutdown_hardware(
        self,
        *,
        safe_normal_completion: bool = False,
        audit_json_path: Path | None = None,
    ) -> dict[str, Any] | None:
        """Run execute_candidate_finally cleanly ONLY if hardware was attached during this session."""
        self._hold_event.set()
        with self._exec_lock:
            with self._state_lock:
                if self._shutdown_done or (self._robot is None and self._image_client is None):
                    self._shutdown_done = True
                    return None
                self._shutdown_done = True
                exit_reason = (
                    "normal_completion"
                    if (safe_normal_completion and not self._holding_object)
                    else "cancelled_or_interrupted"
                )
            sent_chunks_for_exit = self._sent_chunks
            eff_min_safe_z = float(self.config.min_safe_home_z_m)
            if self._returned_home_after_batch and self._robot is not None and not self._holding_object:
                try:
                    _q_cur, pose7_cur = self._measured_pose7()
                    if bool(np.linalg.norm(pose7_cur[:3] - _HOME_XYZ) < 0.035) and float(pose7_cur[6]) >= 0.045:
                        sent_chunks_for_exit = 0
                        eff_min_safe_z = min(eff_min_safe_z, 0.165)
                except Exception:
                    pass
            audit = _guarded.execute_candidate_finally(
                robot=self._robot,
                image_client=self._image_client,
                trace_session=None,
                sent_chunks=sent_chunks_for_exit,
                last_install=self._last_install_mono,
                last_duration=self._last_duration_s,
                switch_delay=float(self.config.switch_delay_s),
                exit_reason=exit_reason,
                policy=self.config.exit_home_policy,
                confirm_exit_home=bool(self.config.confirm_exit_home),
                holding_object=bool(self._holding_object),
                min_safe_home_z_m=eff_min_safe_z,
                placement_completed=bool(self._placement_completed_any),
                margin=float(self.config.margin_rad),
                max_soft_clip_rad=float(self.config.max_soft_clip_rad),
                smooth_waypoints=True,
                waypoint_smooth_alpha=0.20,
                waypoint_smooth_passes=1,
                rate=float(self.config.rate_hz),
                unsafe_exit_action=self.config.unsafe_exit_action,
                lifecycle_audit_json=(str(audit_json_path) if audit_json_path is not None else None),
                sleep_fn=self._sleep,
                hold_confirm_fn=self._hold_confirm_fn,
            )
            with self._state_lock:
                self._robot = None
                self._image_client = None
                self._hardware_attached = False
            return audit
