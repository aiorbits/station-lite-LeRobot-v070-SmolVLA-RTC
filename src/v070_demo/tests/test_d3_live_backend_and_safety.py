"""Stage D3 comprehensive offline hardware-adapter, safety-gate, and closed-loop tests."""

from __future__ import annotations

import base64
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Callable
from urllib import request

import cv2
import numpy as np

from v070_demo.candidate_bridge import D2_BUNDLE_SHA256, verify_sdk_dryrun_report
from v070_demo.candidate_import import import_build_grasp_replay_cases, import_right7_guarded
from v070_demo.contracts import CommandStatus, CommandType, ContactHeldState, ExecutionPhase, SafetyBudgets, SessionState
from v070_demo.live_robot_runner import LiveHardwareConfig, LiveRight7HardwareBridge
from v070_demo.perception_adapter import (
    BROWN_TARGET_HOVER_POSE6,
    GRID_HOVER_POSES,
    HIGH_CAM_GRID_CENTERS_PX,
    detect_all_high_cam_blocks,
)
from v070_demo.run_panel import main as run_panel_main
from v070_demo.session_service import SessionService
from v070_demo.trajectory_planner import CLEAR_OBS_POSE6, build_demo_d3_sdk_dryrun_bundle
from v070_demo.web_panel import DemoPanelServer

_guarded = import_right7_guarded()
_grasp_cases = import_build_grasp_replay_cases()

HOME_POSE6 = np.array([0.1272, 0.0002, 0.1776, 0.0, 0.0, 0.0], dtype=float)
CENTER_Q = np.array([0.45, 1.45, -0.85, 0.35, 0.22, -0.75], dtype=float)
CENTER_POSE6 = np.array([0.275, 0.140, 0.180, -0.33, 1.04, 1.04], dtype=float)
K_POSE_TO_Q = np.array([1.2, 1.2, 1.2, 0.35, 0.35, 0.35], dtype=float)

COLOR_BGR = {
    "red": (15, 15, 225),
    "yellow": (15, 225, 225),
    "blue": (225, 60, 15),
}


def _pose6_to_q(pose6: np.ndarray) -> np.ndarray:
    arr = np.asarray(pose6, dtype=float).reshape(6)
    return CENTER_Q + (arr - CENTER_POSE6) * K_POSE_TO_Q


def _q_to_pose6(q: np.ndarray) -> np.ndarray:
    arr = np.asarray(q, dtype=float).reshape(6)
    return CENTER_POSE6 + (arr - CENTER_Q) / K_POSE_TO_Q


def _encode_jpeg_bytes(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    assert ok
    return bytes(buf)


def _render_high_cam(grid_blocks: dict[str, str], *, small_blue: bool = False) -> bytes:
    bgr = np.full((480, 640, 3), 232, dtype=np.uint8)
    for cell, color in grid_blocks.items():
        cx, cy = HIGH_CAM_GRID_CENTERS_PX[cell]
        if small_blue and color == "blue":
            # Matches real station 20261113_live cyan/blue area ~107px with ~7px center offset.
            cx_i, cy_i = int(round(cx)) - 5, int(round(cy)) + 4
            cv2.rectangle(bgr, (cx_i - 5, cy_i - 5), (cx_i + 5, cy_i + 5), COLOR_BGR[color], -1)
        else:
            cx_i, cy_i = int(round(cx)), int(round(cy))
            cv2.rectangle(bgr, (cx_i - 6, cy_i - 6), (cx_i + 6, cy_i + 6), COLOR_BGR[color], -1)
    return _encode_jpeg_bytes(bgr)


def _render_wrist_cam(
    *,
    block_color: str | None,
    show_brown_target: bool = False,
    target_uv: tuple[float, float] = (320.0, 318.0),
) -> bytes:
    hsv = np.zeros((480, 640, 3), dtype=np.uint8)
    hsv[:, :] = (0, 0, 200)
    if show_brown_target:
        # Muted woodbrown S=56 (<70 red threshold) stays inside detect_wrist_target_obb (S in 40..220, H in 8..28).
        cv2.rectangle(hsv, (285, 205), (420, 360), (17, 56, 125), -1)
    if block_color is not None:
        if show_brown_target:
            blk_hsv = {"red": (4, 230, 225), "yellow": (26, 220, 225), "blue": (106, 210, 215)}[block_color]
            cv2.rectangle(hsv, (315, 248), (372, 305), blk_hsv, -1)
        else:
            cx, cy = int(round(target_uv[0])), int(round(target_uv[1]))
            blk_hsv = {"red": (4, 230, 225), "yellow": (26, 220, 225), "blue": (106, 210, 215)}[block_color]
            cv2.rectangle(hsv, (cx - 28, cy - 28), (cx + 28, cy + 28), blk_hsv, -1)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return _encode_jpeg_bytes(bgr)


class _FakeInnerArm:
    def plan_joint_waypoints_with_gripper(self, q_start, waypoints, grippers, *, time_sec: float):
        arr = np.asarray(waypoints, dtype=float)
        n = max(12, len(arr) * 4)
        out = np.zeros((n, 15), dtype=float)
        t_grid = np.linspace(0.0, max(0.1, float(time_sec)), n)
        out[:, 0] = t_grid
        qs = np.vstack([np.asarray(q_start, dtype=float).reshape(6), arr])
        for j in range(6):
            out[:, 1 + j] = np.interp(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, len(qs)), qs[:, j])
        return out


class _FakeRightArm:
    def __init__(self, state: dict) -> None:
        self._state = state
        self.arm = _FakeInnerArm()

    def get_joint_positions(self) -> list[float]:
        return _pose6_to_q(self._state["pose6"]).tolist()

    def get_gripper_distance(self) -> float:
        return float(self._state["grip_m"])

    def get_ee_pose_euler(self) -> tuple[list[float], list[float]]:
        p = self._state["pose6"]
        return p[:3].tolist(), p[3:6].tolist()

    def solve_ik_euler(self, pos, euler, seed):
        pose = np.array([pos[0], pos[1], pos[2], euler[0], euler[1], euler[2]], dtype=float)
        return _pose6_to_q(pose).tolist(), True

    def update_joint_waypoint_chunk_with_gripper(self, waypoints, grippers, *, time_sec: float, switch_delay_sec: float = 0.04):
        last_q = np.asarray(waypoints[-1], dtype=float).reshape(6)
        self._state["pose6"] = _q_to_pose6(last_q)
        # SDK update_joint_waypoint_chunk_with_gripper takes normalized fraction [0, 1];
        # get_gripper_distance() reports meters [0, GRIPPER_MAX_M=0.085].
        target_grip = float(grippers[-1]) * float(_guarded.GRIPPER_MAX_M)
        # Simulate physical block width (~28 mm) vs empty closure (~8 mm) when grasping near tabletop.
        if target_grip < 0.015:
            st = self._state
            cell = st.get("active_cell")
            hover_xy = GRID_HOVER_POSES[cell][:2] if cell in GRID_HOVER_POSES else st["pose6"][:2]
            near_block = bool(
                cell in st["grid_blocks"]
                and np.linalg.norm(st["pose6"][:2] - hover_xy) < 0.030
                and st["pose6"][2] <= 0.166
            )
            fail_once_cells = st.setdefault("fail_grasp_once", set())
            if near_block and cell in fail_once_cells:
                fail_once_cells.remove(cell)
                st["grip_m"] = 0.008  # Empty grasp! Leave block on tabletop.
                st["held_color"] = None
            elif near_block:
                st["held_color"] = st["grid_blocks"].pop(cell)
                st["grip_m"] = 0.0282  # Real 27mm block closed width.
            else:
                st["grip_m"] = 0.008
                st["held_color"] = None
        else:
            if self._state.get("held_color") is not None and target_grip < 0.045:
                # Rigid 27mm woodblock prevents jaws closing below physical block width (~0.0282m) under preload.
                self._state["grip_m"] = max(0.0282, target_grip)
            else:
                self._state["grip_m"] = target_grip
                if target_grip >= 0.045 and self._state.get("held_color") is not None:
                    # Release block onto brown zone
                    self._state["placed_colors"].append(self._state["held_color"])
                    self._state["held_color"] = None
        self._state["sent_chunks"].append(
            {
                "waypoints": len(waypoints),
                "time_sec": time_sec,
                "end_grip": self._state["grip_m"],
                "end_pose6": self._state["pose6"].copy(),
            }
        )


class _SimulatedStationHardware:
    """Shared world state coordinating FakeDualTouchRobot, FakeImageClient, and Fake8011Server."""

    def __init__(self, initial_grid: dict[str, str], *, fail_grasp_once: set[str] | None = None, small_blue: bool = False) -> None:
        self.state = {
            "pose6": HOME_POSE6.copy(),
            "grip_m": 0.072,
            "grid_blocks": dict(initial_grid),
            "held_color": None,
            "placed_colors": [],
            "active_cell": None,
            "fail_grasp_once": set(fail_grasp_once or set()),
            "sent_chunks": [],
            "small_blue": small_blue,
            "robot_constructions": 0,
            "image_client_constructions": 0,
            "go_home_calls": 0,
            "disconnect_calls": 0,
        }

    def make_robot_cls(self):
        hw = self

        class _FakeDualTouchRobot:
            def __init__(self) -> None:
                hw.state["robot_constructions"] += 1
                hw.state["go_home_calls"] += 1
                hw.state["pose6"] = HOME_POSE6.copy()
                hw.state["grip_m"] = 0.072
                self.right_arm = _FakeRightArm(hw.state)

            def go_home(self) -> None:
                hw.state["go_home_calls"] += 1
                hw.state["pose6"] = HOME_POSE6.copy()

            def disconnect(self) -> None:
                hw.state["disconnect_calls"] += 1

        return _FakeDualTouchRobot

    def make_image_client_cls(self):
        hw = self

        class _FakeImageClient:
            def __init__(self, host: str = "127.0.0.1", auto_subscribe: bool = False) -> None:
                hw.state["image_client_constructions"] += 1
                self.host = host
                self.closed = False

            def start_subscriptions(self) -> None:
                pass

            def get_head_frame(self):
                return _render_high_cam(hw.state["grid_blocks"], small_blue=hw.state["small_blue"]), time.monotonic()

            def get_right_wrist_frame(self):
                st = hw.state
                pos = st["pose6"][:3]
                held = st["held_color"]
                # Near brown pad?
                if pos[0] >= 0.295 and pos[1] >= 0.175:
                    released_color = held or (st["placed_colors"][-1] if st["placed_colors"] else None)
                    return _render_wrist_cam(block_color=released_color, show_brown_target=True), time.monotonic()
                uv = _grasp_cases.wrist_target_uv_for_pose(
                    ee_pose6=st["pose6"],
                    ee_z_m=float(st["pose6"][2]),
                )
                if held is not None:
                    return _render_wrist_cam(block_color=held, target_uv=uv), time.monotonic()
                # Check if directly over any remaining block on tabletop
                best_color = None
                for cell, col in st["grid_blocks"].items():
                    hxy = GRID_HOVER_POSES[cell][:2]
                    if np.linalg.norm(pos[:2] - hxy) < 0.045:
                        best_color = col
                        st["active_cell"] = cell
                        break
                return _render_wrist_cam(block_color=best_color, target_uv=uv), time.monotonic()

            def close(self) -> None:
                self.closed = True

        return _FakeImageClient


class _MockServerState:
    def __init__(
        self,
        hw: _SimulatedStationHardware,
        *,
        checkpoint: str,
        delay_s: float = 0.0,
        on_predict: Callable[[], None] | None = None,
        initial_raw_offset_m: np.ndarray | None = None,
        omit_closure_for_grasp: bool = False,
    ) -> None:
        self.hw = hw
        self.checkpoint = checkpoint
        self.delay_s = delay_s
        self.on_predict = on_predict
        self.initial_raw_offset_m = initial_raw_offset_m
        self.omit_closure_for_grasp = omit_closure_for_grasp
        self.predict_calls = 0


def _start_mock_right7_server(mock_state: _MockServerState) -> ThreadingHTTPServer:
    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args) -> None:
            return

        def do_GET(self) -> None:
            if self.path != "/health":
                self.send_response(HTTPStatus.NOT_FOUND)
                self.end_headers()
                return
            payload = {
                "schema": _guarded.SCHEMA,
                "source_schema": _guarded.REQUEST_SCHEMA,
                "checkpoint": mock_state.checkpoint,
                "chunk_size": 50,
                "state_dim": 7,
                "action_dim": 7,
                "task": _guarded.TASK,
                "device": "cuda",
                "seed_base": 42,
                "seed_mode": "fixed",
                "hardware_connected": False,
            }
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self) -> None:
            if self.path != "/predict":
                self.send_response(HTTPStatus.NOT_FOUND)
                self.end_headers()
                return
            n = int(self.headers.get("Content-Length", "0"))
            req_body = json.loads(self.rfile.read(n).decode("utf-8"))
            if mock_state.delay_s > 0.0:
                time.sleep(mock_state.delay_s)
            if mock_state.on_predict is not None:
                mock_state.on_predict()
            mock_state.predict_calls += 1

            st = mock_state.hw.state
            pose = st["pose6"].copy()
            grip = float(st["grip_m"])
            task = str(req_body.get("task", ""))

            actions = np.zeros((50, 7), dtype=float)
            actions[:, 3:6] = pose[3:6]

            # Near brown pad with gripper closed -> generate placement & open chunk
            if pose[0] >= 0.295 and pose[1] >= 0.175 and grip < 0.045:
                goal_xy = BROWN_TARGET_HOVER_POSE6[:2]
                for i in range(50):
                    a = float(i + 1) / 50.0
                    actions[i, :2] = pose[:2] + a * (goal_xy - pose[:2])
                    actions[i, 2] = pose[2] + min(1.0, a * 1.4) * (0.153 - pose[2])
                    actions[i, 6] = grip if i < 28 else 0.068
            else:
                # Pick matching color cell from task prompt
                target_color = "yellow" if "yellow" in task else ("red" if "red" in task else "blue")
                cell = next((c for c, col in st["grid_blocks"].items() if col == target_color), None)
                goal_pose = GRID_HOVER_POSES[cell] if cell in GRID_HOVER_POSES else pose
                if cell is not None:
                    st["active_cell"] = cell
                if pose[2] > 0.170:
                    # Realistic SmolVLA hover/approach prediction: descend toward tabletop pre-grasp (~0.160m)
                    # with closure (<0.04) in chunk tail so condition_approach_actions enters allow_descent=True.
                    for i in range(50):
                        a = float(i + 1) / 50.0
                        actions[i, :2] = pose[:2] + a * (goal_pose[:2] - pose[:2])
                        actions[i, 2] = pose[2] + min(1.0, a * 1.4) * (0.159 - pose[2])
                        if mock_state.initial_raw_offset_m is not None:
                            taper = max(0.0, 1.0 - float(i) / 10.0)
                            actions[i, :3] = actions[i, :3] + taper * np.asarray(mock_state.initial_raw_offset_m, dtype=float)
                        actions[i, 3:6] = goal_pose[3:6]
                        actions[i, 6] = 0.070 if (mock_state.omit_closure_for_grasp or i < 34) else 0.002
                else:
                    # Aligned low pre-grasp -> grasp descent & close (<0.04).
                    for i in range(50):
                        a = float(i + 1) / 50.0
                        actions[i, :2] = goal_pose[:2]
                        actions[i, 2] = pose[2] + min(1.0, a * 1.5) * (0.146 - pose[2])
                        actions[i, 3:6] = goal_pose[3:6]
                        actions[i, 6] = 0.068 if (mock_state.omit_closure_for_grasp or i < 25) else 0.002

            resp = {
                "schema": _guarded.SCHEMA,
                "source_schema": _guarded.REQUEST_SCHEMA,
                "checkpoint": mock_state.checkpoint,
                "session_id": req_body["session_id"],
                "seq": req_body["seq"],
                "action_dim": 7,
                "hardware_connected": False,
                "seed_mode": "fixed",
                "seed_base": 42,
                "sampling_seed": 42,
                "infer_ms": 4.2,
                "actions": actions.tolist(),
            }
            raw = json.dumps(resp).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    return srv


def _write_dryrun_report(path: Path, *, sha256: str = D2_BUNDLE_SHA256, all_passed: bool = True, cleanup: bool = True) -> Path:
    report = {
        "schema": "v070_g4_sdk_dryrun_report_v1",
        "bundle_schema": "v070_g4_sdk_handoff_bundle_v1",
        "input_sha256": sha256,
        "interfaces": ["lo"],
        "hardware_connected": False,
        "sdk_version": "0.1.9",
        "cases_total": 10,
        "cases_passed": 10 if all_passed else 9,
        "cases_failed": 0 if all_passed else 1,
        "all_passed": all_passed,
        "cases": [{"name": f"case_{i}", "passed": True, "reject": []} for i in range(10)],
        "cleanup_returned": cleanup,
    }
    if not all_passed:
        report["cases"][0]["passed"] = False
        report["cases"][0]["reject"] = ["joint_step_limit"]
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


class TestD3LiveBackendAndSafety(unittest.TestCase):
    def test_verify_sdk_dryrun_report_enforces_hash_and_safety_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            good = _write_dryrun_report(Path(tmp) / "good.json")
            info = verify_sdk_dryrun_report(good)
            self.assertEqual(info["cases_passed"], 10)
            self.assertEqual(info["input_sha256"], D2_BUNDLE_SHA256)

            wrong_sha = _write_dryrun_report(Path(tmp) / "wrong_sha.json", sha256="deadbeef")
            with self.assertRaises(PermissionError):
                verify_sdk_dryrun_report(wrong_sha)

            failed = _write_dryrun_report(Path(tmp) / "failed.json", all_passed=False)
            with self.assertRaises(PermissionError):
                verify_sdk_dryrun_report(failed)

            no_cleanup = _write_dryrun_report(Path(tmp) / "nocleanup.json", cleanup=False)
            with self.assertRaises(PermissionError):
                verify_sdk_dryrun_report(no_cleanup)

    def test_perception_accepts_real_20261113_blue_area_and_offset(self) -> None:
        # Regresses real 20261113_live Finding: ~107px cyan/blue block shifted ~7px from R7 scored <0.70 previously.
        raw = _render_high_cam({"R3": "red", "R9": "yellow", "R7": "blue"}, small_blue=True)
        obs = detect_all_high_cam_blocks(raw, observation_id="calib_check", observation_ts_mono=time.monotonic())
        by_color = {o.color: o for o in obs}
        self.assertEqual(set(by_color), {"red", "yellow", "blue"})
        self.assertGreaterEqual(by_color["blue"].confidence, 0.78)
        self.assertTrue(by_color["blue"].grasp_pose_valid)

    def test_live_bridge_refuses_8010_and_zero_side_effect_before_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rep = _write_dryrun_report(Path(tmp) / "ok.json")
            with self.assertRaises(PermissionError):
                LiveRight7HardwareBridge(config=LiveHardwareConfig(server_url="http://127.0.0.1:8010"))

            hw = _SimulatedStationHardware({"R8": "yellow"})
            srv = _start_mock_right7_server(_MockServerState(hw, checkpoint=_guarded.EXPECTED_CHECKPOINT))
            port = srv.server_address[1]
            try:
                cfg = LiveHardwareConfig(
                    server_url=f"http://127.0.0.1:{port}",
                    dry_run_report_path=rep,
                    operator_admitted=True,
                    confirm_startup_home=True,
                    confirm_exit_home=True,
                    unsafe_exit_action="immediate_disable",
                    robot_init_settle_s=0.0,
                    settle_margin_s=0.0,
                    settle_drift_sleep_s=0.0,
                )
                bridge = LiveRight7HardwareBridge(
                    config=cfg,
                    robot_cls=hw.make_robot_cls(),
                    image_client_cls=hw.make_image_client_cls(),
                )
                bridge.admit_hardware()
                svc = SessionService(bridge)
                panel = DemoPanelServer(svc, host="127.0.0.1", port=0)
                panel.start()
                base = f"http://127.0.0.1:{panel.port}"
                try:
                    # Opening page, reading snapshot/frame, starting session & closing without batch -> ZERO robot init!
                    with request.urlopen(f"{base}/api/frame/cam_high", timeout=3.0) as r:
                        self.assertEqual(r.status, 204)
                    svc.start_session("op1")
                    svc.submit_command(
                        client_id="op1",
                        command_id="close0",
                        command_type=CommandType.CLOSE_SESSION,
                        lease_epoch=svc.get_snapshot()["lease"]["lease_epoch"],
                    )
                    self.assertEqual(hw.state["robot_constructions"], 0)
                    self.assertEqual(hw.state["image_client_constructions"], 0)
                    self.assertEqual(hw.state["go_home_calls"], 0)
                    self.assertFalse(bridge.hardware_attached)
                finally:
                    panel.stop()
                    bridge.shutdown_hardware(safe_normal_completion=False)
            finally:
                srv.shutdown()
                srv.server_close()

    def test_supervised_single_object_regression_with_warmup_and_safe_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rep = _write_dryrun_report(Path(tmp) / "ok.json")
            audit_json = Path(tmp) / "audit.json"
            hw = _SimulatedStationHardware({"R2": "yellow"})
            srv = _start_mock_right7_server(_MockServerState(hw, checkpoint=_guarded.EXPECTED_CHECKPOINT))
            port = srv.server_address[1]
            try:
                cfg = LiveHardwareConfig(
                    server_url=f"http://127.0.0.1:{port}",
                    dry_run_report_path=rep,
                    operator_admitted=True,
                    confirm_startup_home=True,
                    confirm_exit_home=True,
                    unsafe_exit_action="immediate_disable",
                    robot_init_settle_s=0.0,
                    settle_margin_s=0.0,
                    settle_drift_sleep_s=0.0,
                    chunk_wait_slice_s=0.01,
                    grasp_duration_s=0.05,
                    speed=10.0,
                    takeoff_speed=10.0,
                    aerial_approach_speed=10.0,
                    approach_speed=10.0,
                    transport_speed=10.0,
                    place_speed=10.0,
                    snapshot_dir=Path(tmp) / "snaps",
                )
                bridge = LiveRight7HardwareBridge(
                    config=cfg,
                    robot_cls=hw.make_robot_cls(),
                    image_client_cls=hw.make_image_client_cls(),
                )
                bridge.admit_hardware()
                svc = SessionService(bridge)
                panel = DemoPanelServer(svc, host="127.0.0.1", port=0)
                panel.start()
                base = f"http://127.0.0.1:{panel.port}"
                try:
                    svc.start_session("op1")
                    ep = svc.get_snapshot()["lease"]["lease_epoch"]
                    cmd = svc.submit_command(
                        client_id="op1",
                        command_id="single_reg_1",
                        command_type=CommandType.START_BATCH,
                        raw_text="把黄色木块放到棕色区域",
                        lease_epoch=ep,
                    )
                    self.assertEqual(cmd.status, CommandStatus.APPLIED)

                    deadline = time.monotonic() + 8.0
                    while time.monotonic() < deadline:
                        cur = svc.get_snapshot()
                        if cur["session_state"] == SessionState.READY.value and cur["metrics"]["completed_count"] == 1:
                            break
                        time.sleep(0.05)

                    final = svc.get_snapshot()
                    self.assertEqual(final["metrics"]["completed_count"], 1)
                    self.assertEqual(final["contact_held_state"], ContactHeldState.EMPTY_CONFIRMED.value)
                    self.assertEqual(hw.state["robot_constructions"], 1)
                    self.assertEqual(hw.state["placed_colors"], ["yellow"])
                    self.assertIsNotNone(final["runtime_status"]["warmup_infer_ms"])
                    with request.urlopen(f"{base}/api/frame/cam_high", timeout=3.0) as fr:
                        self.assertEqual(fr.status, 200)
                        self.assertTrue(fr.read().startswith(b"\xff\xd8"))
                    audit = bridge.shutdown_hardware(safe_normal_completion=True, audit_json_path=audit_json)
                    self.assertIsNotNone(audit)
                    self.assertTrue(audit["home_called"])
                    self.assertEqual(hw.state["go_home_calls"], 2)  # Startup DualTouchRobot() + safe normal exit
                    self.assertEqual(hw.state["disconnect_calls"], 1)
                finally:
                    panel.stop()
            finally:
                srv.shutdown()
                srv.server_close()

    def test_supervised_three_object_closed_loop_with_empty_recovery_and_checkpoint_switch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rep = _write_dryrun_report(Path(tmp) / "ok.json")
            hw = _SimulatedStationHardware(
                {"R1": "yellow", "R7": "blue", "R9": "red"},
                fail_grasp_once={"R9"},
                small_blue=True,
            )
            srv = _start_mock_right7_server(_MockServerState(hw, checkpoint=_guarded.EXPECTED_CHECKPOINT))
            port = srv.server_address[1]
            try:
                cfg = LiveHardwareConfig(
                    server_url=f"http://127.0.0.1:{port}",
                    dry_run_report_path=rep,
                    operator_admitted=True,
                    confirm_startup_home=True,
                    confirm_exit_home=True,
                    unsafe_exit_action="immediate_disable",
                    use_clear_obs_retreat=True,
                    robot_init_settle_s=0.0,
                    settle_margin_s=0.0,
                    settle_drift_sleep_s=0.0,
                    chunk_wait_slice_s=0.01,
                    grasp_duration_s=0.05,
                    speed=10.0,
                    takeoff_speed=10.0,
                    aerial_approach_speed=10.0,
                    approach_speed=10.0,
                    transport_speed=10.0,
                    place_speed=10.0,
                )
                bridge = LiveRight7HardwareBridge(
                    config=cfg,
                    robot_cls=hw.make_robot_cls(),
                    image_client_cls=hw.make_image_client_cls(),
                )
                bridge.admit_hardware()
                svc = SessionService(bridge)
                svc.start_session("op1")
                ep = svc.get_snapshot()["lease"]["lease_epoch"]

                # Submit live priority switch as soon as PLACING starts on the first object
                sent_update = {"done": False}
                orig_step = bridge.run_phase_step

                def wrapped_step(**kwargs):
                    if kwargs["phase"].value == "PLACING" and not sent_update["done"]:
                        sent_update["done"] = True
                        rec = svc.submit_command(
                            client_id="op1",
                            command_id="prio_red",
                            command_type=CommandType.UPDATE_PRIORITY,
                            raw_text="剩余任务先处理红色",
                            scope="remaining_objects",
                            lease_epoch=ep,
                            expected_active_generation=1,
                        )
                        self.assertEqual(rec.status, CommandStatus.ACCEPTED_PENDING)
                    return orig_step(**kwargs)

                bridge.run_phase_step = wrapped_step  # type: ignore[method-assign]

                svc.submit_command(
                    client_id="op1",
                    command_id="three_obj_1",
                    command_type=CommandType.START_BATCH,
                    raw_text="把桌上的红、黄、蓝三个木块逐个放到棕色区域，优先抓容易的。",
                    lease_epoch=ep,
                )

                deadline = time.monotonic() + 20.0
                while time.monotonic() < deadline:
                    svc.heartbeat("op1", ep)
                    cur = svc.get_snapshot()
                    if cur["session_state"] == SessionState.READY.value and cur["metrics"]["completed_count"] == 3:
                        break
                    time.sleep(0.05)

                final = svc.get_snapshot()
                self.assertEqual(final["metrics"]["completed_count"], 3)
                self.assertEqual(final["metrics"]["empty_grasp_count"], 1)
                self.assertEqual(final["metrics"]["recovery_success_count"], 1)
                self.assertEqual(final["active_generation"], 2)
                self.assertTrue(sent_update["done"])
                # Yellow placed 1st -> checkpoint update prioritizes red next (recovers from empty grasp) -> blue 3rd!
                self.assertEqual(hw.state["placed_colors"], ["yellow", "red", "blue"])
                # Also verify clear_obs pose remained >= safe Z
                self.assertGreaterEqual(float(CLEAR_OBS_POSE6[2]), 0.175)
                bridge.shutdown_hardware(safe_normal_completion=True)
            finally:
                srv.shutdown()
                srv.server_close()

    def test_stale_observation_rejected_and_unsafe_exit_suppresses_go_home(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rep = _write_dryrun_report(Path(tmp) / "ok.json")
            hw = _SimulatedStationHardware({"R8": "red"})
            fake_time = [100.0]
            stale_armed = [False]

            def on_predict() -> None:
                if stale_armed[0]:
                    fake_time[0] += 0.85

            srv = _start_mock_right7_server(
                _MockServerState(hw, checkpoint=_guarded.EXPECTED_CHECKPOINT, on_predict=on_predict)
            )
            port = srv.server_address[1]
            hold_confirm_calls: list[dict] = []
            try:
                cfg = LiveHardwareConfig(
                    server_url=f"http://127.0.0.1:{port}",
                    dry_run_report_path=rep,
                    operator_admitted=True,
                    confirm_startup_home=True,
                    confirm_exit_home=True,
                    unsafe_exit_action="confirm_before_disable",
                    robot_init_settle_s=0.0,
                    settle_margin_s=0.0,
                    settle_drift_sleep_s=0.0,
                )
                def advance_sleep(dt: float) -> None:
                    fake_time[0] += max(0.0, float(dt))

                bridge = LiveRight7HardwareBridge(
                    config=cfg,
                    robot_cls=hw.make_robot_cls(),
                    image_client_cls=hw.make_image_client_cls(),
                    hold_confirm_fn=lambda audit: hold_confirm_calls.append(dict(audit)),
                    clock_fn=lambda: fake_time[0],
                    sleep_fn=advance_sleep,
                )
                bridge.admit_hardware()
                _obs_id, objs = bridge.observe_scene("scene_1_standard", fake_time[0])
                self.assertTrue(objs)

                # Arm > 800 ms latency spike during /predict -> must abort before sending any motor chunk!
                stale_armed[0] = True
                stale_res = bridge.run_phase_step(
                    obj=objs[0],
                    phase=ExecutionPhase.DESCENDING_CONTACT,
                    task_generation=1,
                    attempt_number=1,
                    now_mono=fake_time[0],
                )
                self.assertFalse(stale_res.ok)
                self.assertEqual(stale_res.error_code, "stale_observation")
                self.assertEqual(hw.state["sent_chunks"], [])

                # Conservative held/uncertain exit MUST suppress go_home() and request hold confirmation.
                bridge._holding_object = True
                audit = bridge.shutdown_hardware(safe_normal_completion=False)
                self.assertIsNotNone(audit)
                self.assertFalse(audit["home_called"])
                self.assertEqual(hw.state["go_home_calls"], 1)  # Startup ONLY; exit go_home SUPPRESSED!
                self.assertEqual(len(hold_confirm_calls), 1)
                self.assertEqual(hw.state["disconnect_calls"], 1)
            finally:
                srv.shutdown()
                srv.server_close()

    def test_d3_sdk_bundle_generation_and_run_panel_cli_supervised_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d3_bundle_path = Path(tmp) / "d3_bundle.json"
            bundle = build_demo_d3_sdk_dryrun_bundle(d3_bundle_path)
            self.assertEqual(len(bundle["cases"]), 7)
            self.assertEqual(
                {c["name"] for c in bundle["cases"]},
                {
                    "d3_post_grasp_lift_R8",
                    "d3_post_grasp_lift_R2",
                    "d3_post_release_lift_brown",
                    "d3_brown_to_R3_hover",
                    "d3_brown_to_R5_hover",
                    "d3_brown_to_R6_hover",
                    "d3_brown_to_clear_obs",
                },
            )
            with self.assertRaises(PermissionError):
                run_panel_main(["--mode", "supervised"])
            # Explicitly test non-TTY refusal when --unsafe-exit-action confirm_before_disable is selected.
            orig_stdin = __import__("sys").stdin
            try:
                class _NonTty:
                    def isatty(self) -> bool:
                        return False

                __import__("sys").stdin = _NonTty()
                with self.assertRaises(SystemExit):
                    run_panel_main(["--mode", "supervised", "--unsafe-exit-action", "confirm_before_disable"])
            finally:
                __import__("sys").stdin = orig_stdin

    def test_slow_hardware_startup_keeps_lease_and_second_batch_resets_slots(self) -> None:
        """Regression for live startup Home taking > lease_ttl_s blocking heartbeats and multi-batch slot reset."""
        with tempfile.TemporaryDirectory() as tmp:
            rep = _write_dryrun_report(Path(tmp) / "ok.json")
            hw = _SimulatedStationHardware({"R2": "yellow"})
            srv = _start_mock_right7_server(_MockServerState(hw, checkpoint=_guarded.EXPECTED_CHECKPOINT))
            port = srv.server_address[1]
            try:
                cfg = LiveHardwareConfig(
                    server_url=f"http://127.0.0.1:{port}",
                    dry_run_report_path=rep,
                    operator_admitted=True,
                    confirm_startup_home=True,
                    confirm_exit_home=True,
                    unsafe_exit_action="immediate_disable",
                    robot_init_settle_s=0.55,  # Deliberately exceeds lease_ttl_s=0.30s
                    settle_margin_s=0.0,
                    settle_drift_sleep_s=0.0,
                    chunk_wait_slice_s=0.01,
                    grasp_duration_s=0.05,
                    speed=10.0,
                    takeoff_speed=10.0,
                    aerial_approach_speed=10.0,
                    approach_speed=10.0,
                    transport_speed=10.0,
                    place_speed=10.0,
                )
                bridge = LiveRight7HardwareBridge(
                    config=cfg,
                    robot_cls=hw.make_robot_cls(),
                    image_client_cls=hw.make_image_client_cls(),
                )
                bridge.admit_hardware()
                svc = SessionService(
                    bridge,
                    budgets=SafetyBudgets(lease_ttl_s=0.30, heartbeat_interval_s=0.08),
                    watchdog_poll_s=0.02,
                )
                svc.start_session("op1")
                ep = svc.get_snapshot()["lease"]["lease_epoch"]

                stop_hb = threading.Event()
                hb_ok_during_home = []

                def hb_loop() -> None:
                    while not stop_hb.is_set():
                        ok, _ = svc.heartbeat("op1", ep)
                        hb_ok_during_home.append(ok)
                        time.sleep(0.06)

                hb_thread = threading.Thread(target=hb_loop, daemon=True)
                hb_thread.start()
                try:
                    cmd = svc.submit_command(
                        client_id="op1",
                        command_id="slow_home_batch1",
                        command_type=CommandType.START_BATCH,
                        raw_text="把黄色木块放到棕色区域",
                        lease_epoch=ep,
                    )
                    self.assertEqual(cmd.status, CommandStatus.APPLIED)

                    deadline = time.monotonic() + 8.0
                    while time.monotonic() < deadline:
                        cur = svc.get_snapshot()
                        if cur["session_state"] == SessionState.READY.value and cur["metrics"]["completed_count"] == 1:
                            break
                        time.sleep(0.05)
                    snap1 = svc.get_snapshot()
                    self.assertEqual(snap1["metrics"]["completed_count"], 1)
                    self.assertFalse(any(ev["kind"] == "CONTROLLER_LEASE_EXPIRED" for ev in snap1["events"]))
                    self.assertEqual(bridge.runtime_status()["placed_slots_used"], 1)

                    # Re-populate yellow block on table and run Batch 2 in same session -> slot resets cleanly to 0!
                    hw.state["grid_blocks"]["R2"] = "yellow"
                    cmd2 = svc.submit_command(
                        client_id="op1",
                        command_id="slow_home_batch2",
                        command_type=CommandType.START_BATCH,
                        raw_text="把黄色木块放到棕色区域",
                        lease_epoch=ep,
                    )
                    self.assertEqual(cmd2.status, CommandStatus.APPLIED)
                    deadline2 = time.monotonic() + 8.0
                    while time.monotonic() < deadline2:
                        cur = svc.get_snapshot()
                        if cur["session_state"] == SessionState.READY.value and cur["metrics"]["completed_count"] == 1:
                            break
                        time.sleep(0.05)
                    self.assertEqual(bridge.runtime_status()["placed_slots_used"], 1)  # Reset to 0 -> incremented to 1!
                    self.assertEqual(hw.state["robot_constructions"], 1)
                finally:
                    stop_hb.set()
                    hb_thread.join(timeout=1.0)
                    bridge.shutdown_hardware(safe_normal_completion=True)
            finally:
                srv.shutdown()
                srv.server_close()

    def test_chunk0_home_takeoff_post_plan_guard_matches_guarded_contract(self) -> None:
        """Chunk 0 allows Home takeoff gap <= 0.25 rad & drift <= 0.02 rad; loop_index > 0 stays strict (0.05/0.01)."""
        q_start = CENTER_Q.copy()
        q_fresh_ok = q_start + np.array([0.015, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        wp0_14 = q_fresh_ok + np.array([0.14, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.assertEqual(
            LiveRight7HardwareBridge._evaluate_post_plan_guard(
                loop_index=0,
                q_start=q_start,
                q_fresh=q_fresh_ok,
                wp0=wp0_14,
                seg0_grip=0.072,
                grip_fresh=0.072,
                obs_age_ms=10.0,
                max_gripper_gap_mm=18.0,
                max_observation_age_ms=800.0,
            ),
            [],
        )
        self.assertIn(
            "post_plan_state_drift",
            LiveRight7HardwareBridge._evaluate_post_plan_guard(
                loop_index=0,
                q_start=q_start,
                q_fresh=q_start + 0.025,
                wp0=q_start + 0.025,
                seg0_grip=0.072,
                grip_fresh=0.072,
                obs_age_ms=10.0,
                max_gripper_gap_mm=18.0,
                max_observation_age_ms=800.0,
            ),
        )
        self.assertIn(
            "first_waypoint_gap",
            LiveRight7HardwareBridge._evaluate_post_plan_guard(
                loop_index=0,
                q_start=q_start,
                q_fresh=q_fresh_ok,
                wp0=q_fresh_ok + 0.30,
                seg0_grip=0.072,
                grip_fresh=0.072,
                obs_age_ms=10.0,
                max_gripper_gap_mm=18.0,
                max_observation_age_ms=800.0,
            ),
        )
        # Subsequent settled chunks keep the strict 0.01 / 0.05 rad post-plan guard.
        strict_reasons = LiveRight7HardwareBridge._evaluate_post_plan_guard(
            loop_index=1,
            q_start=q_start,
            q_fresh=q_start + 0.015,
            wp0=q_start + 0.095,
            seg0_grip=0.072,
            grip_fresh=0.072,
            obs_age_ms=10.0,
            max_gripper_gap_mm=18.0,
            max_observation_age_ms=800.0,
        )
        self.assertIn("post_plan_state_drift", strict_reasons)
        self.assertIn("first_waypoint_gap", strict_reasons)

        # End-to-end Home takeoff pipeline with a +0.14 rad IK jump away from exact Home pose.
        with tempfile.TemporaryDirectory() as tmp:
            rep = _write_dryrun_report(Path(tmp) / "ok.json")
            hw = _SimulatedStationHardware({"R9": "red"})

            class _OffsetHomeTakeoffArm(_FakeRightArm):
                def solve_ik_euler(self, pos, euler, seed):
                    q_sol, ok = super().solve_ik_euler(pos, euler, seed)
                    if np.linalg.norm(np.asarray(pos, dtype=float) - HOME_POSE6[:3]) > 1e-4:
                        q_sol[0] = float(q_sol[0]) + 0.14
                    return q_sol, ok

            _BaseRobot = hw.make_robot_cls()

            class _OffsetRobot(_BaseRobot):
                def __init__(self, **kwargs) -> None:
                    super().__init__(**kwargs)
                    self.right_arm = _OffsetHomeTakeoffArm(hw.state)

            srv = _start_mock_right7_server(_MockServerState(hw, checkpoint=_guarded.EXPECTED_CHECKPOINT))
            port = srv.server_address[1]
            try:
                cfg = LiveHardwareConfig(
                    server_url=f"http://127.0.0.1:{port}",
                    dry_run_report_path=rep,
                    operator_admitted=True,
                    confirm_startup_home=True,
                    confirm_exit_home=True,
                    unsafe_exit_action="immediate_disable",
                    settle_margin_s=0.0,
                    settle_drift_sleep_s=0.0,
                    chunk_wait_slice_s=0.01,
                    grasp_duration_s=0.05,
                    speed=10.0,
                    takeoff_speed=10.0,
                    aerial_approach_speed=10.0,
                    approach_speed=10.0,
                    transport_speed=10.0,
                    place_speed=10.0,
                )
                bridge = LiveRight7HardwareBridge(
                    config=cfg,
                    robot_cls=_OffsetRobot,
                    image_client_cls=hw.make_image_client_cls(),
                )
                bridge.admit_hardware()
                obs_id, objs = bridge.observe_scene("scene_1_standard", time.monotonic())
                self.assertTrue(obs_id)
                self.assertEqual(len(objs), 1)
                ok, err_code, err_detail = bridge._run_approaching_empty(objs[0])
                self.assertTrue(ok, msg=f"Home takeoff unexpectedly rejected: {err_code} {err_detail}")
                self.assertGreaterEqual(len(hw.state["sent_chunks"]), 1)
            finally:
                bridge.shutdown_hardware(safe_normal_completion=True)
                srv.shutdown()
                srv.server_close()

    def test_settled_and_placement_bridge_envelope_defaults_and_approach_offset(self) -> None:
        """Defaults match 070-1 benchmark (40/50 mm), accept 22mm approach offset, and reject 45mm."""
        default_cfg = LiveHardwareConfig()
        self.assertEqual(default_cfg.settled_bridge_max_mm, 40.0)
        self.assertEqual(default_cfg.placement_bridge_max_mm, 50.0)
        for bad_settled in (10.0, 55.0):
            with self.assertRaises(ValueError):
                LiveRight7HardwareBridge(config=LiveHardwareConfig(settled_bridge_max_mm=bad_settled))
        for bad_place in (10.0, 65.0):
            with self.assertRaises(ValueError):
                LiveRight7HardwareBridge(config=LiveHardwareConfig(placement_bridge_max_mm=bad_place))

        with tempfile.TemporaryDirectory() as tmp:
            rep = _write_dryrun_report(Path(tmp) / "ok.json")
            hw = _SimulatedStationHardware({"R9": "red"})
            srv = _start_mock_right7_server(
                _MockServerState(
                    hw,
                    checkpoint=_guarded.EXPECTED_CHECKPOINT,
                    initial_raw_offset_m=np.array([0.018, 0.0, -0.012], dtype=float),  # ~21.6 mm (>15 mm, <=40 mm)
                )
            )
            port = srv.server_address[1]
            try:
                cfg = LiveHardwareConfig(
                    server_url=f"http://127.0.0.1:{port}",
                    dry_run_report_path=rep,
                    operator_admitted=True,
                    confirm_startup_home=True,
                    confirm_exit_home=True,
                    unsafe_exit_action="immediate_disable",
                    settle_margin_s=0.0,
                    settle_drift_sleep_s=0.0,
                    chunk_wait_slice_s=0.01,
                    speed=10.0,
                    takeoff_speed=10.0,
                    aerial_approach_speed=10.0,
                    approach_speed=10.0,
                )
                bridge = LiveRight7HardwareBridge(
                    config=cfg,
                    robot_cls=hw.make_robot_cls(),
                    image_client_cls=hw.make_image_client_cls(),
                )
                bridge.admit_hardware()
                obs_id, objs = bridge.observe_scene("scene_1_standard", time.monotonic())
                self.assertTrue(obs_id)
                res_takeoff = bridge.run_phase_step(
                    obj=objs[0],
                    phase=ExecutionPhase.APPROACHING_EMPTY,
                    task_generation=1,
                    attempt_number=1,
                    now_mono=time.monotonic(),
                )
                self.assertTrue(res_takeoff.ok, msg=f"takeoff failed: {res_takeoff.error_code} {res_takeoff.error_detail}")
                res_descend = bridge.run_phase_step(
                    obj=objs[0],
                    phase=ExecutionPhase.DESCENDING_CONTACT,
                    task_generation=1,
                    attempt_number=1,
                    now_mono=time.monotonic(),
                )
                self.assertTrue(
                    res_descend.ok,
                    msg=f"descending approach with 21.6mm raw[0] offset failed: {res_descend.error_code} {res_descend.error_detail}",
                )
            finally:
                bridge.shutdown_hardware(safe_normal_completion=True)
                srv.shutdown()
                srv.server_close()

        # Verify oversized 45 mm raw[0] gap (>40 mm envelope) still fails closed cleanly.
        with tempfile.TemporaryDirectory() as tmp:
            rep = _write_dryrun_report(Path(tmp) / "ok.json")
            hw2 = _SimulatedStationHardware({"R9": "red"})
            srv2 = _start_mock_right7_server(
                _MockServerState(
                    hw2,
                    checkpoint=_guarded.EXPECTED_CHECKPOINT,
                    initial_raw_offset_m=np.array([0.045, 0.0, 0.0], dtype=float),
                )
            )
            port2 = srv2.server_address[1]
            try:
                cfg2 = LiveHardwareConfig(
                    server_url=f"http://127.0.0.1:{port2}",
                    dry_run_report_path=rep,
                    operator_admitted=True,
                    confirm_startup_home=True,
                    confirm_exit_home=True,
                    unsafe_exit_action="immediate_disable",
                    settle_margin_s=0.0,
                    settle_drift_sleep_s=0.0,
                    chunk_wait_slice_s=0.01,
                    speed=10.0,
                    takeoff_speed=10.0,
                    snapshot_dir=Path(tmp) / "rej_snaps",
                )
                bridge2 = LiveRight7HardwareBridge(
                    config=cfg2,
                    robot_cls=hw2.make_robot_cls(),
                    image_client_cls=hw2.make_image_client_cls(),
                )
                bridge2.admit_hardware()
                _obs2, objs2 = bridge2.observe_scene("scene_1_standard", time.monotonic())
                res_takeoff2 = bridge2.run_phase_step(
                    obj=objs2[0],
                    phase=ExecutionPhase.APPROACHING_EMPTY,
                    task_generation=1,
                    attempt_number=1,
                    now_mono=time.monotonic(),
                )
                self.assertTrue(res_takeoff2.ok)
                res_descend2 = bridge2.run_phase_step(
                    obj=objs2[0],
                    phase=ExecutionPhase.DESCENDING_CONTACT,
                    task_generation=1,
                    attempt_number=1,
                    now_mono=time.monotonic(),
                )
                self.assertFalse(res_descend2.ok)
                self.assertEqual(res_descend2.error_code, "guard_rejected")
                self.assertIn("Target outside approach candidate envelope", res_descend2.error_detail or "")
                self.assertTrue((Path(tmp) / "rej_snaps" / "obj_red_R9_1_approach_0_rejected_raw_actions.npy").is_file())
                self.assertTrue((Path(tmp) / "rej_snaps" / "obj_red_R9_1_approach_0_rejected_state.json").is_file())
            finally:
                bridge2.shutdown_hardware(safe_normal_completion=True)
                srv2.shutdown()
                srv2.server_close()

    def test_wrist_visible_offset_beyond_45mm_does_not_pull_back_to_coarse_grid(self) -> None:
        """Regression for r6 step=2: wrist-aligned EE at >45mm from R9 nominal hover must descend, not retreat."""
        with tempfile.TemporaryDirectory() as tmp:
            rep = _write_dryrun_report(Path(tmp) / "ok.json")
            hw = _SimulatedStationHardware({"R9": "red"})
            _BaseImg = hw.make_image_client_cls()

            class _AlignedOffsetWristImg(_BaseImg):
                def get_right_wrist_frame(self):
                    st = hw.state
                    uv = _grasp_cases.wrist_target_uv_for_pose(
                        ee_pose6=st["pose6"],
                        ee_z_m=float(st["pose6"][2]),
                    )
                    return _render_wrist_cam(block_color="red", target_uv=uv), time.monotonic()

            srv = _start_mock_right7_server(_MockServerState(hw, checkpoint=_guarded.EXPECTED_CHECKPOINT))
            port = srv.server_address[1]
            try:
                cfg = LiveHardwareConfig(
                    server_url=f"http://127.0.0.1:{port}",
                    dry_run_report_path=rep,
                    operator_admitted=True,
                    confirm_startup_home=True,
                    confirm_exit_home=True,
                    unsafe_exit_action="immediate_disable",
                    settle_margin_s=0.0,
                    settle_drift_sleep_s=0.0,
                    chunk_wait_slice_s=0.01,
                    speed=10.0,
                    takeoff_speed=10.0,
                    aerial_approach_speed=10.0,
                    approach_speed=10.0,
                )
                bridge = LiveRight7HardwareBridge(
                    config=cfg,
                    robot_cls=hw.make_robot_cls(),
                    image_client_cls=_AlignedOffsetWristImg,
                )
                bridge.admit_hardware()
                _obs_id, objs = bridge.observe_scene("scene_1_standard", time.monotonic())
                # Place EE at exact r6 step=2 aligned hover pose (46.6mm away from nominal GRID_HOVER_POSES['R9']).
                hw.state["pose6"] = np.array([0.1845, 0.0897, 0.1979, -0.3347, 1.0355, 0.9424], dtype=float)
                dist_from_r9 = float(np.linalg.norm(hw.state["pose6"][:2] - GRID_HOVER_POSES["R9"][:2]))
                self.assertGreater(dist_from_r9, 0.045)

                res = bridge.run_phase_step(
                    obj=objs[0],
                    phase=ExecutionPhase.DESCENDING_CONTACT,
                    task_generation=1,
                    attempt_number=1,
                    now_mono=time.monotonic(),
                )
                self.assertTrue(res.ok, msg=f"descend failed: {res.error_code} {res.error_detail}")
                self.assertLessEqual(float(hw.state["pose6"][2]), 0.168)
                self.assertLess(float(hw.state["pose6"][0]), 0.210)  # Must descend locally, not retreat to R9 x=0.230!
            finally:
                bridge.shutdown_hardware(safe_normal_completion=True)
                srv.shutdown()
                srv.server_close()

    def test_held_block_photobombing_high_cam_clears_via_clear_obs_and_fails_closed_if_still_present(self) -> None:
        """Regression for r7 yellow R2 held-in-air false table hit: move held block to CLEAR_OBS keeping clamp."""
        # Part 1: Held block hovering above R2 photobombs cam_high until arm moves aside to CLEAR_OBS -> CONFIRMED + PLACED!
        with tempfile.TemporaryDirectory() as tmp:
            rep = _write_dryrun_report(Path(tmp) / "ok.json")
            hw = _SimulatedStationHardware({"R2": "yellow"})
            _BaseImg = hw.make_image_client_cls()

            class _PhotobombCamImg(_BaseImg):
                def get_head_frame(self):
                    blocks = dict(hw.state["grid_blocks"])
                    # When holding yellow above R2 (y > 0.08), cam_high still sees the lifted block over R2!
                    if hw.state.get("held_color") == "yellow" and float(hw.state["pose6"][1]) > 0.08:
                        blocks["R2"] = "yellow"
                    return _render_high_cam(blocks), time.monotonic()

            srv = _start_mock_right7_server(_MockServerState(hw, checkpoint=_guarded.EXPECTED_CHECKPOINT))
            port = srv.server_address[1]
            try:
                cfg = LiveHardwareConfig(
                    server_url=f"http://127.0.0.1:{port}",
                    dry_run_report_path=rep,
                    operator_admitted=True,
                    confirm_startup_home=True,
                    confirm_exit_home=True,
                    unsafe_exit_action="immediate_disable",
                    use_clear_obs_retreat=True,
                    settle_margin_s=0.0,
                    settle_drift_sleep_s=0.0,
                    chunk_wait_slice_s=0.01,
                    grasp_duration_s=0.05,
                    speed=10.0,
                    takeoff_speed=10.0,
                    aerial_approach_speed=10.0,
                    approach_speed=10.0,
                    transport_speed=10.0,
                    place_speed=10.0,
                    snapshot_dir=Path(tmp) / "snaps",
                )
                bridge = LiveRight7HardwareBridge(
                    config=cfg,
                    robot_cls=hw.make_robot_cls(),
                    image_client_cls=_PhotobombCamImg,
                )
                bridge.admit_hardware()
                svc = SessionService(bridge)
                svc.start_session("op1")
                ep = svc.get_snapshot()["lease"]["lease_epoch"]
                cmd = svc.submit_command(
                    client_id="op1",
                    command_id="photobomb_yellow",
                    command_type=CommandType.START_BATCH,
                    raw_text="Pick up the yellow block and place it on the brown zone",
                    lease_epoch=ep,
                )
                self.assertEqual(cmd.status, CommandStatus.APPLIED)
                deadline = time.monotonic() + 10.0
                while time.monotonic() < deadline:
                    if svc.get_snapshot()["session_state"] == SessionState.READY.value:
                        break
                    time.sleep(0.05)
                snap = svc.get_snapshot()
                self.assertEqual(snap["metrics"]["completed_count"], 1, msg=str(snap))
                self.assertEqual(hw.state["placed_colors"], ["yellow"])
                self.assertTrue((Path(tmp) / "snaps" / "obj_yellow_R2_1_lift_check_cam_high.jpg").is_file())
                self.assertTrue((Path(tmp) / "snaps" / "obj_yellow_R2_1_lift_check_clear_obs_cam_high.jpg").is_file())
            finally:
                bridge.shutdown_hardware(safe_normal_completion=True)
                srv.shutdown()
                srv.server_close()

        # Part 2: If another yellow block STILL sits on R2 even from CLEAR_OBS -> fail closed to HELD_UNCERTAIN!
        with tempfile.TemporaryDirectory() as tmp:
            rep = _write_dryrun_report(Path(tmp) / "ok.json")
            hw2 = _SimulatedStationHardware({"R2": "yellow"})
            _BaseImg2 = hw2.make_image_client_cls()

            class _PersistentR2Img(_BaseImg2):
                def get_head_frame(self):
                    blocks = dict(hw2.state["grid_blocks"])
                    blocks["R2"] = "yellow"  # Block truly remains on R2 regardless of arm position!
                    return _render_high_cam(blocks), time.monotonic()

            srv2 = _start_mock_right7_server(_MockServerState(hw2, checkpoint=_guarded.EXPECTED_CHECKPOINT))
            port2 = srv2.server_address[1]
            try:
                cfg2 = LiveHardwareConfig(
                    server_url=f"http://127.0.0.1:{port2}",
                    dry_run_report_path=rep,
                    operator_admitted=True,
                    confirm_startup_home=True,
                    confirm_exit_home=True,
                    unsafe_exit_action="immediate_disable",
                    use_clear_obs_retreat=True,
                    settle_margin_s=0.0,
                    settle_drift_sleep_s=0.0,
                    chunk_wait_slice_s=0.01,
                    grasp_duration_s=0.05,
                    speed=10.0,
                    takeoff_speed=10.0,
                    aerial_approach_speed=10.0,
                    approach_speed=10.0,
                    transport_speed=10.0,
                    place_speed=10.0,
                )
                bridge2 = LiveRight7HardwareBridge(
                    config=cfg2,
                    robot_cls=hw2.make_robot_cls(),
                    image_client_cls=_PersistentR2Img,
                )
                bridge2.admit_hardware()
                _obs2, objs2 = bridge2.observe_scene("scene_1_standard", time.monotonic())
                hw2.state["pose6"] = GRID_HOVER_POSES["R2"].copy()
                hw2.state["grip_m"] = 0.031
                hw2.state["held_color"] = "yellow"
                res2 = bridge2._run_lifting_check(objs2[0])
                self.assertTrue(res2.ok)
                self.assertTrue(res2.visual_block_on_table)
                self.assertLessEqual(abs(float(hw2.state["pose6"][0]) - 0.200), 0.01)
                self.assertTrue(0.018 <= hw2.state["grip_m"] <= 0.045)  # Maintained clamped preload, never opened!
            finally:
                bridge2.shutdown_hardware(safe_normal_completion=True)
                srv2.shutdown()
                srv2.server_close()

    def test_placement_verifier_matches_target_color_and_ignores_prior_placed_red_block(self) -> None:
        from v070_demo.evidence_verifier import verify_live_placement_outcome

        # Synthetic brown pad containing ONLY previously placed red block (no yellow block on pad yet):
        hsv = np.zeros((480, 640, 3), dtype=np.uint8)
        hsv[:, :] = (0, 0, 200)
        cv2.rectangle(hsv, (285, 205), (420, 360), (17, 56, 125), -1)  # brown cork pad
        cv2.rectangle(hsv, (315, 248), (372, 305), (4, 230, 225), -1)  # red block at slot 0
        red_only_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        red_only_jpg = _encode_jpeg_bytes(red_only_bgr)

        # Verifying yellow placement with require_target_color=True must fail closed when only red is on pad.
        v_missing_yellow = verify_live_placement_outcome(
            0.0662,
            held_grip_m=0.0292,
            wrist_post_release=red_only_jpg,
            target_color="yellow",
            require_target_color=True,
        )
        self.assertFalse(v_missing_yellow.confirmed)
        self.assertIsNone(v_missing_yellow.evidence.get("block_obb"))

        # Add yellow block alongside red inside the brown pad -> yellow check must match yellow & confirm!
        cv2.rectangle(hsv, (370, 255), (412, 302), (26, 220, 225), -1)
        both_jpg = _encode_jpeg_bytes(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
        v_yellow_ok = verify_live_placement_outcome(
            0.0662,
            held_grip_m=0.0292,
            wrist_post_release=both_jpg,
            target_color="yellow",
            require_target_color=True,
        )
        self.assertTrue(v_yellow_ok.confirmed)
        self.assertEqual(v_yellow_ok.evidence["block_obb"]["color_class"], "yellow")

    def test_slot1_placement_approaches_calibrated_pad_center_and_uses_interior_slot_offset(self) -> None:
        """Regression for r8 slot=1: transport arrives at calibrated slot 0 [0.348,0.216], placing separates inside pad."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rep = _write_dryrun_report(tmp_path / "ok.json")
            hw = _SimulatedStationHardware({"R9": "red", "R2": "yellow"})
            srv = _start_mock_right7_server(_MockServerState(hw, checkpoint=_guarded.EXPECTED_CHECKPOINT))
            port = srv.server_address[1]
            try:
                cfg = LiveHardwareConfig(
                    server_url=f"http://127.0.0.1:{port}",
                    dry_run_report_path=rep,
                    operator_admitted=True,
                    confirm_startup_home=True,
                    confirm_exit_home=True,
                    unsafe_exit_action="immediate_disable",
                    use_clear_obs_retreat=True,
                    snapshot_dir=tmp_path / "snaps",
                    settle_margin_s=0.0,
                    settle_drift_sleep_s=0.0,
                    chunk_wait_slice_s=0.01,
                    grasp_duration_s=0.05,
                    speed=10.0,
                    takeoff_speed=10.0,
                    aerial_approach_speed=10.0,
                    approach_speed=10.0,
                    transport_speed=10.0,
                    place_speed=10.0,
                )
                bridge = LiveRight7HardwareBridge(
                    config=cfg,
                    robot_cls=hw.make_robot_cls(),
                    image_client_cls=hw.make_image_client_cls(),
                )
                bridge.admit_hardware()
                from v070_demo.event_logger import DemoRunLogger

                svc = SessionService(bridge, logger=DemoRunLogger(tmp_path / "run"))
                svc.start_session("op1", scene_id="scene_1_standard")
                ep = svc.get_snapshot()["lease"]["lease_epoch"]
                rec = svc.submit_command(
                    client_id="op1",
                    command_id="slot1_reg_batch",
                    command_type=CommandType.START_BATCH,
                    raw_text="Pick up the red and yellow wooden blocks and place them on the brown target area.",
                    lease_epoch=ep,
                )
                self.assertEqual(rec.status, CommandStatus.APPLIED)
                deadline = time.monotonic() + 20.0
                while time.monotonic() < deadline:
                    cur = svc.get_snapshot()
                    if (
                        cur["session_state"] == SessionState.READY.value
                        and cur["metrics"]["completed_count"] == 2
                        and svc._worker is not None
                        and not svc._worker.is_alive()
                    ):
                        break
                    time.sleep(0.05)
                snap = svc.get_snapshot(client_id="op1")
                self.assertEqual(snap["metrics"]["headline_score"], "2/2")
                slots = {o["color"]: o["destination_slot"] for o in snap["objects"]}
                self.assertEqual(slots["red"], 0)
                self.assertEqual(slots["yellow"], 1)

                # Verify transporting always arrived near [0.348, 0.216] (never pulled to [0.323, 0.216])
                transport_arrivals = [
                    c["end_pose6"]
                    for c in hw.state["sent_chunks"]
                    if abs(float(c["end_pose6"][0]) - 0.348) < 0.004
                    and abs(float(c["end_pose6"][1]) - 0.216) < 0.004
                    and c["end_grip"] < 0.045
                ]
                self.assertGreaterEqual(len(transport_arrivals), 2)
                off_pad_arrivals = [
                    c["end_pose6"]
                    for c in hw.state["sent_chunks"]
                    if float(c["end_pose6"][0]) < 0.335 and float(c["end_pose6"][1]) >= 0.200
                ]
                self.assertEqual(off_pad_arrivals, [])

                # Verify both release chunks opened inside calibrated interior bounds and kept >=24mm separation
                release_poses = [
                    c["end_pose6"]
                    for c in hw.state["sent_chunks"]
                    if c["end_pose6"][0] >= 0.345 and c["end_pose6"][2] < 0.175 and c["end_grip"] >= 0.055
                ]
                self.assertEqual(len(release_poses), 2, release_poses)
                p0, p1 = release_poses[0][:2], release_poses[1][:2]
                self.assertGreaterEqual(float(np.linalg.norm(p1 - p0)), 0.022)
                for xy in (p0, p1):
                    self.assertTrue(0.348 <= float(xy[0]) <= 0.380, xy)
                    self.assertTrue(0.224 <= float(xy[1]) <= 0.256, xy)

                # Verify release_check snapshots and scene_final_manifest.json were written
                self.assertTrue((tmp_path / "snaps" / "obj_red_R9_1_release_check_cam_right_wrist.jpg").is_file())
                self.assertTrue((tmp_path / "snaps" / "obj_yellow_R2_1_release_check_cam_right_wrist.jpg").is_file())
                self.assertTrue((tmp_path / "run" / "scene_final_manifest.json").is_file())
            finally:
                bridge.shutdown_hardware(safe_normal_completion=True)
                srv.shutdown()
                srv.server_close()

    def test_safe_return_home_on_completion_runs_only_when_empty_and_safe(self) -> None:
        """Completed normal batches with return_home_on_batch_complete=True return cleanly to Home before READY."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rep = _write_dryrun_report(tmp_path / "ok.json")
            hw = _SimulatedStationHardware({"R9": "red"})
            srv = _start_mock_right7_server(_MockServerState(hw, checkpoint=_guarded.EXPECTED_CHECKPOINT))
            port = srv.server_address[1]
            try:
                cfg = LiveHardwareConfig(
                    server_url=f"http://127.0.0.1:{port}",
                    dry_run_report_path=rep,
                    operator_admitted=True,
                    confirm_startup_home=True,
                    confirm_exit_home=True,
                    return_home_on_batch_complete=True,
                    unsafe_exit_action="immediate_disable",
                    use_clear_obs_retreat=True,
                    robot_init_settle_s=0.0,
                    settle_margin_s=0.0,
                    settle_drift_sleep_s=0.0,
                    chunk_wait_slice_s=0.01,
                    grasp_duration_s=0.05,
                    speed=10.0,
                    takeoff_speed=10.0,
                    aerial_approach_speed=10.0,
                    approach_speed=10.0,
                    transport_speed=10.0,
                    place_speed=10.0,
                )
                bridge = LiveRight7HardwareBridge(
                    config=cfg,
                    robot_cls=hw.make_robot_cls(),
                    image_client_cls=hw.make_image_client_cls(),
                    sleep_fn=lambda _s: None,
                )
                bridge.admit_hardware()
                svc = SessionService(bridge)
                svc.start_session("op1", scene_id="scene_1_standard")
                ep = svc.get_snapshot()["lease"]["lease_epoch"]
                rec = svc.submit_command(
                    client_id="op1",
                    command_id="home_batch_1",
                    command_type=CommandType.START_BATCH,
                    raw_text="Pick up the red wooden block and place it on the brown target area.",
                    lease_epoch=ep,
                )
                self.assertEqual(rec.status, CommandStatus.APPLIED)
                deadline = time.monotonic() + 12.0
                while time.monotonic() < deadline:
                    cur = svc.get_snapshot()
                    if (
                        cur["session_state"] == SessionState.READY.value
                        and cur["metrics"]["completed_count"] == 1
                        and svc._worker is not None
                        and not svc._worker.is_alive()
                    ):
                        break
                    time.sleep(0.03)
                snap = svc.get_snapshot()
                self.assertEqual(snap["session_state"], SessionState.READY.value)
                self.assertEqual(snap["metrics"]["headline_score"], "1/1")
                self.assertEqual(hw.state["go_home_calls"], 2)  # 1 startup + 1 auto-return Home on completion!
                self.assertLess(float(np.linalg.norm(hw.state["pose6"][:3] - HOME_POSE6[:3])), 0.005)
                self.assertTrue(any(ev["kind"] == "RETURNED_HOME_ON_COMPLETION" for ev in snap["events"]))
            finally:
                bridge.shutdown_hardware(safe_normal_completion=True)
                srv.shutdown()
                srv.server_close()

        # Unsafe/held abort MUST suppress return_home_on_completion even when enabled!
        with tempfile.TemporaryDirectory() as tmp:
            rep = _write_dryrun_report(Path(tmp) / "ok.json")
            hw2 = _SimulatedStationHardware({"R9": "red"})
            srv2 = _start_mock_right7_server(_MockServerState(hw2, checkpoint=_guarded.EXPECTED_CHECKPOINT))
            port2 = srv2.server_address[1]
            try:
                cfg2 = LiveHardwareConfig(
                    server_url=f"http://127.0.0.1:{port2}",
                    dry_run_report_path=rep,
                    operator_admitted=True,
                    confirm_startup_home=True,
                    confirm_exit_home=True,
                    return_home_on_batch_complete=True,
                    unsafe_exit_action="immediate_disable",
                    robot_init_settle_s=0.0,
                )
                bridge2 = LiveRight7HardwareBridge(
                    config=cfg2,
                    robot_cls=hw2.make_robot_cls(),
                    image_client_cls=hw2.make_image_client_cls(),
                    sleep_fn=lambda _s: None,
                )
                bridge2.admit_hardware()
                bridge2._ensure_hardware_attached()
                bridge2._holding_object = True
                ok, err_c, _ = bridge2.return_home_on_completion()
                self.assertFalse(ok)
                self.assertEqual(err_c, "unsafe_home_suppressed")
                self.assertEqual(hw2.state["go_home_calls"], 1)  # Startup only!
            finally:
                bridge2.shutdown_hardware(safe_normal_completion=False)
                srv2.shutdown()
                srv2.server_close()

    def test_clear_table_preset_reenqueues_returned_block_recycles_slots_and_homes_when_empty(self) -> None:
        """Clear the table loops continuously as returned blocks reappear, recycles slots safely, and returns Home."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rep = _write_dryrun_report(tmp_path / "ok.json")
            hw = _SimulatedStationHardware({"R9": "red", "R2": "yellow"})
            replenished_once = [False]

            def on_predict() -> None:
                # Simulate operator removing red from brown pad and putting it back on R9 while yellow is held!
                if (
                    not replenished_once[0]
                    and hw.state["placed_colors"] == ["red"]
                    and hw.state.get("held_color") == "yellow"
                ):
                    replenished_once[0] = True
                    hw.state["grid_blocks"]["R9"] = "red"

            srv = _start_mock_right7_server(
                _MockServerState(hw, checkpoint=_guarded.EXPECTED_CHECKPOINT, on_predict=on_predict)
            )
            port = srv.server_address[1]
            try:
                cfg = LiveHardwareConfig(
                    server_url=f"http://127.0.0.1:{port}",
                    dry_run_report_path=rep,
                    operator_admitted=True,
                    confirm_startup_home=True,
                    confirm_exit_home=True,
                    return_home_on_batch_complete=True,
                    unsafe_exit_action="immediate_disable",
                    use_clear_obs_retreat=True,
                    robot_init_settle_s=0.0,
                    settle_margin_s=0.0,
                    settle_drift_sleep_s=0.0,
                    chunk_wait_slice_s=0.01,
                    grasp_duration_s=0.05,
                    speed=10.0,
                    takeoff_speed=10.0,
                    aerial_approach_speed=10.0,
                    approach_speed=10.0,
                    transport_speed=10.0,
                    place_speed=10.0,
                )
                bridge = LiveRight7HardwareBridge(
                    config=cfg,
                    robot_cls=hw.make_robot_cls(),
                    image_client_cls=hw.make_image_client_cls(),
                    sleep_fn=lambda _s: None,
                )
                bridge.admit_hardware()
                from v070_demo.event_logger import DemoRunLogger

                svc = SessionService(bridge, logger=DemoRunLogger(tmp_path / "run_clear"))
                svc.start_session("op1", scene_id="scene_clear_table")
                ep = svc.get_snapshot()["lease"]["lease_epoch"]
                rec = svc.submit_command(
                    client_id="op1",
                    command_id="clear_batch_1",
                    command_type=CommandType.START_BATCH,
                    raw_text="Clear the table: continuously pick up any wooden blocks (red, yellow, or blue) from the white grid and place them on the brown target area until empty.",
                    scene_id="scene_clear_table",
                    lease_epoch=ep,
                )
                self.assertEqual(rec.status, CommandStatus.APPLIED)
                deadline = time.monotonic() + 25.0
                while time.monotonic() < deadline:
                    svc.heartbeat("op1", ep)
                    cur = svc.get_snapshot()
                    if (
                        cur["session_state"] == SessionState.READY.value
                        and cur["metrics"]["completed_count"] == 3
                        and svc._worker is not None
                        and not svc._worker.is_alive()
                    ):
                        break
                    time.sleep(0.03)
                snap = svc.get_snapshot()
                self.assertEqual(snap["session_state"], SessionState.READY.value)
                self.assertEqual(snap["metrics"]["headline_score"], "3/3")
                self.assertEqual(hw.state["placed_colors"], ["red", "yellow", "red"])
                self.assertTrue(any(ev["kind"] == "TABLE_REENQUEUED" for ev in snap["events"]))
                self.assertTrue(any(ev["kind"] == "RETURNED_HOME_ON_COMPLETION" for ev in snap["events"]))
                self.assertEqual(hw.state["go_home_calls"], 2)  # 1 startup + 1 clear-complete return Home!
                # Verify returned red got deduplicated cycle ID and reused freed slot 0!
                objs_by_id = {o["object_id"]: o for o in snap["objects"]}
                self.assertIn("obj_red_R9_1_c2", objs_by_id)
                self.assertEqual(objs_by_id["obj_red_R9_1_c2"]["destination_slot"], 0)
            finally:
                bridge.shutdown_hardware(safe_normal_completion=True)
                srv.shutdown()
                srv.server_close()

    def test_22_lift_preload_and_mid_transport_slip_auto_recovery_in_clear_table(self) -> None:
        """Verify 12.5mm active lift/transport preload, compliant settle tolerance, and mid-air slip recovery."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rep = _write_dryrun_report(tmp_path / "ok.json")
            hw = _SimulatedStationHardware({"R9": "red"})
            dropped_once = [False]

            def on_predict() -> None:
                # On the first placement attempt over the brown zone, simulate the r10 mid-transport slip
                # where the block fell back onto R9 and the empty gripper closed to 0.0219m (< held - 4mm).
                pos = hw.state["pose6"][:3]
                if not dropped_once[0] and pos[0] >= 0.295 and pos[1] >= 0.175 and hw.state.get("held_color") == "red":
                    dropped_once[0] = True
                    hw.state["held_color"] = None
                    hw.state["grid_blocks"]["R9"] = "red"
                    hw.state["grip_m"] = 0.0219

            srv = _start_mock_right7_server(
                _MockServerState(hw, checkpoint=_guarded.EXPECTED_CHECKPOINT, on_predict=on_predict)
            )
            port = srv.server_address[1]
            try:
                cfg = LiveHardwareConfig(
                    server_url=f"http://127.0.0.1:{port}",
                    dry_run_report_path=rep,
                    operator_admitted=True,
                    confirm_startup_home=True,
                    confirm_exit_home=True,
                    return_home_on_batch_complete=True,
                    unsafe_exit_action="immediate_disable",
                    use_clear_obs_retreat=True,
                    robot_init_settle_s=0.0,
                    settle_margin_s=0.0,
                    settle_drift_sleep_s=0.0,
                    chunk_wait_slice_s=0.01,
                    grasp_duration_s=0.05,
                    speed=10.0,
                    takeoff_speed=10.0,
                    aerial_approach_speed=10.0,
                    approach_speed=10.0,
                    transport_speed=10.0,
                    place_speed=10.0,
                )
                self.assertAlmostEqual(cfg.clamp_preload_m, 0.0125, places=4)
                bridge = LiveRight7HardwareBridge(
                    config=cfg,
                    robot_cls=hw.make_robot_cls(),
                    image_client_cls=hw.make_image_client_cls(),
                    sleep_fn=lambda _s: None,
                )
                bridge.admit_hardware()

                # Unit-check compliant settle vs confirmed slip on _handle_potential_slippage:
                from v070_demo.contracts import ObjectRecord

                dummy_obj = ObjectRecord(
                    object_id="obj_red_R9_1",
                    category="woodblock",
                    color="red",
                    pose_estimate={"x": 0.23, "y": 0.10, "z": 0.03, "yaw_deg": 0.0},
                    confidence=0.98,
                    observation_id="obs_test",
                    observation_ts_mono=1.0,
                    grid_cell="R9",
                )
                # 1) Compliant pad settle: held=0.0302 -> measured=0.0252 (>=0.024 and >=clamp+4.5mm) with block at (324, 315)
                held_wrist = _render_wrist_cam(block_color="red", show_brown_target= False, target_uv=(324.0, 315.0))
                ok_settle, new_held, err_c, _ = bridge._handle_potential_slippage(
                    dummy_obj,
                    pose7=np.array([0.348, 0.216, 0.193, -0.47, 0.75, 0.78, 0.0252], dtype=float),
                    held=0.0302,
                    wrist_bytes=held_wrist,
                )
                self.assertTrue(ok_settle)
                self.assertIsNone(err_c)
                self.assertAlmostEqual(new_held, 0.0252, places=4)

                # 2) Full end-to-end clear-table recovery when mid-air slip (0.0219m) occurs on attempt 1:
                from v070_demo.event_logger import DemoRunLogger

                svc = SessionService(bridge, logger=DemoRunLogger(tmp_path / "run_slip_recover"))
                svc.start_session("op1", scene_id="scene_clear_table")
                ep = svc.get_snapshot()["lease"]["lease_epoch"]
                rec = svc.submit_command(
                    client_id="op1",
                    command_id="clear_slip_1",
                    command_type=CommandType.START_BATCH,
                    raw_text="Clear the table: continuously pick up any wooden blocks (red, yellow, or blue) from the white grid and place them on the brown target area until empty.",
                    scene_id="scene_clear_table",
                    lease_epoch=ep,
                )
                self.assertEqual(rec.status, CommandStatus.APPLIED)
                deadline = time.monotonic() + 25.0
                while time.monotonic() < deadline:
                    svc.heartbeat("op1", ep)
                    cur = svc.get_snapshot()
                    if (
                        cur["session_state"] == SessionState.READY.value
                        and cur["metrics"]["completed_count"] == 1
                        and svc._worker is not None
                        and not svc._worker.is_alive()
                    ):
                        break
                    time.sleep(0.03)
                snap = svc.get_snapshot()
                self.assertTrue(dropped_once[0])
                self.assertEqual(snap["session_state"], SessionState.READY.value)
                self.assertEqual(snap["metrics"]["headline_score"], "1/1")
                self.assertEqual(snap["metrics"]["empty_grasp_count"], 1)
                self.assertEqual(snap["metrics"]["recovery_success_count"], 1)
                self.assertEqual(hw.state["placed_colors"], ["red"])
                self.assertTrue(any(ev["kind"] == "EMPTY_GRASP_RECOVERED_FOR_RETRY" for ev in snap["events"]))
                self.assertTrue(any(ev["kind"] == "RETURNED_HOME_ON_COMPLETION" for ev in snap["events"]))
            finally:
                bridge.shutdown_hardware(safe_normal_completion=True)
                srv.shutdown()
                srv.server_close()

    def test_23_aligned_yellow_hover_descends_and_grasps_even_when_policy_omits_closure(self) -> None:
        """Regression for r11 R1 yellow: aligned hover descends to 0.162m and grasps even without policy <0.04."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rep = _write_dryrun_report(tmp_path / "ok.json")
            hw = _SimulatedStationHardware({"R1": "yellow"})
            srv = _start_mock_right7_server(
                _MockServerState(hw, checkpoint=_guarded.EXPECTED_CHECKPOINT, omit_closure_for_grasp=True)
            )
            port = srv.server_address[1]
            try:
                cfg = LiveHardwareConfig(
                    server_url=f"http://127.0.0.1:{port}",
                    dry_run_report_path=rep,
                    operator_admitted=True,
                    confirm_startup_home=True,
                    confirm_exit_home=True,
                    return_home_on_batch_complete=True,
                    unsafe_exit_action="immediate_disable",
                    use_clear_obs_retreat=True,
                    robot_init_settle_s=0.0,
                    settle_margin_s=0.0,
                    settle_drift_sleep_s=0.0,
                    chunk_wait_slice_s=0.01,
                    grasp_duration_s=0.05,
                    speed=10.0,
                    takeoff_speed=10.0,
                    aerial_approach_speed=10.0,
                    approach_speed=10.0,
                    transport_speed=10.0,
                    place_speed=10.0,
                )
                bridge = LiveRight7HardwareBridge(
                    config=cfg,
                    robot_cls=hw.make_robot_cls(),
                    image_client_cls=hw.make_image_client_cls(),
                    sleep_fn=lambda _s: None,
                )
                bridge.admit_hardware()
                from v070_demo.event_logger import DemoRunLogger

                svc = SessionService(bridge, logger=DemoRunLogger(tmp_path / "run_yellow_r1"))
                svc.start_session("op1", scene_id="scene_clear_table")
                ep = svc.get_snapshot()["lease"]["lease_epoch"]
                rec = svc.submit_command(
                    client_id="op1",
                    command_id="clear_yellow_1",
                    command_type=CommandType.START_BATCH,
                    raw_text="Clear the table: continuously pick up any wooden blocks (red, yellow, or blue) from the white grid and place them on the brown target area until empty.",
                    scene_id="scene_clear_table",
                    lease_epoch=ep,
                )
                self.assertEqual(rec.status, CommandStatus.APPLIED)
                deadline = time.monotonic() + 25.0
                while time.monotonic() < deadline:
                    svc.heartbeat("op1", ep)
                    cur = svc.get_snapshot()
                    if (
                        cur["session_state"] == SessionState.READY.value
                        and cur["metrics"]["completed_count"] == 1
                        and svc._worker is not None
                        and not svc._worker.is_alive()
                    ):
                        break
                    time.sleep(0.03)
                snap = svc.get_snapshot()
                self.assertEqual(snap["session_state"], SessionState.READY.value)
                self.assertEqual(snap["metrics"]["headline_score"], "1/1")
                self.assertEqual(hw.state["placed_colors"], ["yellow"])
                self.assertTrue(any(ev["kind"] == "RETURNED_HOME_ON_COMPLETION" for ev in snap["events"]))
            finally:
                bridge.shutdown_hardware(safe_normal_completion=True)
                srv.shutdown()
                srv.server_close()

    def test_24_release_check_survives_hand_merged_brown_contour_and_transient_occlusion(self) -> None:
        """Verify fallback_target_obb + transient occlusion retry confirm valid placement safely."""
        from v070_demo.evidence_verifier import verify_live_placement_outcome
        import cv2

        # 1) Skin-merged cork pad frame: blue block is clean inside pad box, but no brown OBB >2000..45000px.
        no_brown_blue = _render_wrist_cam(block_color="blue", show_brown_target=False, target_uv=(329.0, 258.0))
        cached_pad = {"cx": 350.0, "cy": 280.0, "w": 135.0, "h": 155.0, "angle": 0.0, "area": 20000.0}
        # Without fallback_target_obb, fails closed; with place_0 cached target_obb, confirms cleanly!
        res_no_fb = verify_live_placement_outcome(
            0.068, held_grip_m=0.027, wrist_post_release=no_brown_blue, target_color="blue", require_target_color=True
        )
        self.assertFalse(res_no_fb.confirmed)
        res_fb = verify_live_placement_outcome(
            0.068,
            held_grip_m=0.027,
            wrist_post_release=no_brown_blue,
            target_color="blue",
            require_target_color=True,
            fallback_target_obb=cached_pad,
        )
        self.assertTrue(res_fb.confirmed)
        self.assertTrue(res_fb.evidence["used_fallback_target_obb"])

        # 2) Transient full-hand occlusion on release_check frame 0 clears on retry 1:
        hsv_blank = np.zeros((480, 640, 3), dtype=np.uint8)
        hsv_blank[:, :] = (0, 0, 200)
        occluded_bytes = _encode_jpeg_bytes(cv2.cvtColor(hsv_blank, cv2.COLOR_HSV2BGR))
        clear_bytes = _render_wrist_cam(block_color="red", show_brown_target=True)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rep = _write_dryrun_report(tmp_path / "ok.json")
            hw = _SimulatedStationHardware({})
            hw.state["pose6"] = np.array([0.364, 0.254, 0.174, -0.56, 0.88, 0.72], dtype=float)
            hw.state["grip_m"] = 0.0681
            srv = _start_mock_right7_server(_MockServerState(hw, checkpoint=_guarded.EXPECTED_CHECKPOINT))
            port = srv.server_address[1]
            try:
                cfg = LiveHardwareConfig(
                    server_url=f"http://127.0.0.1:{port}",
                    dry_run_report_path=rep,
                    operator_admitted=True,
                    confirm_startup_home=True,
                    confirm_exit_home=True,
                    unsafe_exit_action="immediate_disable",
                    robot_init_settle_s=0.0,
                    settle_margin_s=0.0,
                    settle_drift_sleep_s=0.0,
                    chunk_wait_slice_s=0.01,
                )
                bridge = LiveRight7HardwareBridge(
                    config=cfg,
                    robot_cls=hw.make_robot_cls(),
                    image_client_cls=hw.make_image_client_cls(),
                    sleep_fn=lambda _s: None,
                )
                bridge.admit_hardware()
                bridge._ensure_hardware_attached()
                calls = [0]

                def fake_read_live_obs(*, task: str):
                    calls[0] += 1
                    wb = occluded_bytes if calls[0] == 1 else clear_bytes
                    return {"observation": True}, {"wrist_bytes": wb, "high_bytes": clear_bytes, "first_camera_mono": 1.0}

                bridge._read_live_observation = fake_read_live_obs  # type: ignore[method-assign]
                from v070_demo.contracts import ObjectRecord

                obj = ObjectRecord(
                    object_id="obj_red_R5_1",
                    category="woodblock",
                    color="red",
                    pose_estimate={"x": 0.25, "y": 0.145, "z": 0.03, "yaw_deg": 0.0},
                    confidence=0.95,
                    observation_id="obs1",
                    observation_ts_mono=1.0,
                    grid_cell="R5",
                    destination_slot=2,
                )
                res = bridge._run_releasing_check(obj)
                self.assertTrue(res.ok)
                self.assertTrue(res.visual_block_in_brown_zone)
                self.assertGreaterEqual(calls[0], 2)
            finally:
                bridge.shutdown_hardware(safe_normal_completion=True)
                srv.shutdown()
                srv.server_close()

    def test_25_scene1_strict_order_red_yellow_blue_ignores_ease_score(self) -> None:
        """Scene 1 default preset strictly picks Red -> Yellow -> Blue (r12 R3/R1/R8 layout)."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rep = _write_dryrun_report(tmp_path / "ok.json")
            hw = _SimulatedStationHardware({"R3": "red", "R1": "yellow", "R8": "blue"})
            srv = _start_mock_right7_server(_MockServerState(hw, checkpoint=_guarded.EXPECTED_CHECKPOINT))
            port = srv.server_address[1]
            try:
                cfg = LiveHardwareConfig(
                    server_url=f"http://127.0.0.1:{port}",
                    dry_run_report_path=rep,
                    operator_admitted=True,
                    confirm_startup_home=True,
                    confirm_exit_home=True,
                    return_home_on_batch_complete=True,
                    unsafe_exit_action="immediate_disable",
                    use_clear_obs_retreat=True,
                    robot_init_settle_s=0.0,
                    settle_margin_s=0.0,
                    settle_drift_sleep_s=0.0,
                    chunk_wait_slice_s=0.01,
                    grasp_duration_s=0.05,
                    speed=10.0,
                    takeoff_speed=10.0,
                    aerial_approach_speed=10.0,
                    approach_speed=10.0,
                    transport_speed=10.0,
                    place_speed=10.0,
                )
                bridge = LiveRight7HardwareBridge(
                    config=cfg,
                    robot_cls=hw.make_robot_cls(),
                    image_client_cls=hw.make_image_client_cls(),
                    sleep_fn=lambda _s: None,
                )
                bridge.admit_hardware()
                from v070_demo.event_logger import DemoRunLogger

                svc = SessionService(bridge, logger=DemoRunLogger(tmp_path / "run_scene1_ryb"))
                svc.start_session("op1", scene_id="scene_1_standard")
                ep = svc.get_snapshot()["lease"]["lease_epoch"]
                rec = svc.submit_command(
                    client_id="op1",
                    command_id="scene1_ryb_1",
                    command_type=CommandType.START_BATCH,
                    raw_text=(
                        "Pick up the three wooden blocks (red, yellow, and blue) in exact order and place them "
                        "on the brown target area. Re-observe and retry on empty grasp (max 2 attempts per block)."
                    ),
                    scene_id="scene_1_standard",
                    lease_epoch=ep,
                )
                self.assertEqual(rec.status, CommandStatus.APPLIED)
                deadline = time.monotonic() + 25.0
                while time.monotonic() < deadline:
                    svc.heartbeat("op1", ep)
                    cur = svc.get_snapshot()
                    if (
                        cur["session_state"] == SessionState.READY.value
                        and cur["metrics"]["completed_count"] == 3
                        and svc._worker is not None
                        and not svc._worker.is_alive()
                    ):
                        break
                    time.sleep(0.03)
                snap = svc.get_snapshot()
                self.assertEqual(snap["session_state"], SessionState.READY.value)
                self.assertEqual(snap["metrics"]["headline_score"], "3/3")
                self.assertEqual(hw.state["placed_colors"], ["red", "yellow", "blue"])
                self.assertTrue(any(ev["kind"] == "RETURNED_HOME_ON_COMPLETION" for ev in snap["events"]))
            finally:
                bridge.shutdown_hardware(safe_normal_completion=True)
                srv.shutdown()
                srv.server_close()

    def test_26_pregrasp_priority_preempts_before_chunk3_grasping(self) -> None:
        """Clicking priority before Chunk 3 GRASPING preempts safely; at GRASPING it queues after placement."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rep = _write_dryrun_report(tmp_path / "ok.json")
            hw = _SimulatedStationHardware({"R3": "red", "R1": "yellow", "R8": "blue"})
            srv = _start_mock_right7_server(_MockServerState(hw, checkpoint=_guarded.EXPECTED_CHECKPOINT))
            port = srv.server_address[1]
            try:
                cfg = LiveHardwareConfig(
                    server_url=f"http://127.0.0.1:{port}",
                    dry_run_report_path=rep,
                    operator_admitted=True,
                    confirm_startup_home=True,
                    confirm_exit_home=True,
                    return_home_on_batch_complete=True,
                    unsafe_exit_action="immediate_disable",
                    use_clear_obs_retreat=True,
                    robot_init_settle_s=0.0,
                    settle_margin_s=0.0,
                    settle_drift_sleep_s=0.0,
                    chunk_wait_slice_s=0.01,
                    grasp_duration_s=0.05,
                    speed=10.0,
                    takeoff_speed=10.0,
                    aerial_approach_speed=10.0,
                    approach_speed=10.0,
                    transport_speed=10.0,
                    place_speed=10.0,
                )
                bridge = LiveRight7HardwareBridge(
                    config=cfg,
                    robot_cls=hw.make_robot_cls(),
                    image_client_cls=hw.make_image_client_cls(),
                    sleep_fn=lambda _s: None,
                )
                bridge.admit_hardware()
                from v070_demo.contracts import ExecutionPhase
                from v070_demo.event_logger import DemoRunLogger

                svc = SessionService(bridge, logger=DemoRunLogger(tmp_path / "run_pregrasp_preempt"))
                svc.start_session("op1", scene_id="scene_1_standard")
                ep = svc.get_snapshot()["lease"]["lease_epoch"]

                observed_windows: dict[str, str] = {}
                orig_run_phase_step = bridge.run_phase_step
                fired = {"red_descend": False, "blue_grasp": False}

                def hooked_run_phase_step(*, obj, phase, task_generation, attempt_number, now_mono):
                    res = orig_run_phase_step(
                        obj=obj,
                        phase=phase,
                        task_generation=task_generation,
                        attempt_number=attempt_number,
                        now_mono=now_mono,
                    )
                    if obj.color == "red" and phase == ExecutionPhase.DESCENDING_CONTACT and not fired["red_descend"]:
                        fired["red_descend"] = True
                        observed_windows["red_descend"] = svc.get_snapshot()["switch_window"]
                        pri_rec = svc.submit_command(
                            client_id="op1",
                            command_id="pri_blue_preempt",
                            command_type=CommandType.UPDATE_PRIORITY,
                            raw_text="Prioritize blue for the remaining objects",
                            scope="remaining_objects",
                            lease_epoch=ep,
                            expected_active_generation=task_generation,
                        )
                        self.assertEqual(pri_rec.status, CommandStatus.ACCEPTED_PENDING)
                    elif obj.color == "blue" and phase == ExecutionPhase.GRASPING and not fired["blue_grasp"]:
                        fired["blue_grasp"] = True
                        observed_windows["blue_grasp"] = svc.get_snapshot()["switch_window"]
                        pri_rec = svc.submit_command(
                            client_id="op1",
                            command_id="pri_yellow_queued",
                            command_type=CommandType.UPDATE_PRIORITY,
                            raw_text="Prioritize yellow for the remaining objects",
                            scope="remaining_objects",
                            lease_epoch=ep,
                            expected_active_generation=task_generation,
                        )
                        self.assertEqual(pri_rec.status, CommandStatus.ACCEPTED_PENDING)
                    return res

                bridge.run_phase_step = hooked_run_phase_step  # type: ignore[method-assign]
                rec = svc.submit_command(
                    client_id="op1",
                    command_id="scene1_preempt_1",
                    command_type=CommandType.START_BATCH,
                    raw_text=(
                        "Pick up the three wooden blocks (red, yellow, and blue) in exact order and place them "
                        "on the brown target area. Re-observe and retry on empty grasp (max 2 attempts per block)."
                    ),
                    scene_id="scene_1_standard",
                    lease_epoch=ep,
                )
                self.assertEqual(rec.status, CommandStatus.APPLIED)
                deadline = time.monotonic() + 25.0
                while time.monotonic() < deadline:
                    svc.heartbeat("op1", ep)
                    cur = svc.get_snapshot()
                    if (
                        cur["session_state"] == SessionState.READY.value
                        and cur["metrics"]["completed_count"] == 3
                        and svc._worker is not None
                        and not svc._worker.is_alive()
                    ):
                        break
                    time.sleep(0.03)
                snap = svc.get_snapshot()
                self.assertEqual(observed_windows["red_descend"], "PRE_GRASP_IMMEDIATE")
                self.assertEqual(observed_windows["blue_grasp"], "HOLDING_QUEUED_NEXT")
                self.assertEqual(snap["session_state"], SessionState.READY.value)
                self.assertEqual(snap["metrics"]["headline_score"], "3/3")
                self.assertEqual(snap["metrics"]["operator_interventions"], 0)
                self.assertEqual(hw.state["placed_colors"], ["blue", "yellow", "red"])
                preempt_events = [ev for ev in snap["events"] if ev["kind"] == "PREEMPTED_BEFORE_GRASP"]
                self.assertEqual(len(preempt_events), 1)
                self.assertEqual(preempt_events[0]["preempted_object_id"], "obj_red_R3_1")
                self.assertEqual(preempt_events[0]["after_phase"], "DESCENDING_CONTACT")
                self.assertEqual(preempt_events[0]["before_phase"], "GRASPING")
            finally:
                bridge.shutdown_hardware(safe_normal_completion=True)
                srv.shutdown()
                srv.server_close()

    def test_27_manual_return_home_button_safety_and_recovery_to_ready(self) -> None:
        """RETURN_HOME rejects closed gripper (<45mm), and when open recovers PAUSED->READY safely via lift+go_home."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rep = _write_dryrun_report(tmp_path / "ok.json")
            hw = _SimulatedStationHardware({"R3": "red"})
            srv = _start_mock_right7_server(_MockServerState(hw, checkpoint=_guarded.EXPECTED_CHECKPOINT))
            port = srv.server_address[1]
            try:
                cfg = LiveHardwareConfig(
                    server_url=f"http://127.0.0.1:{port}",
                    dry_run_report_path=rep,
                    operator_admitted=True,
                    confirm_startup_home=True,
                    confirm_exit_home=True,
                    return_home_on_batch_complete=False,
                    unsafe_exit_action="immediate_disable",
                    robot_init_settle_s=0.0,
                    settle_margin_s=0.0,
                    settle_drift_sleep_s=0.0,
                    chunk_wait_slice_s=0.01,
                    grasp_duration_s=0.05,
                    speed=10.0,
                    takeoff_speed=10.0,
                    aerial_approach_speed=10.0,
                    approach_speed=10.0,
                    transport_speed=10.0,
                    place_speed=10.0,
                )
                bridge = LiveRight7HardwareBridge(
                    config=cfg,
                    robot_cls=hw.make_robot_cls(),
                    image_client_cls=hw.make_image_client_cls(),
                    sleep_fn=lambda _s: None,
                )
                bridge.admit_hardware()
                bridge._ensure_hardware_attached()
                from v070_demo.contracts import ContactHeldState
                from v070_demo.event_logger import DemoRunLogger

                svc = SessionService(bridge, logger=DemoRunLogger(tmp_path / "run_manual_home"))
                svc.start_session("op1", scene_id="scene_1_standard")
                ep = svc.get_snapshot()["lease"]["lease_epoch"]
                home_calls_before = hw.state["go_home_calls"]

                # Simulate paused-after-placement_unconfirmed with gripper accidentally closed (< 45mm): MUST REJECT
                with svc._lock:
                    svc.session_state = SessionState.PAUSED
                    svc.contact_state = ContactHeldState.HELD_UNCERTAIN
                bridge._holding_object = True
                hw.state["grip_m"] = 0.028
                rec_reject = svc.submit_command(
                    client_id="op1",
                    command_id="home_reject_closed",
                    command_type=CommandType.RETURN_HOME,
                    lease_epoch=ep,
                )
                self.assertEqual(rec_reject.status, CommandStatus.REJECTED)
                self.assertIn("unsafe_home_gripper_closed", rec_reject.reason)
                self.assertEqual(hw.state["go_home_calls"], home_calls_before)

                # Now simulate post-release open gripper (>45mm) at low hover z=0.165m: lift + go_home + PAUSED->READY
                hw.state["grip_m"] = 0.068
                hw.state["pose6"] = np.array([0.348, 0.216, 0.165, 0.0, 0.0, 0.0], dtype=float)
                rec_ok = svc.submit_command(
                    client_id="op1",
                    command_id="home_allow_open",
                    command_type=CommandType.RETURN_HOME,
                    lease_epoch=ep,
                )
                self.assertEqual(rec_ok.status, CommandStatus.APPLIED)
                snap = svc.get_snapshot()
                self.assertEqual(snap["session_state"], SessionState.READY.value)
                self.assertEqual(snap["contact_held_state"], ContactHeldState.EMPTY_CONFIRMED.value)
                self.assertEqual(hw.state["go_home_calls"], home_calls_before + 1)
                self.assertTrue(any(ev["kind"] == "RETURNED_HOME_MANUAL" for ev in snap["events"]))
            finally:
                bridge.shutdown_hardware(safe_normal_completion=True)
                srv.shutdown()
                srv.server_close()

    def test_28_phase_a_step0_hover_descent_and_lazy_clear_obs(self) -> None:
        """Phase A: aligned hover descends on step 0 without flat hover loop; CLEAR_OBS runs lazily on observe_scene."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rep = _write_dryrun_report(tmp_path / "ok.json")
            hw = _SimulatedStationHardware({"R3": "red", "R1": "yellow", "R8": "blue"})
            mock_state = _MockServerState(hw, checkpoint=_guarded.EXPECTED_CHECKPOINT)
            # Omit gripper <0.04 closure so condition_approach_actions stays flat at hover z=0.200m,
            # verifying step_idx == 0 immediate visual-aligned descent triggers on the very first step!
            mock_state.omit_closure_for_grasp = True
            srv = _start_mock_right7_server(mock_state)
            port = srv.server_address[1]
            try:
                snap_dir = tmp_path / "snaps_a"
                cfg = LiveHardwareConfig(
                    server_url=f"http://127.0.0.1:{port}",
                    dry_run_report_path=rep,
                    operator_admitted=True,
                    confirm_startup_home=True,
                    confirm_exit_home=True,
                    return_home_on_batch_complete=True,
                    unsafe_exit_action="immediate_disable",
                    use_clear_obs_retreat=True,
                    snapshot_dir=snap_dir,
                    robot_init_settle_s=0.0,
                    settle_margin_s=0.0,
                    settle_drift_sleep_s=0.0,
                    chunk_wait_slice_s=0.01,
                    grasp_duration_s=0.05,
                    speed=10.0,
                    takeoff_speed=10.0,
                    aerial_approach_speed=10.0,
                    approach_speed=10.0,
                    transport_speed=10.0,
                    place_speed=10.0,
                )
                bridge = LiveRight7HardwareBridge(
                    config=cfg,
                    robot_cls=hw.make_robot_cls(),
                    image_client_cls=hw.make_image_client_cls(),
                    sleep_fn=lambda _s: None,
                )
                bridge.admit_hardware()
                svc = SessionService(bridge)
                svc.start_session("op1", scene_id="scene_1_standard")
                ep = svc.get_snapshot()["lease"]["lease_epoch"]
                rec = svc.submit_command(
                    client_id="op1",
                    command_id="phase_a_1",
                    command_type=CommandType.START_BATCH,
                    raw_text="Pick up the three wooden blocks (red, yellow, and blue) in exact order and place them on the brown target area.",
                    scene_id="scene_1_standard",
                    lease_epoch=ep,
                )
                self.assertEqual(rec.status, CommandStatus.APPLIED)
                deadline = time.monotonic() + 25.0
                while time.monotonic() < deadline:
                    svc.heartbeat("op1", ep)
                    cur = svc.get_snapshot()
                    if (
                        cur["session_state"] == SessionState.READY.value
                        and cur["metrics"]["completed_count"] == 3
                        and svc._worker is not None
                        and not svc._worker.is_alive()
                    ):
                        break
                    time.sleep(0.03)
                snap = svc.get_snapshot()
                self.assertEqual(snap["metrics"]["headline_score"], "3/3")
                self.assertEqual(hw.state["placed_colors"], ["red", "yellow", "blue"])

                # Verify step 0 descended immediately on all 3 objects -> ZERO approach_1 snapshots!
                self.assertEqual(list(snap_dir.glob("*_approach_1_*")), [])

                # Smart Phase B CLEAR_OBS skips all unnecessary post-release detours when remaining colors are visible!
                clear_obs_chunks = [
                    c for c in hw.state["sent_chunks"]
                    if np.linalg.norm(np.asarray(c["end_pose6"][:3], dtype=float) - CLEAR_OBS_POSE6[:3]) < 0.03
                ]
                self.assertEqual(clear_obs_chunks, [], msg=f"Unexpected CLEAR_OBS detours in Scene 1: {clear_obs_chunks}")

                # Verify CLEAR_OBS DOES trigger on demand when checking empty table from brown hover!
                hw.state["pose6"] = np.array([0.364, 0.239, 0.200, -0.45, 0.90, 0.94], dtype=float)
                hw.state["grip_m"] = 0.068
                bridge.observe_scene("scene_clear_table", time.monotonic())
                clear_obs_after_obs = [
                    c for c in hw.state["sent_chunks"]
                    if np.linalg.norm(np.asarray(c["end_pose6"][:3], dtype=float) - CLEAR_OBS_POSE6[:3]) < 0.03
                ]
                self.assertEqual(len(clear_obs_after_obs), 1)
            finally:
                bridge.shutdown_hardware(safe_normal_completion=True)
                srv.shutdown()
                srv.server_close()

    def test_29_slow_observe_scene_does_not_starve_heartbeat_or_expire_lease(self) -> None:
        """Regression for r15: releasing _lock around observe_scene prevents lease starvation during lazy CLEAR_OBS."""
        from v070_demo.contracts import SafetyBudgets
        from v070_demo.executor_bridge import MockRobotPublisher

        bridge = MockRobotPublisher(step_delay_s=0.01)
        orig_observe = bridge.observe_scene

        def slow_observe(scene_id: str, now_mono: float):
            time.sleep(0.55)  # Exceeds short 0.35s lease_ttl_s!
            return orig_observe(scene_id, now_mono)

        bridge.observe_scene = slow_observe  # type: ignore[method-assign]
        svc = SessionService(
            bridge,
            budgets=SafetyBudgets(lease_ttl_s=0.35, heartbeat_interval_s=0.05),
            watchdog_poll_s=0.05,
        )
        try:
            svc.start_session("op_hb", scene_id="scene_1_standard")
            ep = svc.get_snapshot(client_id="op_hb")["lease"]["lease_epoch"]
            hb_stop = threading.Event()
            hb_failures: list[str] = []

            def hb_loop() -> None:
                while not hb_stop.is_set():
                    ok, reason = svc.heartbeat("op_hb", ep)
                    if not ok:
                        hb_failures.append(reason)
                    time.sleep(0.05)

            t = threading.Thread(target=hb_loop, daemon=True)
            t.start()
            try:
                rec = svc.submit_command(
                    client_id="op_hb",
                    command_id="slow_obs_batch",
                    command_type=CommandType.START_BATCH,
                    raw_text="Pick up the three wooden blocks (red, yellow, and blue) in exact order and place them on the brown target area.",
                    scene_id="scene_1_standard",
                    lease_epoch=ep,
                )
                self.assertEqual(rec.status, CommandStatus.APPLIED)
                deadline = time.monotonic() + 10.0
                while time.monotonic() < deadline:
                    cur = svc.get_snapshot(client_id="op_hb")
                    if cur["session_state"] == SessionState.READY.value and cur["metrics"]["completed_count"] == 3:
                        break
                    time.sleep(0.03)
            finally:
                hb_stop.set()
                t.join(timeout=2.0)

            snap = svc.get_snapshot(client_id="op_hb")
            self.assertEqual(hb_failures, [])
            self.assertFalse(snap["lease"]["expired"])
            self.assertEqual(snap["session_state"], SessionState.READY.value)
            self.assertEqual(snap["metrics"]["headline_score"], "3/3")
            self.assertFalse(any(ev["kind"] == "CONTROLLER_LEASE_EXPIRED" for ev in snap["events"]))
        finally:
            svc._watchdog_stop.set()

    def test_30_grasp_ood_envelope_jump_uses_calibrated_wrist_fallback_when_tight_aligned(self) -> None:
        """Regression for r16 blue R7: OOD >40mm grasp raw jump uses wrist-OBB closure iff tight aligned, else raises."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rep = _write_dryrun_report(tmp_path / "ok.json")
            hw = _SimulatedStationHardware({"R7": "blue"})
            _BaseImg = hw.make_image_client_cls()
            wrist_visible = {"on": True}

            class _BlueCornerWristImg(_BaseImg):
                def get_right_wrist_frame(self):
                    if not wrist_visible["on"]:
                        return _render_wrist_cam(block_color=None, show_brown_target=False), time.monotonic()
                    uv = _grasp_cases.wrist_target_uv_for_pose(
                        ee_pose6=hw.state["pose6"],
                        ee_z_m=float(hw.state["pose6"][2]),
                    )
                    return _render_wrist_cam(block_color="blue", target_uv=uv), time.monotonic()

            mock_state = _MockServerState(hw, checkpoint=_guarded.EXPECTED_CHECKPOINT)
            srv = _start_mock_right7_server(mock_state)
            port = srv.server_address[1]
            try:
                cfg = LiveHardwareConfig(
                    server_url=f"http://127.0.0.1:{port}",
                    dry_run_report_path=rep,
                    operator_admitted=True,
                    confirm_startup_home=True,
                    confirm_exit_home=True,
                    unsafe_exit_action="immediate_disable",
                    robot_init_settle_s=0.0,
                    settle_margin_s=0.0,
                    settle_drift_sleep_s=0.0,
                    chunk_wait_slice_s=0.01,
                    grasp_duration_s=0.05,
                    speed=10.0,
                )
                bridge = LiveRight7HardwareBridge(
                    config=cfg,
                    robot_cls=hw.make_robot_cls(),
                    image_client_cls=_BlueCornerWristImg,
                    sleep_fn=lambda _s: None,
                )
                bridge.admit_hardware()
                bridge._ensure_hardware_attached()

                from v070_demo.contracts import ExecutionPhase, ObjectRecord

                obj = ObjectRecord(
                    object_id="obj_blue_R7_1",
                    category="woodblock",
                    color="blue",
                    pose_estimate={"x": 0.22, "y": 0.17, "z": 0.03, "yaw_deg": 0.0},
                    confidence=0.98,
                    observation_id="obs1",
                    observation_ts_mono=time.monotonic(),
                    grid_cell="R7",
                )
                # Inject exact r16 OOD raw prediction (+45.7mm jump + lift while gripper predicts closure < 0.04)
                ood_raw = np.repeat(np.array([[0.2380, 0.1843, 0.1756, -0.3035, 0.9678, 0.9275, 0.0245]]), 50, axis=0)
                ood_raw[-1, :3] = [0.2493, 0.1506, 0.2051]

                def fake_query(*, task: str):
                    _, raw_obs = bridge._read_live_observation(task=task)
                    return {"observation": True}, raw_obs, ood_raw.copy()

                bridge._query_policy_actions = fake_query  # type: ignore[method-assign]

                # 1. With tight-aligned blue block under open jaws at z=0.1617 -> succeeds via wrist fallback!
                hw.state["pose6"] = np.array([0.1962, 0.1966, 0.1617, -0.3035, 0.9678, 0.9275], dtype=float)
                hw.state["grip_m"] = 0.0678
                ok, err_c, err_d = bridge._run_grasping(obj)
                self.assertTrue(ok, msg=f"{err_c}: {err_d}")
                self.assertLessEqual(float(hw.state["grip_m"]), 0.01)

                # 2. Without visible block under jaws -> same OOD jump MUST still fail closed!
                wrist_visible["on"] = False
                hw.state["pose6"] = np.array([0.1962, 0.1966, 0.1617, -0.3035, 0.9678, 0.9275], dtype=float)
                hw.state["grip_m"] = 0.0678
                with self.assertRaises(ValueError):
                    bridge._run_grasping(obj)
            finally:
                bridge.shutdown_hardware(safe_normal_completion=True)
                srv.shutdown()
                srv.server_close()

    def test_31_phase_b_offcenter_40px_single_arc_descent_no_approach_1(self) -> None:
        """Phase B: ~40px off-center wrist OBB descends in 1 continuous overlapped arc (no approach_0+approach_1 split)."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rep = _write_dryrun_report(tmp_path / "ok.json")
            hw = _SimulatedStationHardware({"R3": "red"})
            _BaseImg = hw.make_image_client_cls()
            block_pose = np.array(GRID_HOVER_POSES["R3"], dtype=float)
            # Offset physical block by ~25mm (~32-40px) from nominal R3 center!
            block_pose[0] += 0.024
            block_pose[1] -= 0.018

            class _OffCenterWristImg(_BaseImg):
                def get_right_wrist_frame(self):
                    cur = hw.state["pose6"]
                    # Render wrist image shifted by remaining delta relative to shifted physical block!
                    rel_pose = cur.copy()
                    rel_pose[0] = GRID_HOVER_POSES["R3"][0] + (cur[0] - block_pose[0])
                    rel_pose[1] = GRID_HOVER_POSES["R3"][1] + (cur[1] - block_pose[1])
                    uv = _grasp_cases.wrist_target_uv_for_pose(
                        ee_pose6=rel_pose,
                        ee_z_m=float(cur[2]),
                    )
                    return _render_wrist_cam(block_color="red", target_uv=uv), time.monotonic()

            mock_state = _MockServerState(hw, checkpoint=_guarded.EXPECTED_CHECKPOINT)
            mock_state.omit_closure_for_grasp = True
            srv = _start_mock_right7_server(mock_state)
            port = srv.server_address[1]
            try:
                snap_dir = tmp_path / "snaps_b"
                cfg = LiveHardwareConfig(
                    server_url=f"http://127.0.0.1:{port}",
                    dry_run_report_path=rep,
                    operator_admitted=True,
                    confirm_startup_home=True,
                    confirm_exit_home=True,
                    unsafe_exit_action="immediate_disable",
                    snapshot_dir=snap_dir,
                    robot_init_settle_s=0.0,
                    settle_margin_s=0.0,
                    settle_drift_sleep_s=0.0,
                    chunk_wait_slice_s=0.01,
                    grasp_duration_s=0.05,
                    speed=10.0,
                )
                bridge = LiveRight7HardwareBridge(
                    config=cfg,
                    robot_cls=hw.make_robot_cls(),
                    image_client_cls=_OffCenterWristImg,
                    sleep_fn=lambda _s: None,
                )
                bridge.admit_hardware()
                bridge._ensure_hardware_attached()

                from v070_demo.contracts import ObjectRecord

                obj = ObjectRecord(
                    object_id="obj_red_R3_1",
                    category="woodblock",
                    color="red",
                    pose_estimate={"x": 0.28, "y": 0.11, "z": 0.03, "yaw_deg": 0.0},
                    confidence=0.98,
                    observation_id="obs1",
                    observation_ts_mono=time.monotonic(),
                    grid_cell="R3",
                )
                hw.state["pose6"] = np.asarray(GRID_HOVER_POSES["R3"], dtype=float).copy()
                hw.state["grip_m"] = 0.072

                ok, err_c, err_d = bridge._run_descending_contact(obj)
                self.assertTrue(ok, msg=f"{err_c}: {err_d}")
                self.assertLessEqual(float(hw.state["pose6"][2]), 0.168)
                # Must descend on step 0 in 1 continuous arc (20 or 26 pts), ZERO approach_1!
                self.assertEqual(len(hw.state["sent_chunks"]), 1)
                self.assertIn(hw.state["sent_chunks"][0]["waypoints"], (20, 26))
                self.assertEqual(list(snap_dir.glob("*_approach_1_*")), [])
            finally:
                bridge.shutdown_hardware(safe_normal_completion=True)
                srv.shutdown()
                srv.server_close()


if __name__ == "__main__":
    unittest.main()
