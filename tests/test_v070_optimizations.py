#!/usr/bin/env python3
"""Offline unit and fault-injection tests for v0.7.0 cycle-time & smoothness optimizations."""
from __future__ import annotations

import base64
import json
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

import os

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import build_grasp_replay_cases as cases  # noqa: E402
import station_lite_smolvla_right7_guarded as guarded  # noqa: E402
from station_lite_smolvla_single_chunk import (  # noqa: E402
    GRIPPER_MAX_M,
    JOINT_LOW,
    JOINT_UP,
)


def _encode_jpg_b64(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _make_cam_high_with_yellow_block(cell: str = "R2") -> np.ndarray:
    """Create a 640x480 RGB cam_high image (matching ImageClient RGB contract) with a yellow block (~144-169 px)."""
    img = np.full((480, 640, 3), 235, dtype=np.uint8)
    cx_f, cy_f = cases.HIGH_CAM_GRID_CENTERS_PX[cell]
    cx, cy = int(round(cx_f)), int(round(cy_f))
    img[cy - 6:cy + 7, cx - 6:cx + 7] = (245, 215, 0)  # RGB bright yellow -> BGR (0, 215, 245)
    return img


def _make_wrist_img(offset_px: tuple[int, int] = (0, 0)) -> np.ndarray:
    """Create a 640x480 RGB cam_right_wrist image with a yellow block near grasp target center."""
    img = np.full((480, 640, 3), 230, dtype=np.uint8)
    cx = 320 + offset_px[0]
    cy = 320 + offset_px[1]
    img[cy - 28:cy + 28, cx - 28:cx + 28] = (245, 215, 0)  # RGB bright yellow
    return img


def _make_smooth_actions(start_pose7: np.ndarray, target_xyz=(0.30, 0.20, 0.19), n: int = 50) -> np.ndarray:
    out = np.zeros((n, 7), dtype=float)
    target = np.array([
        target_xyz[0],
        target_xyz[1],
        target_xyz[2],
        start_pose7[3],
        start_pose7[4],
        start_pose7[5],
        start_pose7[6],
    ], dtype=float)
    for i in range(n):
        u = float(i) / float(max(1, n - 1))
        out[i] = start_pose7 + u * (target - start_pose7)
    return out


class _MockSingleOwnerArm:
    def __init__(self, call_threads: list[int], initial_q=None, initial_ee=None, initial_grip=0.079):
        self._threads = call_threads
        self.q = np.array(
            [-0.002, 0.25, -0.0055, -0.45, 0.02, -0.01] if initial_q is None else initial_q,
            dtype=float,
        )
        self.ee_pos = np.array([0.2586, 0.0322, 0.0332] if initial_ee is None else initial_ee[:3], dtype=float)
        self.ee_euler = np.array([0.0, 0.78, 0.0] if initial_ee is None else initial_ee[3:6], dtype=float)
        self.grip = float(initial_grip)
        self.sent_chunks: list[dict] = []
        self.cleanup_calls = 0
        self.soft_boundary_prefix_mode = False
        self.fail_home_takeoff_ik = False
        self.fail_brown_transport_ik = False
        self.simulated_block_width_m = 0.0
        self._block_grasped = False
        self._q_to_pose: dict[tuple[float, ...], tuple[np.ndarray, np.ndarray]] = {}

        class _LowArm:
            def __init__(self, outer):
                self._outer = outer

            def solve_ik(self, target_xyz, target_euler, q_init=None):
                self._outer._threads.append(threading.get_ident())
                if self._outer.fail_home_takeoff_ik and float(target_xyz[2]) > 0.16:
                    return None, False
                if (self._outer.fail_brown_transport_ik
                        and float(target_xyz[0]) > 0.30 and float(target_xyz[2]) > 0.20):
                    return None, False
                base = np.asarray(self._outer.q if q_init is None else q_init, dtype=float).copy()
                # Map xyz slightly so every pose has distinct valid finite IK within bounds
                d = np.asarray(target_xyz, dtype=float) - np.array([0.2586, 0.0322, 0.0332])
                e = np.asarray(target_euler, dtype=float) - np.array([0.0, 0.78, 0.0])
                cand = np.array([
                    np.clip(d[1] * 0.8, JOINT_LOW[0] + 0.05, JOINT_UP[0] - 0.05),
                    np.clip(0.25 + d[0] * 1.2 + d[2] * 0.4, JOINT_LOW[1] + 0.05, JOINT_UP[1] - 0.05),
                    np.clip(-0.08 - d[2] * 1.5 - d[0] * 0.6, JOINT_LOW[2] + 0.05, JOINT_UP[2] - 0.05),
                    np.clip(-0.45 + 0.3 * d[2], JOINT_LOW[3] + 0.05, JOINT_UP[3] - 0.05),
                    np.clip(e[1] * 0.2, JOINT_LOW[4] + 0.05, JOINT_UP[4] - 0.05),
                    np.clip(e[2] * 0.2, JOINT_LOW[5] + 0.05, JOINT_UP[5] - 0.05),
                ], dtype=float)
                if self._outer.soft_boundary_prefix_mode and float(target_xyz[2]) < 0.038:
                    # Simulate real hardware q[2] > -0.020 right near Home until z >= 0.038
                    cand[2] = -0.010
                self._outer._q_to_pose[tuple(np.round(cand, 5))] = (
                    np.asarray(target_xyz, dtype=float).copy(),
                    np.asarray(target_euler, dtype=float).copy(),
                )
                _ = base
                return cand, True

            def plan_joint_waypoints_with_gripper(self, start_q, waypoints, grippers, time_sec):
                self._outer._threads.append(threading.get_ident())
                n = max(4, min(20, len(waypoints)))
                out = np.zeros((n, 16), dtype=float)
                out[:, 0] = np.linspace(0.0, float(time_sec), n)
                q0 = np.asarray(start_q, dtype=float)
                q1 = np.asarray(waypoints[-1], dtype=float)
                for i in range(n):
                    alpha = float(i) / float(n - 1)
                    out[i, 1:7] = q0 + alpha * (q1 - q0)
                return out

            def cleanup(self):
                self._outer._threads.append(threading.get_ident())
                self._outer.cleanup_calls += 1

        self.arm = _LowArm(self)

    def get_joint_positions(self):
        self._threads.append(threading.get_ident())
        return self.q.copy()

    def get_gripper_distance(self):
        self._threads.append(threading.get_ident())
        return float(self.grip)

    def get_ee_pose_euler(self):
        self._threads.append(threading.get_ident())
        return self.ee_pos.copy(), self.ee_euler.copy()

    def solve_ik_euler(self, pos, euler, seed=None):
        return self.arm.solve_ik(pos, euler, seed)

    def update_joint_waypoint_chunk_with_gripper(self, waypoints, grippers, time_sec, switch_delay_sec=0.05):
        self._threads.append(threading.get_ident())
        q_end = np.asarray(waypoints[-1], dtype=float).copy()
        self.q = q_end
        cmd_grip = float(grippers[-1]) * GRIPPER_MAX_M
        if self.simulated_block_width_m > 0.0:
            if not self._block_grasped and float(self.ee_pos[2]) <= 0.170 and cmd_grip < self.simulated_block_width_m:
                self._block_grasped = True
                cmd_grip = self.simulated_block_width_m
            elif self._block_grasped:
                if cmd_grip >= 0.045:
                    self._block_grasped = False
                elif cmd_grip < self.simulated_block_width_m:
                    cmd_grip = self.simulated_block_width_m
        self.grip = cmd_grip
        pose_end = self._q_to_pose.get(tuple(np.round(q_end, 5)))
        if pose_end is not None:
            self.ee_pos = pose_end[0].copy()
            self.ee_euler = pose_end[1].copy()
        self.sent_chunks.append({
            "waypoints": np.asarray(waypoints, dtype=float).copy(),
            "grippers": np.asarray(grippers, dtype=float).copy(),
            "time_sec": float(time_sec),
            "segment_end": None if pose_end is None else np.r_[pose_end[0], pose_end[1], self.grip],
        })

    def set_next_ee_from_hover(self, pose7: np.ndarray):
        self.ee_pos = np.asarray(pose7[:3], dtype=float).copy()
        self.ee_euler = np.asarray(pose7[3:6], dtype=float).copy()
        self.grip = float(pose7[6])


class TestV070Optimizations(unittest.TestCase):
    def test_option_a_home_takeoff_all_grid_cells_r1_to_r9_and_prefix(self):
        start_home = np.array([0.2586, 0.0322, 0.0332, -0.42, 0.99, 0.96, 0.079], dtype=float)
        for cell, hover_pose in cases.GRID_HOVER_POSES.items():
            traj = cases.condition_home_takeoff_actions(
                start_home,
                hover_pose=hover_pose,
                hover_z_m=0.200,
                open_grip=0.079,
                hold_tail_steps=3,
            )
            self.assertGreaterEqual(len(traj), 30, f"cell={cell} too short")
            self.assertLessEqual(len(traj), 44, f"cell={cell} too long")
            self.assertTrue(np.isfinite(traj).all(), f"cell={cell} non-finite")
            entry_gap_mm = 1000.0 * float(np.linalg.norm(traj[0, :3] - start_home[:3]))
            entry_rot_deg = float(np.degrees(np.max(np.abs(
                (traj[0, 3:6] - start_home[3:6] + np.pi) % (2.0 * np.pi) - np.pi
            ))))
            self.assertLessEqual(entry_gap_mm, 23.0, f"cell={cell} entry_gap_mm={entry_gap_mm:.2f}")
            self.assertLessEqual(entry_rot_deg, 10.0, f"cell={cell} entry_rot_deg={entry_rot_deg:.2f}")
            step_xyz_mm = 1000.0 * np.linalg.norm(np.diff(traj[:, :3], axis=0), axis=1)
            self.assertLessEqual(float(np.max(step_xyz_mm)), 14.5, f"cell={cell} max step mm")
            self.assertGreaterEqual(float(traj[-1, 2]), 0.200 - 1e-6, f"cell={cell} hover Z")
            self.assertLessEqual(float(np.linalg.norm(traj[-1, :2] - hover_pose[:2])), 1e-6)
            self.assertTrue(np.allclose(traj[-3:], traj[-1], atol=1e-9), f"cell={cell} stationary tail")
            self.assertTrue(np.all((traj[:, 6] >= 0.0) & (traj[:, 6] <= GRIPPER_MAX_M)))

        # Test soft-boundary filtering in make_initial_prefix + select_initial_prefix
        threads: list[int] = []
        arm = _MockSingleOwnerArm(threads)
        arm.soft_boundary_prefix_mode = True
        r2_traj = cases.condition_home_takeoff_actions(
            start_home, hover_pose=cases.GRID_HOVER_POSES["R2"], hover_z_m=0.200
        )
        raw_prefix = guarded.make_initial_prefix(start_home, r2_traj[0], 6)
        filtered, omitted = guarded.select_initial_prefix(arm, raw_prefix, margin=0.02)
        self.assertGreaterEqual(omitted, 1)
        self.assertGreaterEqual(len(filtered), 1)
        self.assertLessEqual(1000.0 * float(np.linalg.norm(filtered[0, :3] - start_home[:3])), 25.0)

    def test_r2_r1_far_grid_shortcut_on_real_and_synthetic_snapshots(self):
        # 1. Synthetic edge wrist_obb trap (>120mm from R2 hover)
        start = np.array([0.214, 0.289, 0.220, -0.42, 0.99, 0.96, 0.080], dtype=float)
        raw = _make_smooth_actions(start, target_xyz=(0.24, 0.22, 0.19), n=50)
        r2_hover = cases.GRID_HOVER_POSES["R2"]
        edge_obb = {"cx": 210.0, "cy": 330.0, "w": 60.0, "h": 60.0, "angle": 90.0}

        seg_legacy, _, anchored_legacy = cases.condition_approach_actions(
            start, raw, wrist_obb=edge_obb, hover_z_m=0.200,
            grid_hover_pose=r2_hover, prefer_grid_hover_when_far=False, adaptive_steps=False,
        )
        seg_fast, held_open, anchored_fast = cases.condition_approach_actions(
            start, raw, wrist_obb=edge_obb, hover_z_m=0.200,
            grid_hover_pose=r2_hover, prefer_grid_hover_when_far=True, adaptive_steps=True,
        )
        self.assertTrue(anchored_legacy)
        self.assertTrue(anchored_fast)
        self.assertEqual(held_open, 0)
        self.assertLessEqual(len(seg_fast), 44)
        # Legacy crawled max 55mm while fast shortcut lands right over calibrated R2 hover_xy
        self.assertGreater(float(np.linalg.norm(seg_legacy[-1, :2] - r2_hover[:2])), 0.06)
        self.assertLessEqual(float(np.linalg.norm(seg_fast[-1, :2] - r2_hover[:2])), 1e-5)

        # 2. Near-hover case (<= 45mm from hover): centered wrist_obb takes over instead of locking at hover
        near_start = np.r_[r2_hover[:2] + np.array([0.005, -0.005]), 0.200, r2_hover[3:6], 0.080]
        raw_near = _make_smooth_actions(near_start, target_xyz=(r2_hover[0], r2_hover[1], 0.150), n=50)
        offset_obb = {"cx": 360.0, "cy": 300.0, "w": 62.0, "h": 60.0, "angle": 90.0}
        seg_near, _, _ = cases.condition_approach_actions(
            near_start, raw_near, wrist_obb=offset_obb, hover_z_m=0.200,
            grid_hover_pose=r2_hover, prefer_grid_hover_when_far=True, adaptive_steps=True,
        )
        self.assertGreater(float(np.linalg.norm(seg_near[-1, :2] - near_start[:2])), 0.002)

        # 3. Real 20261113_live R2 (122628) and R1 (122818) snapshots when present in workspace
        for run_name in (
            "right7_R2_yellow_pick_place_20261113_122628_snapshot",
            "right7_R1_yellow_pick_place_20261113_122818_snapshot",
        ):
            snap_root = os.environ.get("SMOLVLA_LOCAL_DIAGNOSTICS_DIR", "")
            if not snap_root:
                continue
            snap_dir = Path(snap_root) / run_name
            if not snap_dir.exists():
                continue
            st = json.loads((snap_dir / "loop_01_state.json").read_text(encoding="utf-8"))
            raw_actions = np.load(snap_dir / "loop_01_raw_actions.npy")
            st0 = json.loads((snap_dir / "loop_00_state.json").read_text(encoding="utf-8"))
            _ = st0
            hi = cases.detect_high_cam_grid_target((snap_dir / "loop_01_cam_high.jpg").read_bytes(), target_color="yellow")
            wrist = cases.detect_wrist_block_obb((snap_dir / "loop_01_cam_right_wrist.jpg").read_bytes(), target_color="yellow")
            self.assertIsNotNone(hi, msg=run_name)
            start_real = np.r_[st["right_ee_position_m"], st["right_ee_euler_rad"], st["state"][6]]
            seg_real, _, _ = cases.condition_approach_actions(
                start_real, raw_actions, max_position_m=0.050, wrist_obb=wrist, hover_z_m=0.200,
                grid_hover_pose=hi["hover_pose"], prefer_grid_hover_when_far=True, adaptive_steps=True,
            )
            self.assertLessEqual(float(np.linalg.norm(seg_real[-1, :2] - hi["hover_pose"][:2])), 1e-4, msg=run_name)
            self.assertLess(len(seg_real), 50, msg=run_name)

    def test_trim_segment_idle_tail_preserves_target_hold_and_min_points(self):
        # 25 moving points + 25 repeated stationary points -> trim keeping hold_tail=2
        moving = np.column_stack([
            np.linspace(0.25, 0.32, 25),
            np.linspace(0.05, 0.20, 25),
            np.linspace(0.20, 0.16, 25),
            np.full(25, -0.42),
            np.full(25, 0.99),
            np.full(25, 0.96),
            np.full(25, 0.080),
        ])
        tail = np.tile(moving[-1], (25, 1))
        seg = np.vstack([moving, tail])

        trimmed, removed = cases.trim_segment_idle_tail(seg, keep_tail_hold=2, min_points=12)
        self.assertEqual(removed, 24)
        self.assertEqual(len(trimmed), 26)
        self.assertTrue(np.allclose(trimmed[-1], seg[-1]))
        self.assertTrue(np.allclose(trimmed[-2:], seg[-1]))

        # Short or min_points floor respected
        trimmed_min, removed_min = cases.trim_segment_idle_tail(seg[:15], keep_tail_hold=2, min_points=14)
        self.assertEqual(len(trimmed_min), 14 if removed_min else 15)
        self.assertGreaterEqual(len(trimmed_min), 14)

    def test_async_snapshot_writer_offloads_and_flushes_cleanly(self):
        main_tid = threading.get_ident()
        with tempfile.TemporaryDirectory() as tmp:
            snap = Path(tmp) / "snap"
            writer = guarded.AsyncSnapshotWriter(enabled=True)
            writer.submit_npy(snap / "arr.npy", np.arange(14, dtype=float).reshape(2, 7))
            writer.submit_bytes(snap / "cam.jpg", b"jpeg-bytes-ok")
            writer.submit_text(snap / "state.json", '{"seq": 1}\n')
            writer.submit_save_preflight(
                snap / "pre",
                {"state": [0.0] * 7, "images_jpeg_base64": {"observation.images.cam_high": base64.b64encode(b"hi").decode()}},
                np.zeros((50, 7)),
                np.zeros(6),
                0.08,
                {"t": 1.0},
                {"ok": True},
            )
            worker_tid = writer._thread.ident if writer._thread else main_tid
            self.assertNotEqual(worker_tid, main_tid)
            writer.close(timeout_s=2.0)
            self.assertEqual(writer.errors, [])
            self.assertGreaterEqual(writer.completed_jobs, 4)
            self.assertEqual(np.load(snap / "arr.npy").shape, (2, 7))
            self.assertEqual((snap / "cam.jpg").read_bytes(), b"jpeg-bytes-ok")
            self.assertIn('"seq": 1', (snap / "state.json").read_text(encoding="utf-8"))
            self.assertTrue((snap / "pre" / "preflight.json").exists())

    def test_async_predict_client_overlap_and_fault_injection(self):
        main_tid = threading.get_ident()
        seen_threads: list[int] = []

        def _ok_predict(payload):
            seen_threads.append(threading.get_ident())
            time.sleep(0.08)
            return {
                "schema": guarded.SCHEMA,
                "session_id": payload["session_id"],
                "seq": payload["seq"],
                "action_dim": 7,
                "hardware_connected": False,
                "sampling_seed": 42,
                "actions": np.zeros((50, 7), dtype=float).tolist(),
            }

        client = guarded.AsyncPredictClient("http://127.0.0.1:8011", expected_seed_mode="fixed", expected_seed_base=42, predict_fn=_ok_predict)
        try:
            q0 = np.zeros(6, dtype=float)
            t0 = time.monotonic()
            slot = client.submit(
                {"session_id": "s1", "seq": 1, "state": [0.0] * 7, "images_jpeg_base64": {}},
                obs_q=q0, obs_gripper=0.08,
                timing={"first_camera": t0, "last_camera": t0, "state_received": t0},
            )
            # Simulate concurrent settle sleep on main thread
            time.sleep(0.09)
            wait_t0 = time.monotonic()
            ok, res = client.poll(slot, current_q=q0, current_gripper=0.08, max_wait_s=0.10, max_observation_age_ms=800.0)
            poll_elapsed = time.monotonic() - wait_t0
            self.assertTrue(ok, msg=res)
            self.assertLess(poll_elapsed, 0.04)  # HTTP finished concurrently during main thread sleep
            self.assertNotEqual(seen_threads[-1], main_tid)

            # Fault injection: joint drift > 0.015 rad -> fail-closed
            slot_drift = client.submit(
                {"session_id": "s1", "seq": 2, "state": [0.0] * 7, "images_jpeg_base64": {}},
                obs_q=q0, obs_gripper=0.08,
                timing={"first_camera": time.monotonic(), "last_camera": time.monotonic(), "state_received": time.monotonic()},
            )
            ok_drift, reason_drift = client.poll(slot_drift, current_q=q0 + 0.03, current_gripper=0.08, max_wait_s=0.3)
            self.assertFalse(ok_drift)
            self.assertEqual(reason_drift["reason"], "prefetch_joint_drift")

            # Fault injection: stale observation (> 0.78 * 800ms) -> fail-closed
            old_t = time.monotonic() - 0.70
            slot_stale = client.submit(
                {"session_id": "s1", "seq": 3, "state": [0.0] * 7, "images_jpeg_base64": {}},
                obs_q=q0, obs_gripper=0.08,
                timing={"first_camera": old_t, "last_camera": old_t, "state_received": old_t},
            )
            ok_stale, reason_stale = client.poll(slot_stale, current_q=q0, current_gripper=0.08, max_wait_s=0.3, max_observation_age_ms=800.0)
            self.assertFalse(ok_stale)
            self.assertEqual(reason_stale["reason"], "prefetch_obs_stale")

            # Fault injection: timeout on slow server -> fail-closed
            slot_timeout = client.submit(
                {"session_id": "s1", "seq": 4, "state": [0.0] * 7, "images_jpeg_base64": {}},
                obs_q=q0, obs_gripper=0.08,
                timing={"first_camera": time.monotonic(), "last_camera": time.monotonic(), "state_received": time.monotonic()},
            )
            ok_to, reason_to = client.poll(slot_timeout, current_q=q0, max_wait_s=0.01)
            self.assertFalse(ok_to)
            self.assertEqual(reason_to["reason"], "prefetch_timeout")
        finally:
            client.close()

        # Fault injection: server error -> fail-closed
        err_client = guarded.AsyncPredictClient(
            "http://127.0.0.1:8011",
            predict_fn=lambda _p: (_ for _ in ()).throw(RuntimeError("http 503")),
        )
        try:
            slot_err = err_client.submit(
                {"session_id": "s2", "seq": 0, "state": [0.0] * 7, "images_jpeg_base64": {}},
                obs_q=np.zeros(6), obs_gripper=0.08,
                timing={"first_camera": time.monotonic()},
            )
            ok_err, reason_err = err_client.poll(slot_err, current_q=np.zeros(6), max_wait_s=0.2)
            self.assertFalse(ok_err)
            self.assertIn("prefetch_error:RuntimeError", reason_err["reason"])
        finally:
            err_client.close()

    def test_t3_evaluate_async_splice_candidate_pass_and_reject(self):
        prev_seg = np.column_stack([
            np.linspace(0.25, 0.35, 30),
            np.linspace(0.05, 0.15, 30),
            np.linspace(0.20, 0.20, 30),
            np.full(30, -0.42),
            np.full(30, 0.99),
            np.full(30, 0.96),
            np.full(30, 0.080),
        ])
        prev_wps = np.tile(np.linspace(-0.3, 0.1, 30)[:, None], (1, 6)) * 0.2
        raw_new = np.tile(prev_seg[15], (50, 1))
        raw_new[:, 0] += np.linspace(0.0, 0.02, 50)
        q_obs = prev_wps[15].copy()
        now = 100.5
        ok, seg, info, reasons = guarded.evaluate_async_splice_candidate(
            previous_segment=prev_seg,
            previous_waypoints=prev_wps,
            raw_actions=raw_new,
            observed_q=q_obs,
            measured_q=q_obs,
            now_monotonic=now,
            install_monotonic=100.0,
            first_camera_monotonic=100.2,
            segment_duration_s=1.0,
            max_observation_age_ms=800.0,
        )
        self.assertTrue(ok, msg=reasons)
        self.assertIsNotNone(seg)
        self.assertEqual(reasons, [])
        self.assertGreaterEqual(info["reserve"], 0)

        # Reject on expired observation
        ok_exp, _, _, reasons_exp = guarded.evaluate_async_splice_candidate(
            previous_segment=prev_seg,
            previous_waypoints=prev_wps,
            raw_actions=raw_new,
            observed_q=q_obs,
            measured_q=q_obs,
            now_monotonic=101.2,
            install_monotonic=100.0,
            first_camera_monotonic=100.1,
            segment_duration_s=1.0,
            max_observation_age_ms=800.0,
        )
        self.assertFalse(ok_exp)
        self.assertIn("observation_expired", reasons_exp)

    def _run_main_with_mock_env(self, *, cam_high_bgr: np.ndarray, argv_overrides=(), fail_ik: bool = False,
                                fail_brown_transport_ik: bool = False, simulated_block_width_m: float = 0.0,
                                interrupt_on_second_send: bool = False, predict_delay_s: float = 0.02,
                                close_from_seq: int | None = None, action_fn=None):
        main_tid = threading.get_ident()
        sdk_threads: list[int] = []
        img_threads: list[int] = []
        predict_seqs: list[int] = []

        arm_holder: list[_MockSingleOwnerArm] = []

        class _ServerHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                return

            def do_GET(self):
                if self.path == "/health":
                    body = json.dumps({
                        "schema": guarded.SCHEMA,
                        "source_schema": guarded.REQUEST_SCHEMA,
                        "status": "ok",
                        "checkpoint": guarded.EXPECTED_CHECKPOINT,
                        "chunk_size": 50,
                        "state_dim": 7,
                        "action_dim": 7,
                        "task": guarded.TASK,
                        "device": "cuda",
                        "hardware_connected": False,
                        "seed_base": 42,
                        "seed_mode": "fixed",
                    }).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                if self.path != "/predict":
                    self.send_response(404)
                    self.end_headers()
                    return
                n = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(n).decode("utf-8"))
                seq_i = int(req["seq"])
                predict_seqs.append(seq_i)
                if predict_delay_s > 0:
                    time.sleep(predict_delay_s)
                start_pose7 = np.array(arm_holder[0].ee_pos.tolist() + arm_holder[0].ee_euler.tolist() + [arm_holder[0].grip])
                if action_fn is not None:
                    actions = action_fn(seq_i, start_pose7)
                elif close_from_seq is not None and seq_i >= close_from_seq:
                    actions = _make_smooth_actions(
                        start_pose7,
                        target_xyz=(cases.GRID_HOVER_POSES["R2"][0], cases.GRID_HOVER_POSES["R2"][1], 0.145),
                        n=50,
                    )
                    actions[20:, 6] = 0.028
                else:
                    # Smooth approach near R2 hover with gentle z step and 15 repeated idle points at tail
                    actions = _make_smooth_actions(
                        start_pose7,
                        target_xyz=(cases.GRID_HOVER_POSES["R2"][0], cases.GRID_HOVER_POSES["R2"][1], 0.195),
                        n=35,
                    )
                    tail = np.tile(actions[-1], (15, 1))
                    actions = np.vstack([actions, tail])
                resp = {
                    "schema": guarded.SCHEMA,
                    "session_id": req["session_id"],
                    "seq": req["seq"],
                    "action_dim": 7,
                    "hardware_connected": False,
                    "sampling_seed": 42,
                    "actions": actions.tolist(),
                }
                body = json.dumps(resp).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), _ServerHandler)
        port = server.server_address[1]
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        class _MockRobot:
            def __init__(self):
                sdk_threads.append(threading.get_ident())
                self.right_arm = _MockSingleOwnerArm(
                    sdk_threads,
                    initial_ee=[0.2586, 0.0322, 0.0332, -0.42, 0.99, 0.96],
                    initial_grip=0.079,
                )
                self.right_arm.fail_home_takeoff_ik = fail_ik
                self.right_arm.fail_brown_transport_ik = fail_brown_transport_ik
                self.right_arm.simulated_block_width_m = simulated_block_width_m
                self.left_arm = _MockSingleOwnerArm(sdk_threads)
                arm_holder.append(self.right_arm)
                self.disconnect_calls = 0
                orig_update = self.right_arm.update_joint_waypoint_chunk_with_gripper

                def _wrapped_update(waypoints, grippers, time_sec, switch_delay_sec=0.05):
                    orig_update(waypoints, grippers, time_sec, switch_delay_sec=switch_delay_sec)
                    if interrupt_on_second_send and len(self.right_arm.sent_chunks) == 2:
                        raise KeyboardInterrupt()

                self.right_arm.update_joint_waypoint_chunk_with_gripper = _wrapped_update

            def safe_stop(self):
                sdk_threads.append(threading.get_ident())

            def disconnect(self):
                sdk_threads.append(threading.get_ident())
                self.disconnect_calls += 1
                self.left_arm.arm.cleanup()
                self.right_arm.arm.cleanup()

        class _MockImageClient:
            def __init__(self, host="127.0.0.1", auto_subscribe=False):
                img_threads.append(threading.get_ident())
                self.closed = 0

            def start_subscriptions(self):
                img_threads.append(threading.get_ident())

            def get_head_frame(self):
                img_threads.append(threading.get_ident())
                return cam_high_bgr.copy(), time.monotonic()

            def get_left_wrist_frame(self):
                img_threads.append(threading.get_ident())
                return _make_wrist_img((0, 0)), time.monotonic()

            def get_right_wrist_frame(self):
                img_threads.append(threading.get_ident())
                return _make_wrist_img((0, 0)), time.monotonic()

            def close(self):
                img_threads.append(threading.get_ident())
                self.closed += 1

        try:
            with tempfile.TemporaryDirectory() as tmp:
                audit_path = Path(tmp) / "audit.json"
                snap_dir = Path(tmp) / "snaps"
                argv = [
                    "--server", f"http://127.0.0.1:{port}",
                    "--confirm-startup-home",
                    "--unsafe-exit-action", "confirm_before_disable",
                    "--lifecycle-audit-json", str(audit_path),
                    "--expected-seed-mode", "fixed",
                    "--target-color", "yellow",
                    "--hover-z-m", "0.200",
                    "--send-right-chunks",
                    "--settled-chunks",
                    "--adaptive-refine",
                    "--rate", "250.0",
                    "--phase-speeds",
                    "--takeoff-speed", "1.0",
                    "--approach-speed", "1.0",
                    "--settle-margin-s", "0.02",
                    "--settle-drift-sleep-s", "0.04",
                    "--exec-steps", "50",
                    "--switch-after", "20",
                    "--blend-overlap", "10",
                    "--max-initial-gap-mm", "25.0",
                    "--max-observation-age-ms", "800.0",
                    "--snapshot-dir", str(snap_dir),
                    "--switch-delay", "0.01",
                    *argv_overrides,
                ]
                import io
                from contextlib import redirect_stdout
                buf = io.StringIO()
                raised = None
                try:
                    with redirect_stdout(buf):
                        guarded.main(
                            argv,
                            robot_cls=_MockRobot,
                            image_client_cls=_MockImageClient,
                            hold_confirm_fn=lambda *_a, **_kw: None,
                        )
                except BaseException as exc:
                    raised = exc
                stdout = buf.getvalue()
                audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
                snap_files = sorted(p.name for p in snap_dir.iterdir()) if snap_dir.exists() else []
                return {
                    "stdout": stdout,
                    "audit": audit,
                    "snap_files": snap_files,
                    "predict_seqs": list(predict_seqs),
                    "sdk_threads": set(sdk_threads),
                    "img_threads": set(img_threads),
                    "main_tid": main_tid,
                    "raised": raised,
                }
        finally:
            server.shutdown()
            server.server_close()

    def test_end_to_end_main_option_a_half_settle_async_prefetch_single_owner(self):
        res = self._run_main_with_mock_env(
            cam_high_bgr=_make_cam_high_with_yellow_block("R2"),
            argv_overrides=("--condition-final-grasp", "--hold-grasp-pose", "--grasp-duration-s", "1.5", "--loops", "4"),
            predict_delay_s=0.02,
            close_from_seq=2,
        )
        self.assertIsNone(res["raised"], msg=res["stdout"])
        self.assertIn("[HOME TAKEOFF 0]", res["stdout"])
        self.assertIn("phase=takeoff", res["stdout"])
        self.assertIn("[SETTLE] joint drift over 40 ms", res["stdout"])
        self.assertIn("[ASYNC PREFETCH 1] hit", res["stdout"])
        self.assertIn("[ASYNC PREFETCH 2] hit", res["stdout"])
        self.assertIn("[ASYNC PREFETCH 3] hit", res["stdout"])
        self.assertIn("[APPROACH TRIM]", res["stdout"])
        self.assertIn("[GRASP COMPLETED]", res["stdout"])
        # Chunk 0 skipped /predict via Option A; Chunks 1..3 consumed prefetched seqs=[1, 2, 3]!
        self.assertEqual(res["predict_seqs"], [1, 2, 3])
        # Strict single-owner invariant: ZERO off-thread calls to robot or ImageClient
        self.assertEqual(res["sdk_threads"], {res["main_tid"]})
        self.assertEqual(res["img_threads"], {res["main_tid"]})
        # Async snapshot writer flushed cleanly
        self.assertIn("loop_00_raw_actions.npy", res["snap_files"])
        self.assertIn("loop_01_conditioned_approach.npy", res["snap_files"])
        self.assertIn("loop_03_conditioned_grasp.npy", res["snap_files"])
        self.assertEqual(res["audit"]["exit_reason"], "normal_completion")

    def test_main_home_takeoff_fallback_on_missing_cam_high_and_ik_failure(self):
        # Case A: blank cam_high -> no target -> raw SmolVLA Chunk 0 /predict called (seqs=[0, 1])
        blank = np.full((480, 640, 3), 235, dtype=np.uint8)
        res_blank = self._run_main_with_mock_env(
            cam_high_bgr=blank,
            argv_overrides=("--loops", "2"),
        )
        self.assertIsNone(res_blank["raised"], msg=res_blank["stdout"])
        self.assertNotIn("[HOME TAKEOFF 0]", res_blank["stdout"])
        self.assertEqual(res_blank["predict_seqs"], [0, 1])

        # Case B: cam_high finds R2, Option A IK fails -> fails closed back to raw SmolVLA Chunk 0
        res_ik = self._run_main_with_mock_env(
            cam_high_bgr=_make_cam_high_with_yellow_block("R2"),
            fail_ik=True,
            argv_overrides=("--loops", "2"),
        )
        self.assertIn("[HOME TAKEOFF FALLBACK]", res_ik["stdout"])
        self.assertGreaterEqual(len(res_ik["predict_seqs"]), 1)

    def test_main_keyboard_interrupt_immediate_teardown_with_async_active(self):
        res = self._run_main_with_mock_env(
            cam_high_bgr=_make_cam_high_with_yellow_block("R2"),
            argv_overrides=("--loops", "3"),
            interrupt_on_second_send=True,
        )
        self.assertIsInstance(res["raised"], KeyboardInterrupt)
        self.assertEqual(res["audit"]["exit_reason"], "keyboard_interrupt")
        self.assertFalse(res["audit"]["home_called"])
        self.assertTrue(res["audit"]["motors_disabled_on_exit"])
        self.assertEqual(res["sdk_threads"], {res["main_tid"]})
        self.assertEqual(res["img_threads"], {res["main_tid"]})

    def test_condition_direct_transport_actions_geometry_and_guards(self):
        self.assertFalse(cases.is_in_brown_place_hover_zone([0.22, 0.08, 0.16, -0.42, 1.0, 0.99]))
        self.assertTrue(cases.is_in_brown_place_hover_zone(cases.BROWN_TARGET_HOVER_POSE6))

        start = np.array([0.220, 0.085, 0.162, -0.42, 1.00, 0.98, 0.032], dtype=float)
        traj = cases.condition_direct_transport_actions(start, held_grip=0.032)
        self.assertGreaterEqual(len(traj), 15)
        self.assertLessEqual(len(traj), 64)
        self.assertTrue(np.isfinite(traj).all())
        self.assertGreaterEqual(float(np.max(traj[:, 2])), 0.208 - 1e-6)
        # First point that leaves XY start must already be at clearance height >= 0.208m
        moved_xy = np.flatnonzero(np.linalg.norm(traj[:, :2] - start[:2], axis=1) > 0.002)
        self.assertGreater(len(moved_xy), 0)
        self.assertGreaterEqual(float(traj[moved_xy[0] - 1, 2]), 0.208 - 1e-6)
        self.assertGreaterEqual(float(traj[moved_xy[0], 2]), 0.208 - 1e-6)
        # Final pose matches BROWN_TARGET_HOVER_POSE6 and holds stationary tail
        self.assertLessEqual(float(np.linalg.norm(traj[-1, :3] - cases.BROWN_TARGET_HOVER_POSE6[:3])), 1e-6)
        self.assertTrue(np.allclose(traj[-3:], traj[-1], atol=1e-9))
        # Continuous Cartesian/rotation/gripper guard bounds
        full = np.vstack([start[None, :], traj])
        self.assertLessEqual(float(np.max(np.linalg.norm(np.diff(full[:, :3], axis=0), axis=1))), 0.015 + 1e-6)
        drot = (np.diff(full[:, 3:6], axis=0) + np.pi) % (2.0 * np.pi) - np.pi
        self.assertLessEqual(float(np.max(np.abs(drot))), np.radians(12.0) + 1e-6)
        self.assertLessEqual(float(np.max(np.abs(np.diff(full[:, 6])))), 0.015 + 1e-6)

        with self.assertRaises(ValueError):
            cases.condition_direct_transport_actions(np.r_[start[:6], 0.002], held_grip=0.002)
        with self.assertRaises(ValueError):
            cases.condition_direct_transport_actions(np.full(7, np.nan), held_grip=0.03)

    def test_condition_placement_actions_trim_release_tail_preserves_legacy_default(self):
        start = np.array([0.348, 0.216, 0.193, -0.45, 0.90, 0.94, 0.025], dtype=float)
        raw = _make_smooth_actions(start, target_xyz=(0.352, 0.218, 0.172), n=50)
        raw[:22, 6] = 0.025
        # Simulate slow SmolVLA creep from 0.045 -> 0.080 over 28 steps
        raw[22:, 6] = np.linspace(0.045, 0.080, 28)

        legacy_seg, _, _ = cases.condition_placement_actions(start, raw, held_grip=0.032)
        trimmed_cand, _, _ = cases.condition_placement_actions(
            start, raw, held_grip=0.032, trim_release_tail=True,
        )
        # Legacy default preserves the gradual raw release curve
        self.assertLess(float(legacy_seg[27, 6]), 0.060)
        # Optimized candidate opens decisively at <=12mm/step then holds constant target_open
        self.assertLessEqual(float(np.max(np.abs(np.diff(trimmed_cand[:, 6])))), 0.0125)
        self.assertGreaterEqual(float(trimmed_cand[-1, 6]), 0.079)
        self.assertTrue(np.allclose(trimmed_cand[-10:], trimmed_cand[-1], atol=1e-9))

        final_seg, removed = cases.trim_segment_idle_tail(trimmed_cand, keep_tail_hold=4, min_points=14)
        self.assertGreaterEqual(removed, 10)
        self.assertTrue(np.allclose(final_seg[-4:], trimmed_cand[-1], atol=1e-9))

    def test_aerial_approach_phase_speed_1p5x_vs_descend_approach(self):
        from types import SimpleNamespace
        ns = SimpleNamespace(
            phase_speeds=True, rate=30.0, speed=0.25,
            takeoff_speed=0.30, aerial_approach_speed=0.36, approach_speed=0.24,
            transport_speed=0.30, place_speed=0.28, grasp_duration_s=2.0, hold_grasp_pose=True,
        )
        spd_aerial, dur_aerial = guarded.get_phase_speed_and_duration("aerial_approach", 36, ns)
        spd_app, dur_app = guarded.get_phase_speed_and_duration("approach", 36, ns)
        self.assertAlmostEqual(spd_aerial, 0.36)
        self.assertAlmostEqual(spd_app, 0.24)
        self.assertAlmostEqual(dur_aerial * 1.5, dur_app)

        # Fallback compatibility when SimpleNamespace omits aerial_approach_speed
        ns_minimal = SimpleNamespace(
            phase_speeds=True, rate=30.0, speed=0.25, approach_speed=0.20,
        )
        spd_fb, _ = guarded.get_phase_speed_and_duration("aerial_approach", 30, ns_minimal)
        self.assertAlmostEqual(spd_fb, 0.30)

    def test_end_to_end_main_direct_transport_visual_place_and_fallback(self):
        r2_hover = cases.GRID_HOVER_POSES["R2"]
        brown = cases.BROWN_TARGET_HOVER_POSE6

        def _pick_and_place_actions(seq_i: int, start_pose7: np.ndarray) -> np.ndarray:
            if float(start_pose7[0]) >= 0.305 and float(start_pose7[1]) >= 0.185:
                # At brown hover after Direct Transport -> Placement release chunk
                seg = _make_smooth_actions(
                    start_pose7,
                    target_xyz=(brown[0] + 0.005, brown[1] + 0.004, brown[2] - 0.020),
                    n=50,
                )
                seg[:20, 6] = start_pose7[6]
                seg[20:, 6] = np.linspace(0.045, 0.080, 30)
                return seg
            if float(start_pose7[2]) <= 0.175:
                # Already descended to grasp height -> close gripper on block
                seg = _make_smooth_actions(start_pose7, target_xyz=(r2_hover[0], r2_hover[1], 0.155), n=50)
                seg[20:, 6] = 0.028
                return seg
            if seq_i <= 2:
                # Aerial approach wholly >= 0.195m (>= 0.188m -> phase=aerial_approach)
                seg = _make_smooth_actions(start_pose7, target_xyz=(r2_hover[0], r2_hover[1], 0.195), n=35)
                return np.vstack([seg, np.tile(seg[-1], (15, 1))])
            # Hover-height chunk with z_last < 0.188m -> deferred vertical descent to table (phase=approach)
            seg = _make_smooth_actions(start_pose7, target_xyz=(r2_hover[0], r2_hover[1], 0.165), n=50)
            seg[20:, 6] = 0.028
            return seg

        res = self._run_main_with_mock_env(
            cam_high_bgr=_make_cam_high_with_yellow_block("R2"),
            simulated_block_width_m=0.032,
            argv_overrides=(
                "--condition-final-grasp", "--hold-grasp-pose", "--grasp-duration-s", "1.5",
                "--place-after-grasp", "--loops", "6",
                "--aerial-approach-speed", "0.90", "--approach-speed", "0.60",
                "--transport-speed", "0.90", "--place-speed", "0.80",
            ),
            predict_delay_s=0.01,
            action_fn=_pick_and_place_actions,
        )
        self.assertIsNone(res["raised"], msg=res["stdout"])
        self.assertIn("[HOME TAKEOFF 0]", res["stdout"])
        self.assertIn("phase=aerial_approach, speed=0.90", res["stdout"])
        self.assertIn("phase=approach, speed=0.60", res["stdout"])
        self.assertIn("[GRASP COMPLETED] chunk=4", res["stdout"])
        self.assertIn("[DIRECT TRANSPORT 5]", res["stdout"])
        self.assertIn("mode=direct_transport", res["stdout"])
        self.assertIn("[ASYNC PREFETCH 6] hit", res["stdout"])
        self.assertIn("[PLACEMENT TRIM]", res["stdout"])
        self.assertIn("[PLACEMENT CONFIRMED] chunk=6 confirmed=True", res["stdout"])
        self.assertIn("[PLACEMENT COMPLETED] chunk=6", res["stdout"])
        # Chunk 0 skipped /predict via Option A and Chunk 5 skipped /predict via direct transport
        self.assertEqual(res["predict_seqs"], [1, 2, 3, 4, 6])
        self.assertEqual(res["audit"]["exit_reason"], "normal_completion")
        self.assertEqual(res["sdk_threads"], {res["main_tid"]})
        self.assertEqual(res["img_threads"], {res["main_tid"]})

        # Fail-closed fallback when direct transport IK is rejected at clearance height
        res_fb = self._run_main_with_mock_env(
            cam_high_bgr=_make_cam_high_with_yellow_block("R2"),
            fail_brown_transport_ik=True,
            simulated_block_width_m=0.032,
            argv_overrides=(
                "--condition-final-grasp", "--hold-grasp-pose", "--grasp-duration-s", "1.5",
                "--place-after-grasp", "--loops", "6",
                "--aerial-approach-speed", "0.90", "--approach-speed", "0.60",
            ),
            predict_delay_s=0.01,
            action_fn=_pick_and_place_actions,
        )
        self.assertIn("[DIRECT TRANSPORT FALLBACK]", res_fb["stdout"])
        # Because fallback invoked SmolVLA /predict on Chunk 5, seq 5 was still requested
        self.assertGreaterEqual(len(res_fb["predict_seqs"]), 5)


if __name__ == "__main__":
    unittest.main()
