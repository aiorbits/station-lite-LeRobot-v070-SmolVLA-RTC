#!/usr/bin/env python3
"""Offline unit tests for G4 execution adapter, lifecycle safety guards, and SDK dry-run gate."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import g4_execution_adapter_gate as g4_exec  # noqa: E402
import g4_sdk_dryrun_gate as g4_sdk  # noqa: E402
import station_lite_smolvla_right7_guarded as guarded  # noqa: E402


class TestG4CandidateGuardsAndLifecycle(unittest.TestCase):
    def test_validate_predict_response_identity_and_shape(self):
        good_actions = np.zeros((50, 7), dtype=float).tolist()
        valid = {
            "schema": guarded.SCHEMA,
            "session_id": "sess-1",
            "seq": 3,
            "action_dim": 7,
            "hardware_connected": False,
            "sampling_seed": 42,
            "actions": good_actions,
        }
        arr = guarded.validate_predict_response(
            valid,
            expected_session_id="sess-1",
            expected_seq=3,
            expected_seed_mode="fixed",
            expected_seed_base=42,
        )
        self.assertEqual(arr.shape, (50, 7))

        for bad_patch in (
            {"session_id": "wrong"},
            {"seq": 99},
            {"sampling_seed": 7},
            {"hardware_connected": True},
            {"action_dim": 6},
            {"actions": np.zeros((49, 7)).tolist()},
            {"actions": [[float("nan")] * 7] * 50},
        ):
            bad = {**valid, **bad_patch}
            with self.assertRaises(ValueError):
                guarded.validate_predict_response(
                    bad,
                    expected_session_id="sess-1",
                    expected_seq=3,
                    expected_seed_mode="fixed",
                    expected_seed_base=42,
                )

    def test_validate_post_smoothing_waypoints_enforces_limits_steps_and_endpoints(self):
        mid = 0.5 * (guarded.JOINT_LOW + guarded.JOINT_UP)
        q0 = mid - 0.02
        q1 = mid + 0.03
        raw = np.linspace(q0, q1, 10)
        smoothed = guarded.smooth_joint_waypoints(raw, alpha=0.20, passes=1)
        stats = guarded.validate_post_smoothing_waypoints(
            smoothed, raw_waypoints=raw, margin=0.02, max_soft_clip_rad=0.0, max_joint_step_rad=0.08
        )
        self.assertTrue(stats["physical_limits_ok"])
        self.assertTrue(stats["soft_limits_ok"])
        self.assertLessEqual(stats["endpoint_diff_rad"], 1e-9)

        # Endpoint drift rejected
        perturbed = smoothed.copy()
        perturbed[0, 0] += 1e-3
        with self.assertRaises(ValueError):
            guarded.validate_post_smoothing_waypoints(perturbed, raw_waypoints=raw)

        # Step bound exceeded rejected
        steep = raw.copy()
        steep[4, 0] += 0.12
        with self.assertRaises(ValueError):
            guarded.validate_post_smoothing_waypoints(steep, max_joint_step_rad=0.08)

        # Soft margin breach rejected when max_soft_clip_rad == 0
        near_soft = np.tile(mid, (5, 1))
        near_soft[:, 1] = guarded.JOINT_LOW[1] + 0.015
        with self.assertRaises(ValueError):
            guarded.validate_post_smoothing_waypoints(near_soft, margin=0.02, max_soft_clip_rad=0.0)

        # Physical limit breach rejected
        over_phys = np.tile(guarded.JOINT_UP + 0.005, (5, 1))
        with self.assertRaises(ValueError):
            guarded.validate_post_smoothing_waypoints(over_phys)

    def test_evaluate_exit_home_safety_fail_closed_matrix(self):
        ok = guarded.evaluate_exit_home_safety(
            policy="normal_if_safe_and_confirmed",
            exit_reason="normal_completion",
            confirm_exit_home=True,
            holding_object=False,
            ee_z_m=0.20,
            min_safe_home_z_m=0.175,
        )
        self.assertTrue(ok["home_permitted"])
        self.assertEqual(ok["suppress_reasons"], [])

        cases = [
            ({"policy": "never"}, "policy_never"),
            ({"exit_reason": "guard_rejected"}, "abort_guard_rejected"),
            ({"exit_reason": "exception"}, "abort_exception"),
            ({"exit_reason": "keyboard_interrupt"}, "abort_keyboard_interrupt"),
            ({"confirm_exit_home": False}, "exit_home_not_confirmed"),
            ({"holding_object": True}, "holding_object_or_unreleased_grasp"),
            ({"ee_z_m": 0.160}, "near_table_contact_low_z"),
            ({"ee_z_m": None}, "unknown_ee_z"),
        ]
        for patch, expected_reason in cases:
            kw = {
                "policy": "normal_if_safe_and_confirmed",
                "exit_reason": "normal_completion",
                "confirm_exit_home": True,
                "holding_object": False,
                "ee_z_m": 0.20,
                "min_safe_home_z_m": 0.175,
                **patch,
            }
            dec = guarded.evaluate_exit_home_safety(**kw)
            self.assertFalse(dec["home_permitted"])
            self.assertIn(expected_reason, dec["suppress_reasons"])

    def test_execute_candidate_finally_interrupt_and_held_object_suppression(self):
        class _Arm:
            def __init__(self, grip=0.068, ee_z=0.200, with_hold=False):
                self._grip = grip
                self._ee_z = ee_z
                self.hold_calls = 0
                if with_hold:
                    self.hold = self._hold

            def _hold(self):
                self.hold_calls += 1

            def get_gripper_distance(self):
                return self._grip

            def get_joint_positions(self):
                return [0.0] * 6

            def get_ee_pose_euler(self):
                return [0.15, 0.0, self._ee_z], [0.0, 0.0, 0.0]

        class _Robot:
            def __init__(self, *, grip=0.068, ee_z=0.200, with_hold=False, with_cleanup=True):
                self.right_arm = _Arm(grip=grip, ee_z=ee_z, with_hold=with_hold)
                self.home_calls = 0
                self.cleanup_calls = 0
                if with_cleanup:
                    self.cleanup = self._cleanup

            def go_home(self):
                self.home_calls += 1

            def _cleanup(self):
                self.cleanup_calls += 1

        class _Img:
            def __init__(self):
                self.closed = 0

            def close(self):
                self.closed += 1

        class _Trace:
            def __init__(self):
                self.stopped_chunks = None

            def stop(self, sent_chunks):
                self.stopped_chunks = sent_chunks
                return 12

        # 1. KeyboardInterrupt skips chunk sleep, invokes hold(), suppresses go_home()
        slept = []
        robot = _Robot(grip=0.068, ee_z=0.200, with_hold=True, with_cleanup=True)
        img = _Img()
        trace = _Trace()
        with tempfile.TemporaryDirectory() as td:
            audit_file = Path(td) / "interrupt.json"
            audit = guarded.execute_candidate_finally(
                robot=robot,
                image_client=img,
                trace_session=trace,
                sent_chunks=2,
                last_install=10.0,
                last_duration=8.0,
                switch_delay=0.05,
                exit_reason="keyboard_interrupt",
                policy="normal_if_safe_and_confirmed",
                confirm_exit_home=True,
                holding_object=False,
                min_safe_home_z_m=0.175,
                unsafe_exit_action="immediate_disable",
                lifecycle_audit_json=str(audit_file),
                monotonic_fn=lambda: 10.5,
                sleep_fn=slept.append,
            )
            self.assertEqual(slept, [])
            self.assertTrue(audit["skipped_wait_for_interrupt"])
            self.assertFalse(audit["home_called"])
            self.assertEqual(robot.home_calls, 0)
            self.assertEqual(robot.right_arm.hold_calls, 1)
            self.assertTrue(audit["safe_hold_called"])
            self.assertEqual(robot.cleanup_calls, 1)
            self.assertEqual(img.closed, 1)
            self.assertEqual(trace.stopped_chunks, 2)
            self.assertTrue(audit_file.exists())

        # 2. Closed measured gripper (< 0.045m) with sent_chunks > 0 suppresses go_home even if holding_object=False
        robot_closed = _Robot(grip=0.029, ee_z=0.210, with_hold=False, with_cleanup=False)
        img2 = _Img()
        audit2 = guarded.execute_candidate_finally(
            robot=robot_closed,
            image_client=img2,
            trace_session=None,
            sent_chunks=1,
            last_install=10.0,
            last_duration=1.0,
            switch_delay=0.05,
            exit_reason="normal_completion",
            policy="normal_if_safe_and_confirmed",
            confirm_exit_home=True,
            holding_object=False,
            min_safe_home_z_m=0.175,
            unsafe_exit_action="immediate_disable",
            monotonic_fn=lambda: 12.0,
            sleep_fn=slept.append,
        )
        self.assertFalse(audit2["home_called"])
        self.assertIn("holding_object_or_unreleased_grasp", audit2["home_decision"]["suppress_reasons"])
        self.assertFalse(audit2["robot_cleanup_callable"])
        self.assertFalse(audit2["safe_hold_verified"])
        self.assertEqual(img2.closed, 1)

        # 2b. Non-TTY confirm_before_disable without hold_confirm_fn records skip reason without hanging
        class _NonTtyStdin:
            def isatty(self):
                return False

        robot_nontty = _Robot(grip=0.029, ee_z=0.210, with_hold=False, with_cleanup=True)
        img2b = _Img()
        orig_stdin = guarded.sys.stdin
        guarded.sys.stdin = _NonTtyStdin()
        try:
            audit2b = guarded.execute_candidate_finally(
                robot=robot_nontty,
                image_client=img2b,
                trace_session=None,
                sent_chunks=1,
                last_install=10.0,
                last_duration=1.0,
                switch_delay=0.05,
                exit_reason="normal_completion",
                policy="normal_if_safe_and_confirmed",
                confirm_exit_home=True,
                holding_object=False,
                min_safe_home_z_m=0.175,
                unsafe_exit_action="confirm_before_disable",
                monotonic_fn=lambda: 12.0,
                sleep_fn=slept.append,
            )
        finally:
            guarded.sys.stdin = orig_stdin
        self.assertFalse(audit2b["home_called"])
        self.assertEqual(audit2b["hold_pause_skipped_reason"], "non_tty_without_confirm_fn")
        self.assertEqual(robot_nontty.cleanup_calls, 1)
        self.assertTrue(audit2b["motors_disabled_on_exit"])

        # 3. Realistic DualTouchRobot shape (disconnect/safe_stop + arm.cleanup, no hold/stop/robot.cleanup)
        class _DualTouchShape(_Robot):
            def __init__(self):
                super().__init__(grip=0.029, ee_z=0.160, with_hold=False, with_cleanup=False)
                self.disconnect_calls = 0
                self.safe_stop_calls = 0
                self.left_arm_cleanup_calls = 0
                self.right_arm_cleanup_calls = 0
                if getattr(self, "left_arm", None) is None:
                    self.left_arm = type("_ArmStub", (), {})()
                if getattr(self, "right_arm", None) is None:
                    self.right_arm = type("_ArmStub", (), {})()

                def _left_cleanup():
                    self.left_arm_cleanup_calls += 1

                def _right_cleanup():
                    self.right_arm_cleanup_calls += 1

                self.left_arm.cleanup = _left_cleanup
                self.right_arm.cleanup = _right_cleanup

            def safe_stop(self):
                self.safe_stop_calls += 1

            def disconnect(self):
                self.disconnect_calls += 1
                self.left_arm.cleanup()
                self.right_arm.cleanup()

        robot_dual = _DualTouchShape()
        img3 = _Img()
        disconnect_at_confirm = []
        img_closed_at_confirm = []

        def _confirm_hold(*_args, **_kwargs):
            disconnect_at_confirm.append(robot_dual.disconnect_calls)
            img_closed_at_confirm.append(img3.closed)

        audit3 = guarded.execute_candidate_finally(
            robot=robot_dual,
            image_client=img3,
            trace_session=None,
            sent_chunks=1,
            last_install=10.0,
            last_duration=1.0,
            switch_delay=0.05,
            exit_reason="keyboard_interrupt",
            policy="normal_if_safe_and_confirmed",
            confirm_exit_home=True,
            holding_object=False,
            min_safe_home_z_m=0.175,
            unsafe_exit_action="confirm_before_disable",
            hold_confirm_fn=_confirm_hold,
            monotonic_fn=lambda: 10.5,
            sleep_fn=slept.append,
        )
        self.assertFalse(audit3["home_called"])
        self.assertFalse(audit3["has_motor_hold_or_stop_api"])
        self.assertFalse(audit3["safe_hold_called"])
        self.assertFalse(audit3["safe_hold_verified"])
        self.assertTrue(audit3["hold_pause_requested"])
        self.assertTrue(audit3["hold_pause_kept_motors_enabled"])
        self.assertTrue(audit3["hold_pause_confirmed"])
        self.assertFalse(audit3["hold_pause_interrupted"])
        self.assertEqual(disconnect_at_confirm, [0])
        self.assertEqual(img_closed_at_confirm, [1])
        self.assertEqual(robot_dual.disconnect_calls, 1)
        self.assertTrue(audit3["robot_disconnect_callable"])
        self.assertTrue(audit3["robot_disconnect_called"])
        self.assertFalse(audit3["robot_cleanup_callable"])
        self.assertTrue(audit3["motors_disabled_on_exit"])

        # 3b. Second KeyboardInterrupt inside hold_confirm_fn still tears down cleanly
        robot_dual2 = _DualTouchShape()
        img4 = _Img()

        def _interrupt_hold(*_args, **_kwargs):
            raise KeyboardInterrupt

        audit4 = guarded.execute_candidate_finally(
            robot=robot_dual2,
            image_client=img4,
            trace_session=None,
            sent_chunks=1,
            last_install=10.0,
            last_duration=1.0,
            switch_delay=0.05,
            exit_reason="keyboard_interrupt",
            policy="normal_if_safe_and_confirmed",
            confirm_exit_home=True,
            holding_object=False,
            min_safe_home_z_m=0.175,
            unsafe_exit_action="confirm_before_disable",
            hold_confirm_fn=_interrupt_hold,
            monotonic_fn=lambda: 10.5,
            sleep_fn=slept.append,
        )
        self.assertTrue(audit4["hold_pause_requested"])
        self.assertTrue(audit4["hold_pause_interrupted"])
        self.assertFalse(audit4["hold_pause_confirmed"])
        self.assertEqual(robot_dual2.disconnect_calls, 1)
        self.assertTrue(audit4["motors_disabled_on_exit"])

    def test_post_placement_low_z_guarded_lift_before_exit_home(self):
        class _InnerArm:
            def __init__(self, *, bad_plan=False):
                self.bad_plan = bad_plan
                self.plan_calls = 0

            def plan_joint_waypoints_with_gripper(self, start_q, wp, grip, time_sec=1.0):
                self.plan_calls += 1
                if self.bad_plan:
                    return np.zeros((1, 15), dtype=float)
                arr = np.zeros((10, 15), dtype=float)
                arr[:, 0] = np.linspace(0.0, float(time_sec), 10)
                arr[:, 1:7] = np.asarray(start_q, dtype=float)
                return arr

        class _LiftArm:
            def __init__(self, *, ee_z=0.1465, grip=0.0643, lift_final_z=0.2000, fail_ik=False, bad_plan=False):
                self.arm = _InnerArm(bad_plan=bad_plan)
                self.ee_z = ee_z
                self.grip = grip
                self.lift_final_z = lift_final_z
                self.fail_ik = fail_ik
                self.q = [0.0, 0.2, -0.3, 0.1, 0.0, 0.0]
                self.lift_calls = 0

            def get_gripper_distance(self):
                return self.grip

            def get_joint_positions(self):
                return list(self.q)

            def get_ee_pose_euler(self):
                return [0.3452, 0.2321, self.ee_z], [0.0, 0.0, 0.0]

            def solve_ik_euler(self, pos, euler, seed):
                if self.fail_ik:
                    return [0.0] * 6, False
                frac = max(0.0, min(1.0, (float(pos[2]) - 0.1465) / 0.0535))
                return [0.0, 0.2 + 0.05 * frac, -0.3, 0.1, 0.0, 0.0], True

            def update_joint_waypoint_chunk_with_gripper(self, wp, grip, time_sec=1.0, switch_delay_sec=0.05):
                self.lift_calls += 1
                self.q = list(wp[-1])
                self.ee_z = self.lift_final_z
                self.grip = float(grip[-1]) * guarded.GRIPPER_MAX_M

        class _LiftRobot:
            def __init__(self, arm):
                self.right_arm = arm
                self.left_arm = type("_Left", (), {"cleanup": lambda self: None})()
                self.home_calls = 0
                self.disconnect_calls = 0

            def go_home(self):
                self.home_calls += 1

            def disconnect(self):
                self.disconnect_calls += 1

        class _Img:
            def __init__(self):
                self.closed = 0

            def close(self):
                self.closed += 1

        # (a) Confirmed placement at ee_z=0.1465m vertically lifts to 0.2000m then calls go_home()
        slept = []
        hold_calls = []
        robot_ok = _LiftRobot(_LiftArm(ee_z=0.1465, grip=0.0643, lift_final_z=0.2000))
        audit_ok = guarded.execute_candidate_finally(
            robot=robot_ok,
            image_client=_Img(),
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
            hold_confirm_fn=hold_calls.append,
            monotonic_fn=lambda: 15.0,
            sleep_fn=slept.append,
        )
        self.assertTrue(audit_ok["post_placement_lift_attempted"])
        self.assertTrue(audit_ok["post_placement_lift_succeeded"])
        self.assertIsNone(audit_ok["post_placement_lift_error"])
        self.assertEqual(robot_ok.right_arm.lift_calls, 1)
        self.assertEqual(robot_ok.right_arm.arm.plan_calls, 1)
        self.assertEqual(robot_ok.home_calls, 1)
        self.assertTrue(audit_ok["home_called"])
        self.assertFalse(audit_ok["hold_pause_requested"])
        self.assertEqual(hold_calls, [])
        self.assertEqual(robot_ok.disconnect_calls, 1)
        self.assertGreaterEqual(audit_ok["measured_ee_z_m"], 0.200)

        # (b) Non-placement run at low Z NEVER auto-lifts and stays held until operator confirm
        hold_nonplace = []
        robot_nonplace = _LiftRobot(_LiftArm(ee_z=0.1465, grip=0.0643))
        audit_nonplace = guarded.execute_candidate_finally(
            robot=robot_nonplace,
            image_client=_Img(),
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
            placement_completed=False,
            unsafe_exit_action="confirm_before_disable",
            hold_confirm_fn=hold_nonplace.append,
            monotonic_fn=lambda: 15.0,
            sleep_fn=slept.append,
        )
        self.assertFalse(audit_nonplace["post_placement_lift_attempted"])
        self.assertEqual(robot_nonplace.right_arm.lift_calls, 0)
        self.assertEqual(robot_nonplace.home_calls, 0)
        self.assertTrue(audit_nonplace["hold_pause_requested"])
        self.assertEqual(len(hold_nonplace), 1)
        self.assertEqual(audit_nonplace["home_decision"]["suppress_reasons"], ["near_table_contact_low_z"])

        # (c) IK failure during lift refuses motion, keeps Home suppressed, enters hold pause
        hold_ikfail = []
        robot_ikfail = _LiftRobot(_LiftArm(ee_z=0.1465, grip=0.0643, fail_ik=True))
        audit_ikfail = guarded.execute_candidate_finally(
            robot=robot_ikfail,
            image_client=_Img(),
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
            hold_confirm_fn=hold_ikfail.append,
            monotonic_fn=lambda: 15.0,
            sleep_fn=slept.append,
        )
        self.assertTrue(audit_ikfail["post_placement_lift_attempted"])
        self.assertFalse(audit_ikfail["post_placement_lift_succeeded"])
        self.assertIsNotNone(audit_ikfail["post_placement_lift_error"])
        self.assertEqual(robot_ikfail.right_arm.lift_calls, 0)
        self.assertEqual(robot_ikfail.home_calls, 0)
        self.assertEqual(len(hold_ikfail), 1)

        # (d) Abnormally low Z (<0.120m) refuses lift and enters hold pause
        hold_abn = []
        robot_abn = _LiftRobot(_LiftArm(ee_z=0.110, grip=0.0643))
        audit_abn = guarded.execute_candidate_finally(
            robot=robot_abn,
            image_client=_Img(),
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
            hold_confirm_fn=hold_abn.append,
            monotonic_fn=lambda: 15.0,
            sleep_fn=slept.append,
        )
        self.assertTrue(audit_abn["post_placement_lift_attempted"])
        self.assertFalse(audit_abn["post_placement_lift_succeeded"])
        self.assertIn("0.120m", audit_abn["post_placement_lift_error"])
        self.assertEqual(robot_abn.right_arm.lift_calls, 0)
        self.assertEqual(robot_abn.home_calls, 0)
        self.assertEqual(len(hold_abn), 1)

        # (e) Lift executes but resulting ee_z still < min_safe_home_z_m -> still suppresses Home
        hold_short = []
        robot_short = _LiftRobot(_LiftArm(ee_z=0.1465, grip=0.0643, lift_final_z=0.165))
        audit_short = guarded.execute_candidate_finally(
            robot=robot_short,
            image_client=_Img(),
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
            hold_confirm_fn=hold_short.append,
            monotonic_fn=lambda: 15.0,
            sleep_fn=slept.append,
        )
        self.assertTrue(audit_short["post_placement_lift_attempted"])
        self.assertFalse(audit_short["post_placement_lift_succeeded"])
        self.assertEqual(robot_short.right_arm.lift_calls, 1)
        self.assertEqual(robot_short.home_calls, 0)
        self.assertEqual(len(hold_short), 1)
        self.assertEqual(audit_short["home_decision"]["suppress_reasons"], ["near_table_contact_low_z"])

        # (f) Without --confirm-exit-home, placement_completed does NOT auto-lift or Home
        hold_unconf = []
        robot_unconf = _LiftRobot(_LiftArm(ee_z=0.1465, grip=0.0643))
        audit_unconf = guarded.execute_candidate_finally(
            robot=robot_unconf,
            image_client=_Img(),
            trace_session=None,
            sent_chunks=6,
            last_install=10.0,
            last_duration=2.0,
            switch_delay=0.05,
            exit_reason="normal_completion",
            policy="normal_if_safe_and_confirmed",
            confirm_exit_home=False,
            holding_object=False,
            min_safe_home_z_m=0.175,
            placement_completed=True,
            unsafe_exit_action="confirm_before_disable",
            hold_confirm_fn=hold_unconf.append,
            monotonic_fn=lambda: 15.0,
            sleep_fn=slept.append,
        )
        self.assertFalse(audit_unconf["post_placement_lift_attempted"])
        self.assertEqual(robot_unconf.right_arm.lift_calls, 0)
        self.assertEqual(robot_unconf.home_calls, 0)
        self.assertEqual(len(hold_unconf), 1)

    def test_programmatic_lifecycle_fault_injection_suite(self):
        summary = g4_exec.run_lifecycle_audit_suite()
        self.assertTrue(summary["all_passed"], json.dumps(summary, indent=2))
        self.assertEqual(summary["scenarios_total"], 14)
        self.assertEqual(summary["scenarios_passed"], 14)
        self.assertFalse(summary["robot_safe_hold_hardware_verified"])


class TestG4SdkDryRunGate(unittest.TestCase):
    def test_enforce_loopback_only_net(self):
        self.assertEqual(g4_sdk.enforce_loopback_only_net(["lo"]), ["lo"])
        with self.assertRaises(SystemExit):
            g4_sdk.enforce_loopback_only_net(["lo", "eth0"])
        with self.assertRaises(SystemExit):
            g4_sdk.enforce_loopback_only_net(["lo", "can0"])

    def test_validate_planned_timeline_and_bundle_evaluation(self):
        class _FakeSdkArm:
            def __init__(self, *, fail_ik_at=None, nonmono_plan=False, with_cleanup=True):
                self.fail_ik_at = fail_ik_at
                self.nonmono_plan = nonmono_plan
                self.calls = 0
                self.plan_calls = 0
                if with_cleanup:
                    self.cleanup = lambda: None

            def solve_ik_euler(self, pos, euler, seed):
                idx = self.calls
                self.calls += 1
                if self.fail_ik_at is not None and idx == self.fail_ik_at:
                    return seed, False
                q = np.asarray(seed, dtype=float) + np.array([0.005, 0.004, -0.003, 0.0, 0.0, 0.0])
                return q.tolist(), True

            def plan_joint_waypoints_with_gripper(self, start_q, wp, grip, time_sec=1.0):
                self.plan_calls += 1
                n = max(10, len(wp) * 4)
                arr = np.zeros((n, 15), dtype=float)
                t = np.linspace(0.0, float(time_sec), n)
                if self.nonmono_plan:
                    t[3] = t[2]
                arr[:, 0] = t
                for j in range(6):
                    arr[:, 1 + j] = np.linspace(start_q[j], wp[-1][j], n)
                return arr

            def go_home(self):
                return

        start_action = [0.20, 0.0, 0.22, 0.0, 0.0, 0.0, 0.068]
        actions = []
        for k in range(1, 11):
            actions.append([0.20 + 0.002 * k, 0.0, 0.22, 0.0, 0.0, 0.0, 0.068])

        valid_case = {
            "name": "valid_approach",
            "loop_index": 1,
            "phase": "approach",
            "duration_s": 1.67,
            "margin_rad": 0.02,
            "smooth_waypoints": True,
            "waypoint_smooth_alpha": 0.20,
            "waypoint_smooth_passes": 1,
            "max_interior_joint_step_rad": 0.07,
            "seed": [0.0, 0.1, -0.2, 0.0, 0.0, 0.0],
            "start_action": start_action,
            "actions": actions,
        }
        bundle = {"schema": g4_sdk.BUNDLE_SCHEMA, "cases": [valid_case]}
        arm_ok = _FakeSdkArm(with_cleanup=True)
        rep_ok = g4_sdk.run_bundle_evaluation(arm_ok, bundle, input_sha256="abc", interfaces=["lo"])
        self.assertTrue(rep_ok["all_passed"])
        self.assertEqual(arm_ok.plan_calls, 1)
        self.assertTrue(rep_ok["sdk_capabilities"]["has_cleanup"])
        self.assertFalse(rep_ok["sdk_capabilities"]["has_hold"])
        self.assertEqual(rep_ok["cases"][0]["planner"], "passed")
        self.assertAlmostEqual(rep_ok["cases"][0]["timeline"]["duration_error_s"], 0.0, places=6)

        arm_fail_ik = _FakeSdkArm(fail_ik_at=3, with_cleanup=False)
        rep_ik = g4_sdk.run_bundle_evaluation(arm_fail_ik, bundle, interfaces=["lo"])
        self.assertFalse(rep_ik["all_passed"])
        self.assertIn("ik", rep_ik["cases"][0]["reject"])
        self.assertEqual(rep_ik["cases"][0]["planner"], "not_run")
        self.assertEqual(arm_fail_ik.plan_calls, 0)
        self.assertFalse(rep_ik["sdk_capabilities"]["has_cleanup"])

        class _StepJumpArm(_FakeSdkArm):
            def solve_ik_euler(self, pos, euler, seed):
                idx = self.calls
                self.calls += 1
                q = np.asarray(seed, dtype=float).copy()
                q[0] += 0.085 if idx == 4 else 0.005
                return q.tolist(), True

        arm_jump = _StepJumpArm()
        rep_jump = g4_sdk.run_bundle_evaluation(arm_jump, bundle, interfaces=["lo"])
        self.assertFalse(rep_jump["all_passed"])
        self.assertIn("refined_joint_step", rep_jump["cases"][0]["reject"])
        self.assertEqual(arm_jump.plan_calls, 0)

        arm_bad_plan = _FakeSdkArm(nonmono_plan=True)
        rep_plan = g4_sdk.run_bundle_evaluation(arm_bad_plan, bundle, interfaces=["lo"])
        self.assertFalse(rep_plan["all_passed"])
        self.assertIn("planner", rep_plan["cases"][0]["reject"])
        self.assertEqual(rep_plan["cases"][0]["planner"], "failed")

        # Loop 0 initial-prefix soft-boundary omission + 16.5 mm raw interior step with solve_ik-only ArmController interface
        class _PrefixArm:
            def solve_ik(self, pos, euler, seed):
                idx = self.calls
                self.calls += 1
                q = np.array([0.0, 0.2, -0.3, 0.1, 0.0, 0.0], dtype=float)
                if idx < 2:
                    q[1] = float(guarded.JOINT_LOW[1] + 0.005)
                else:
                    q[0] += min(0.20, 0.01 * idx)
                return q.tolist(), True

            def __init__(self):
                self.calls = 0

            def plan_joint_waypoints_with_gripper(self, start_q, wp, grip, time_sec=1.0):
                n = max(10, len(wp) * 4)
                arr = np.zeros((n, 15), dtype=float)
                arr[:, 0] = np.linspace(0.0, float(time_sec), n)
                for j in range(6):
                    arr[:, 1 + j] = np.linspace(start_q[j], wp[-1][j], n)
                return arr

        raw_loop0 = [[0.20 + 0.002 * k, 0.0, 0.22, 0.0, 0.0, 0.0, 0.068] for k in range(50)]
        raw_loop0[17][0] += 0.0165
        for k in range(18, 50):
            raw_loop0[k][0] = raw_loop0[17][0] + 0.002 * (k - 17)
        prefix_cand = guarded.make_initial_prefix(np.asarray(start_action), np.asarray(raw_loop0[0]), 6)
        loop0_case = {
            **valid_case,
            "name": "loop0_with_prefix_omission",
            "loop_index": 0,
            "max_interior_joint_step_rad": 0.25,
            "requires_initial_prefix_selection": True,
            "initial_prefix_candidates": prefix_cand.tolist(),
            "seed": [0.0, 0.2, -0.3, 0.1, 0.0, 0.0],
            "actions": raw_loop0,
            "raw_actions": raw_loop0,
        }
        arm_loop0 = _PrefixArm()
        rep_loop0 = g4_sdk.run_bundle_evaluation(arm_loop0, {"schema": g4_sdk.BUNDLE_SCHEMA, "cases": [loop0_case]}, interfaces=["lo"])
        self.assertTrue(rep_loop0["all_passed"], json.dumps(rep_loop0, indent=2))
        self.assertEqual(rep_loop0["cases"][0]["omitted_prefix_candidates"], 2)
        self.assertEqual(rep_loop0["cases"][0]["inserted_prefix_points"], len(prefix_cand) - 2)

    def test_fast_cycle_sdk_bundle_matches_sdk_dryrun_gate_schema(self):
        class _FastCycleMockArm:
            def solve_ik(self, pos, euler, seed):
                q = np.asarray(seed, dtype=float).copy()
                # Nudge within valid soft bounds cleanly proportional to Cartesian target delta
                target = np.clip(
                    np.asarray([pos[0] - 0.12, pos[1] * 0.8, (pos[2] - 0.18) * 0.5, euler[0] * 0.15, euler[1] * 0.15, euler[2] * 0.15]),
                    guarded.JOINT_LOW + 0.05,
                    guarded.JOINT_UP - 0.05,
                )
                delta = np.clip(target - q, -0.02, 0.02)
                return (q + delta).tolist(), True

            def plan_joint_waypoints_with_gripper(self, start_q, wp, grip, time_sec=1.0):
                n = max(12, len(wp) * 4)
                arr = np.zeros((n, 15), dtype=float)
                arr[:, 0] = np.linspace(0.0, float(time_sec), n)
                for j in range(6):
                    arr[:, 1 + j] = np.linspace(start_q[j], wp[-1][j], n)
                return arr

        import build_grasp_replay_cases as cases

        start_home = np.array([0.130, 0.018, 0.205, -0.450, 0.900, 0.940, 0.002], dtype=float)
        takeoff_seg = cases.condition_home_takeoff_actions(
            start_home,
            hover_pose=cases.GRID_HOVER_POSES["R8"],
            hover_z_m=0.200,
        )
        grasp_start = np.array([0.230, 0.140, 0.160, -0.420, 1.020, 0.990, 0.056], dtype=float)
        direct_seg = cases.condition_direct_transport_actions(grasp_start, held_grip=0.058)

        place_start = np.r_[cases.BROWN_TARGET_HOVER_POSE6, 0.058]
        place_raw = np.tile(place_start, (50, 1))
        place_raw[5:25, 0] += np.linspace(0.0, 0.015, 20)
        place_raw[5:25, 2] -= np.linspace(0.0, 0.045, 20)
        place_raw[25:, 0] += 0.015
        place_raw[25:, 2] -= 0.045
        place_raw[28:, 6] = 0.075
        place_seg, _, _ = cases.condition_placement_actions(
            place_start,
            place_raw,
            held_grip=0.058,
            trim_release_tail=True,
        )
        place_seg, _ = cases.trim_segment_idle_tail(place_seg, keep_tail_hold=4, min_points=14)
        self.assertLessEqual(len(place_seg), 50)

        synthetic_bundle = {
            "schema": "v070_g4_sdk_dryrun_bundle_v1",
            "cases": [
                {
                    "name": "synthetic_takeoff_R8",
                    "loop_index": 0,
                    "phase": "takeoff",
                    "duration_s": max(0.8, len(takeoff_seg) / 30.0 / 0.30),
                    "margin_rad": 0.02,
                    "smooth_waypoints": True,
                    "waypoint_smooth_alpha": 0.20,
                    "waypoint_smooth_passes": 1,
                    "max_interior_joint_step_rad": 0.25,
                    "requires_initial_prefix_selection": False,
                    "seed": [0.0, 0.1, -0.2, 0.0, 0.0, 0.0],
                    "start_action": start_home.tolist(),
                    "actions": takeoff_seg.tolist(),
                    "raw_actions": takeoff_seg.tolist(),
                },
                {
                    "name": "synthetic_direct_transport",
                    "phase": "transport",
                    "duration_s": max(0.8, len(direct_seg) / 30.0 / 0.30),
                    "margin_rad": 0.02,
                    "smooth_waypoints": True,
                    "waypoint_smooth_alpha": 0.20,
                    "waypoint_smooth_passes": 1,
                    "max_interior_joint_step_rad": 0.07,
                    "requires_initial_prefix_selection": False,
                    "seed": [0.0, 0.1, -0.2, 0.0, 0.0, 0.0],
                    "start_action": grasp_start.tolist(),
                    "actions": direct_seg.tolist(),
                    "raw_actions": direct_seg.tolist(),
                },
                {
                    "name": "synthetic_place_trimmed",
                    "phase": "place",
                    "duration_s": max(0.8, len(place_seg) / 30.0 / 0.28),
                    "margin_rad": 0.02,
                    "smooth_waypoints": True,
                    "waypoint_smooth_alpha": 0.20,
                    "waypoint_smooth_passes": 1,
                    "max_interior_joint_step_rad": 0.07,
                    "requires_initial_prefix_selection": False,
                    "seed": [0.0, 0.1, -0.2, 0.0, 0.0, 0.0],
                    "start_action": place_start.tolist(),
                    "actions": place_seg.tolist(),
                    "raw_actions": place_seg.tolist(),
                },
            ],
        }
        rep = g4_sdk.run_bundle_evaluation(_FastCycleMockArm(), synthetic_bundle, interfaces=["lo"])
        self.assertTrue(rep["all_passed"], json.dumps(rep, indent=2))
        self.assertEqual(rep["cases_passed"], 3)
        self.assertEqual(rep["cases_total"], 3)


if __name__ == "__main__":
    unittest.main()
