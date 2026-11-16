"""Stage D2 integration tests: SessionService + GuardedCandidateBridge closed-loop."""

from __future__ import annotations

import threading
import time
import unittest

from v070_demo.candidate_bridge import GuardedBridgeConfig, GuardedCandidateBridge
from v070_demo.contracts import CommandStatus, CommandType, ExecutionPhase, SessionState
from v070_demo.session_service import SessionService
from v070_demo.tests.test_d2_perception_evidence_trajectory import (
    make_synthetic_cam_high,
    make_synthetic_wrist_release,
)


BATCH_PROMPT = "把桌上的红、黄、蓝三个木块逐个放到棕色区域，优先抓容易的。没抓到就重新观察并重试；每件最多尝试两次。"


def _wait_until(pred, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for batch condition")


class TestD2CandidateBridgeIntegration(unittest.TestCase):
    def test_hardware_admission_gate_fail_closed(self) -> None:
        bridge = GuardedCandidateBridge(config=GuardedBridgeConfig())
        self.assertEqual(bridge.admission_state.value, "UNADMITTED")
        self.assertEqual(bridge.publisher_count, 1)

        with self.assertRaises(PermissionError):
            bridge.admit_hardware(dry_run_report_sha256="abc")

        bridge_ready = GuardedCandidateBridge(
            config=GuardedBridgeConfig(
                dry_run_report_verified=True,
                confirm_startup_home=True,
                confirm_exit_home=True,
                operator_admitted=True,
            )
        )
        bridge_ready.admit_hardware(dry_run_report_sha256="deadbeef")
        self.assertEqual(bridge_ready.admission_state.value, "ADMITTED_SUPERVISED")

    def test_three_object_closed_loop_with_empty_grasp_recovery_and_live_priority(self) -> None:
        scene_cells = {
            "obj_red_R8_1": ("R8", "red"),
            "obj_yellow_R2_1": ("R2", "yellow"),
            "obj_blue_R4_1": ("R4", "blue"),
        }

        def frame_provider(query: str, placed_ids: set[str], seq: int) -> dict:
            active_cells = {
                cell: color
                for oid, (cell, color) in scene_cells.items()
                if oid not in placed_ids
            }
            if query.startswith("post_lift:"):
                _, oid, kind = query.split(":")
                if kind == "held":
                    # Held -> object disappeared from grid cell!
                    active_cells.pop(scene_cells[oid][0], None)
                return {
                    "cam_high_bgr": make_synthetic_cam_high(active_cells),
                    "wrist_bgr": None,
                    "observation_ts_mono": time.monotonic(),
                }
            if query.startswith("post_release:"):
                return {
                    "cam_high_bgr": make_synthetic_cam_high(active_cells),
                    "wrist_bgr": make_synthetic_wrist_release(block_in_pad=True),
                    "observation_ts_mono": time.monotonic(),
                }
            return {
                "cam_high_bgr": make_synthetic_cam_high(active_cells),
                "wrist_bgr": None,
                "observation_ts_mono": time.monotonic(),
            }

        bridge = GuardedCandidateBridge(
            config=GuardedBridgeConfig(step_delay_s=0.0),
            frame_provider=frame_provider,
        )
        # Inject an empty grasp on whichever object goes first (likely R8 red near home),
        # AND trigger live priority update while grasping!
        bridge.set_empty_grasp_once("obj_red_R8_1")

        update_sent = threading.Event()
        svc = SessionService(bridge=bridge)

        def hook(obj, phase):
            # When the system moves onto yellow after red's empty grasp, inject priority for blue!
            if obj.color == "yellow" and phase == ExecutionPhase.GRASPING and not update_sent.is_set():
                update_sent.set()
                rec = svc.submit_command(
                    client_id="op1",
                    command_id="d2_live_blue",
                    command_type=CommandType.UPDATE_PRIORITY,
                    raw_text="剩余任务先处理蓝色",
                    expected_active_generation=1,
                )
                assert rec.status == CommandStatus.ACCEPTED_PENDING, (rec.status, rec.reason)

        bridge.phase_hook = hook
        svc.start_session("op1", scene_id="cam_high_live")
        start_rec = svc.submit_command(
            client_id="op1",
            command_id="d2_batch",
            command_type=CommandType.START_BATCH,
            raw_text=BATCH_PROMPT,
        )
        self.assertEqual(start_rec.status, CommandStatus.APPLIED)

        _wait_until(
            lambda: svc.get_snapshot()["session_state"] == SessionState.READY.value
            and svc._worker is not None
            and not svc._worker.is_alive()
        )
        snap = svc.get_snapshot(client_id="op1")
        self.assertEqual(snap["metrics"]["headline_score"], "3/3")
        self.assertEqual(snap["metrics"]["empty_grasp_count"], 1)
        self.assertEqual(snap["metrics"]["recovery_success_count"], 1)
        self.assertEqual(snap["active_generation"], 2)
        self.assertEqual(bridge.state.recovery_segments_executed, 1)
        # Red attempt 1 leaves Home (home_takeoff_hover), then 3 inter-object transitions follow:
        # R8 hover -> Yellow R2, Brown Slot 0 -> Blue R4, Brown Slot 1 -> Red R8 retry.
        self.assertEqual(bridge.state.inter_object_transitions_executed, 3)
        self.assertEqual(bridge.state.placed_slots_used, 3)

        # Verify slot assignments 0, 1, 2 were distinct and placement order respected live priority
        slot_by_color = {o["color"]: o["destination_slot"] for o in snap["objects"]}
        self.assertEqual(slot_by_color["yellow"], 0)
        self.assertEqual(slot_by_color["blue"], 1)
        self.assertEqual(slot_by_color["red"], 2)

    def test_brown_pad_capacity_limit_halts_safely(self) -> None:
        scene_cells = {
            "obj_red_R8_1": ("R8", "red"),
            "obj_yellow_R2_1": ("R2", "yellow"),
            "obj_blue_R4_1": ("R4", "blue"),
        }

        def frame_provider(query: str, placed_ids: set[str], seq: int) -> dict:
            active_cells = {cell: color for oid, (cell, color) in scene_cells.items() if oid not in placed_ids}
            if query.startswith("post_lift:"):
                _, oid, kind = query.split(":")
                if kind == "held":
                    active_cells.pop(scene_cells[oid][0], None)
            return {
                "cam_high_bgr": make_synthetic_cam_high(active_cells),
                "wrist_bgr": make_synthetic_wrist_release(block_in_pad=True) if query.startswith("post_release:") else None,
                "observation_ts_mono": time.monotonic(),
            }

        bridge = GuardedCandidateBridge(
            config=GuardedBridgeConfig(max_brown_slots=2, step_delay_s=0.0),
            frame_provider=frame_provider,
        )
        svc = SessionService(bridge=bridge)
        svc.start_session("op1", scene_id="cam_high_live")
        svc.submit_command(client_id="op1", command_id="cap_batch", command_type=CommandType.START_BATCH, raw_text=BATCH_PROMPT)
        _wait_until(
            lambda: svc.get_snapshot()["session_state"] == SessionState.READY.value
            and svc._worker is not None
            and not svc._worker.is_alive()
        )
        snap = svc.get_snapshot(client_id="op1")
        self.assertEqual(snap["metrics"]["completed_count"], 2)
        self.assertEqual(snap["metrics"]["needs_operator_count"], 1)
        self.assertEqual(snap["metrics"]["operator_interventions"], 1)


if __name__ == "__main__":
    unittest.main()
