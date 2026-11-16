"""Deterministic software & fault-injection suite covering Section 9 D1 scenarios."""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from v070_demo.contracts import (
    CommandStatus,
    CommandType,
    ContactHeldState,
    ExecutionPhase,
    ObjectState,
    SafetyBudgets,
    SessionState,
)
from v070_demo.event_logger import DemoRunLogger
from v070_demo.executor_bridge import FaultConfig, MockRobotPublisher
from v070_demo.session_service import SessionService


DEFAULT_BATCH = "把桌上的红、黄、蓝三个木块逐个放到棕色区域，优先抓容易的。没抓到就重新观察并重试；每件最多尝试两次。"


def _wait_until(predicate, *, timeout_s: float = 4.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for condition")


class FakeClock:
    def __init__(self, initial: float = 100.0) -> None:
        self.now = initial

    def __call__(self) -> float:
        return self.now

    def advance(self, delta: float) -> None:
        self.now += delta


class TestD1SessionAndFaultInjection(unittest.TestCase):
    def _new_service(
        self,
        *,
        faults: FaultConfig | None = None,
        budgets: SafetyBudgets | None = None,
        clock: FakeClock | None = None,
        run_dir: Path | None = None,
    ) -> tuple[SessionService, MockRobotPublisher]:
        clk = clock or FakeClock()
        bridge = MockRobotPublisher(step_delay_s=0.0, faults=faults, clock_fn=clk)
        svc = SessionService(
            bridge=bridge,
            budgets=budgets or SafetyBudgets(),
            logger=DemoRunLogger(run_dir),
            clock_fn=clk,
            watchdog_poll_s=0.02,
        )
        return svc, bridge

    def _finish_batch(self, svc: SessionService) -> dict:
        _wait_until(
            lambda: svc.get_snapshot()["session_state"]
            in ("READY", "PAUSED", "FAULTED", "CLOSED")
            and svc._worker is not None
            and not svc._worker.is_alive()
        )
        return svc.get_snapshot(client_id="op1")

    def test_start_session_has_zero_hardware_side_effects(self) -> None:
        svc, bridge = self._new_service()
        snap = svc.start_session("op1", scene_id="scene_1_standard")
        self.assertEqual(snap["session_state"], SessionState.READY.value)
        self.assertEqual(snap["hardware_admission"], "SIMULATION_MOCK")
        self.assertEqual(bridge.robot_connect_calls, 0)
        self.assertEqual(bridge.home_calls, 0)
        self.assertEqual(bridge.publisher_count, 1)

    def test_happy_path_writes_all_manifest_and_event_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_01"
            svc, _ = self._new_service(run_dir=run_dir)
            svc.start_session("op1", scene_id="scene_2_mixed_distance")
            cmd = svc.submit_command(
                client_id="op1",
                command_id="start_1",
                command_type=CommandType.START_BATCH,
                raw_text=DEFAULT_BATCH,
                scene_id="scene_2_mixed_distance",
            )
            self.assertEqual(cmd.status, CommandStatus.APPLIED)
            snap = self._finish_batch(svc)
            self.assertEqual(snap["metrics"]["headline_score"], "3/3")
            self.assertEqual(snap["metrics"]["operator_interventions"], 0)

            for fname in (
                "run_manifest.json",
                "scene_manifest.json",
                "metrics.json",
                "session_events.jsonl",
                "task_events.jsonl",
                "command_events.jsonl",
            ):
                self.assertTrue((run_dir / fname).is_file(), f"Missing artifact {fname}")
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["completed_count"], 3)

    def test_empty_grasp_recovery_and_two_failure_budget(self) -> None:
        # Attempt 1 fails -> verified recovery -> attempt 2 succeeds (scene_3 preset)
        svc, _ = self._new_service()
        svc.start_session("op1", scene_id="scene_3_empty_grasp_recovery")
        svc.submit_command(
            client_id="op1",
            command_id="b1",
            command_type=CommandType.START_BATCH,
            raw_text=DEFAULT_BATCH,
            scene_id="scene_3_empty_grasp_recovery",
        )
        snap = self._finish_batch(svc)
        self.assertEqual(snap["metrics"]["headline_score"], "3/3")
        self.assertEqual(snap["metrics"]["empty_grasp_count"], 1)
        self.assertEqual(snap["metrics"]["recovery_success_count"], 1)

        # Two consecutive failures on obj_red_1 -> capped at FAILED, continues safely to remaining 2
        faults = FaultConfig(empty_grasp_attempts={("obj_red_1", 1): True, ("obj_red_1", 2): True})
        svc2, _ = self._new_service(faults=faults)
        svc2.start_session("op1", scene_id="scene_1_standard")
        svc2.submit_command(
            client_id="op1",
            command_id="b2",
            command_type=CommandType.START_BATCH,
            raw_text=DEFAULT_BATCH,
            scene_id="scene_1_standard",
        )
        snap2 = self._finish_batch(svc2)
        self.assertEqual(snap2["metrics"]["headline_score"], "2/3")
        self.assertEqual(snap2["metrics"]["failed_count"], 1)
        red_obj = next(o for o in snap2["objects"] if o["object_id"] == "obj_red_1")
        self.assertEqual(red_obj["status"], ObjectState.FAILED.value)
        self.assertEqual(red_obj["attempt_count"], 2)

    def test_ambiguous_detection_and_missing_target(self) -> None:
        svc, _ = self._new_service()
        svc.start_session("op1", scene_id="scene_ambiguous_duplicate")
        rej = svc.submit_command(
            client_id="op1",
            command_id="b_amb",
            command_type=CommandType.START_BATCH,
            raw_text=DEFAULT_BATCH,
            scene_id="scene_ambiguous_duplicate",
        )
        self.assertEqual(rej.status, CommandStatus.REJECTED)
        self.assertIn("clarification_needed", rej.reason)

        # Mid-batch disappearance of yellow block -> deferred two rounds -> NEEDS_OPERATOR
        faults = FaultConfig(hide_color_on_reobserve={"yellow"})
        svc2, _ = self._new_service(faults=faults)
        svc2.start_session("op1", scene_id="scene_1_standard")
        svc2.submit_command(
            client_id="op1",
            command_id="b_miss",
            command_type=CommandType.START_BATCH,
            raw_text=DEFAULT_BATCH,
            scene_id="scene_1_standard",
        )
        snap2 = self._finish_batch(svc2)
        self.assertEqual(snap2["metrics"]["completed_count"], 2)
        self.assertEqual(snap2["metrics"]["needs_operator_count"], 1)

    def test_uncertain_held_and_unconfirmed_placement_halt_safely(self) -> None:
        svc, bridge = self._new_service(faults=FaultConfig(uncertain_held_objects={"obj_red_1"}))
        svc.start_session("op1", scene_id="scene_1_standard")
        svc.submit_command(
            client_id="op1",
            command_id="b_unc",
            command_type=CommandType.START_BATCH,
            raw_text=DEFAULT_BATCH,
        )
        snap = self._finish_batch(svc)
        self.assertEqual(snap["session_state"], SessionState.PAUSED.value)
        self.assertEqual(snap["contact_held_state"], ContactHeldState.HELD_UNCERTAIN.value)
        self.assertEqual(bridge.home_calls, 0)

        svc2, _ = self._new_service(faults=FaultConfig(placement_unconfirmed_objects={"obj_red_1"}))
        svc2.start_session("op1", scene_id="scene_1_standard")
        svc2.submit_command(
            client_id="op1",
            command_id="b_place_fail",
            command_type=CommandType.START_BATCH,
            raw_text=DEFAULT_BATCH,
        )
        snap2 = self._finish_batch(svc2)
        self.assertEqual(snap2["session_state"], SessionState.PAUSED.value)
        self.assertEqual(snap2["metrics"]["needs_operator_count"], 1)
        self.assertEqual(snap2["metrics"]["completed_count"], 0)

    def test_timeout_guard_rejection_stale_obs_and_two_round_infeasible(self) -> None:
        # 1. Inference timeout
        svc_to, _ = self._new_service(faults=FaultConfig(timeout_on=("obj_red_1", ExecutionPhase.GRASPING)))
        svc_to.start_session("op1")
        svc_to.submit_command(client_id="op1", command_id="b_to", command_type=CommandType.START_BATCH, raw_text=DEFAULT_BATCH)
        self.assertEqual(self._finish_batch(svc_to)["session_state"], SessionState.FAULTED.value)

        # 2. Path guard rejection during execution
        svc_gr, _ = self._new_service(faults=FaultConfig(guard_reject_on=("obj_red_1", ExecutionPhase.TRANSPORTING)))
        svc_gr.start_session("op1")
        svc_gr.submit_command(client_id="op1", command_id="b_gr", command_type=CommandType.START_BATCH, raw_text=DEFAULT_BATCH)
        snap_gr = self._finish_batch(svc_gr)
        self.assertEqual(snap_gr["metrics"]["needs_operator_count"], 1)

        # 3. Stale observation > 800 ms across 2 planning rounds -> NEEDS_OPERATOR
        svc_st, _ = self._new_service(faults=FaultConfig(stale_obs_age_ms=1200.0))
        svc_st.start_session("op1")
        svc_st.submit_command(client_id="op1", command_id="b_st", command_type=CommandType.START_BATCH, raw_text=DEFAULT_BATCH)
        snap_st = self._finish_batch(svc_st)
        self.assertEqual(snap_st["metrics"]["needs_operator_count"], 3)

        # 4. Two rounds all infeasible -> stops with NEEDS_OPERATOR instead of infinite loop
        svc_inf, _ = self._new_service(
            faults=FaultConfig(infeasible_objects_next_rounds={"obj_red_1": 3, "obj_yellow_1": 3, "obj_blue_1": 3})
        )
        svc_inf.start_session("op1")
        svc_inf.submit_command(client_id="op1", command_id="b_inf", command_type=CommandType.START_BATCH, raw_text=DEFAULT_BATCH)
        snap_inf = self._finish_batch(svc_inf)
        self.assertEqual(snap_inf["metrics"]["needs_operator_count"], 3)

    def test_stale_generation_result_and_budget_exhaustion(self) -> None:
        svc_sg, _ = self._new_service(faults=FaultConfig(stale_result_generation_once=99))
        svc_sg.start_session("op1")
        svc_sg.submit_command(client_id="op1", command_id="b_sg", command_type=CommandType.START_BATCH, raw_text=DEFAULT_BATCH)
        snap_sg = self._finish_batch(svc_sg)
        self.assertEqual(snap_sg["session_state"], SessionState.FAULTED.value)

        clk = FakeClock(initial=10.0)
        svc_bud, bridge_bud = self._new_service(budgets=SafetyBudgets(batch_deadline_s=5.0), clock=clk)

        def bump_clock(obj, phase):
            if phase == ExecutionPhase.RELEASING_CHECK:
                clk.advance(6.0)

        bridge_bud.phase_hook = bump_clock
        svc_bud.start_session("op1")
        svc_bud.submit_command(client_id="op1", command_id="b_bud", command_type=CommandType.START_BATCH, raw_text=DEFAULT_BATCH)
        snap_bud = self._finish_batch(svc_bud)
        self.assertEqual(snap_bud["metrics"]["completed_count"], 1)
        self.assertEqual(snap_bud["metrics"]["needs_operator_count"], 2)

    def test_in_flight_priority_update_supersede_and_checkpoint_switch(self) -> None:
        # Scene 2 starts with yellow easiest; pause mid-grasp of yellow and queue red -> superseded by blue!
        svc, bridge = self._new_service()
        gate_entered = threading.Event()
        release_gate = threading.Event()

        def hook(obj, phase):
            if obj.color == "yellow" and phase == ExecutionPhase.GRASPING:
                gate_entered.set()
                release_gate.wait(timeout=3.0)

        bridge.phase_hook = hook
        svc.start_session("op1", scene_id="scene_2_mixed_distance")
        svc.submit_command(
            client_id="op1",
            command_id="start_gen",
            command_type=CommandType.START_BATCH,
            raw_text=DEFAULT_BATCH,
            scene_id="scene_2_mixed_distance",
        )
        self.assertTrue(gate_entered.wait(timeout=2.0))

        # Submit two updates while holding/grasping yellow: first becomes SUPERSEDED by second
        c1 = svc.submit_command(
            client_id="op1",
            command_id="prio_red",
            command_type=CommandType.UPDATE_PRIORITY,
            raw_text="剩余任务先处理红色",
            expected_active_generation=1,
        )
        self.assertEqual(c1.status, CommandStatus.ACCEPTED_PENDING)

        c2 = svc.submit_command(
            client_id="op1",
            command_id="prio_blue",
            command_type=CommandType.UPDATE_PRIORITY,
            raw_text="剩余任务先处理蓝色",
            expected_active_generation=1,
        )
        self.assertEqual(c2.status, CommandStatus.ACCEPTED_PENDING)
        # Generation must NOT bump mid-grasp!
        self.assertEqual(svc.get_snapshot()["active_generation"], 1)

        release_gate.set()
        snap = self._finish_batch(svc)

        self.assertEqual(snap["metrics"]["headline_score"], "3/3")
        # Yellow finished first, then checkpoint bumped gen -> blue processed before red!
        self.assertEqual(bridge.placed_history, ["obj_yellow_2", "obj_blue_2", "obj_red_2"])
        self.assertEqual(snap["active_generation"], 2)
        self.assertGreaterEqual(len(bridge.cancelled_generations), 1)
        cmds = {c["command_id"]: c for c in snap["commands"]}
        self.assertEqual(cmds["prio_red"]["status"], CommandStatus.SUPERSEDED.value)
        self.assertEqual(cmds["prio_blue"]["status"], CommandStatus.APPLIED.value)
        self.assertEqual(cmds["prio_blue"]["switch_boundary"]["last_old_object_id"], "obj_yellow_2")

    def test_idempotency_single_owner_disconnect_and_pause_resume(self) -> None:
        clk = FakeClock(initial=50.0)
        svc, bridge = self._new_service(clock=clk)
        svc.start_session("op1")

        # Secondary browser tab cannot mutate
        denied = svc.submit_command(
            client_id="tab2",
            command_id="bad_start",
            command_type=CommandType.START_BATCH,
            raw_text=DEFAULT_BATCH,
        )
        self.assertEqual(denied.status, CommandStatus.REJECTED)
        self.assertIn("read_only_client", denied.reason)

        # Slow phase to test pause, disconnect expiry, reclaim & resume
        entered = threading.Event()
        resume_gate = threading.Event()

        def hook(obj, phase):
            if phase == ExecutionPhase.APPROACHING_EMPTY and not entered.is_set():
                entered.set()
                resume_gate.wait(timeout=3.0)

        bridge.phase_hook = hook
        r1 = svc.submit_command(client_id="op1", command_id="dup_batch", command_type=CommandType.START_BATCH, raw_text=DEFAULT_BATCH)
        r2 = svc.submit_command(client_id="op1", command_id="dup_batch", command_type=CommandType.START_BATCH, raw_text=DEFAULT_BATCH)
        self.assertEqual(r1.batch_id, r2.batch_id)

        self.assertTrue(entered.wait(timeout=2.0))
        pause_rec = svc.submit_command(client_id="op1", command_id="pause_1", command_type=CommandType.PAUSE)
        self.assertEqual(pause_rec.status, CommandStatus.APPLIED)
        self.assertEqual(svc.get_snapshot()["session_state"], SessionState.PAUSED.value)
        resume_gate.set()

        # Simulate heartbeat expiry while paused -> late resume rejected until reclaim
        clk.advance(10.0)
        self.assertTrue(svc.check_lease_watchdog())
        late_resume = svc.submit_command(client_id="op1", command_id="late_res", command_type=CommandType.RESUME)
        self.assertEqual(late_resume.status, CommandStatus.REJECTED)
        self.assertIn("lease_expired", late_resume.reason)

        # Claim control and resume cleanly with fresh observation
        svc.claim_control("op1")
        res_ok = svc.submit_command(client_id="op1", command_id="res_ok", command_type=CommandType.RESUME)
        self.assertEqual(res_ok.status, CommandStatus.APPLIED)

        snap = self._finish_batch(svc)
        self.assertEqual(snap["metrics"]["headline_score"], "3/3")

    def test_pause_precedence_over_priority_cancel_and_close_session(self) -> None:
        svc, bridge = self._new_service()
        entered_second = threading.Event()
        release_second = threading.Event()

        def hook(obj, phase):
            if obj.color != "yellow" and phase == ExecutionPhase.APPROACHING_EMPTY and not entered_second.is_set():
                entered_second.set()
                release_second.wait(timeout=3.0)

        bridge.phase_hook = hook
        svc.start_session("op1", scene_id="scene_2_mixed_distance")
        svc.submit_command(
            client_id="op1",
            command_id="b_ctrl",
            command_type=CommandType.START_BATCH,
            raw_text=DEFAULT_BATCH,
            scene_id="scene_2_mixed_distance",
        )
        self.assertTrue(entered_second.wait(timeout=2.0))

        # Pause must remain PAUSED even if a priority update is submitted while paused
        p = svc.submit_command(client_id="op1", command_id="p_pause", command_type=CommandType.PAUSE)
        self.assertEqual(p.status, CommandStatus.APPLIED)
        u = svc.submit_command(
            client_id="op1",
            command_id="u_while_paused",
            command_type=CommandType.UPDATE_PRIORITY,
            raw_text="剩余任务先处理红色",
        )
        self.assertEqual(u.status, CommandStatus.ACCEPTED_PENDING)
        self.assertEqual(svc.get_snapshot()["session_state"], SessionState.PAUSED.value)

        # Cancel batch must cancel the pending priority update and never auto-Home
        release_second.set()
        c = svc.submit_command(client_id="op1", command_id="c_cancel", command_type=CommandType.CANCEL_BATCH)
        self.assertEqual(c.status, CommandStatus.APPLIED)
        snap = self._finish_batch(svc)
        self.assertEqual(snap["session_state"], SessionState.READY.value)
        self.assertEqual(snap["metrics"]["completed_count"], 1)
        self.assertEqual(bridge.home_calls, 0)
        cmds = {item["command_id"]: item for item in snap["commands"]}
        self.assertEqual(cmds["u_while_paused"]["status"], CommandStatus.CANCELLED.value)

        # Close session prevents further batch starts without implicit Home
        cl = svc.submit_command(client_id="op1", command_id="cl_close", command_type=CommandType.CLOSE_SESSION)
        self.assertEqual(cl.status, CommandStatus.APPLIED)
        self.assertEqual(svc.get_snapshot()["session_state"], SessionState.CLOSED.value)
        after_close = svc.submit_command(
            client_id="op1",
            command_id="after_close",
            command_type=CommandType.START_BATCH,
            raw_text=DEFAULT_BATCH,
        )
        self.assertEqual(after_close.status, CommandStatus.REJECTED)
        self.assertEqual(bridge.home_calls, 0)


if __name__ == "__main__":
    unittest.main()
