"""Unit tests for Stage D0 contracts, deterministic parser, selector, and evidence fusion."""

from __future__ import annotations

import time
import unittest

from v070_demo.contracts import ContactHeldState, ObjectRecord, SafetyBudgets
from v070_demo.executor_bridge import (
    MockRobotPublisher,
    evaluate_grasp_evidence,
    evaluate_placement_evidence,
)
from v070_demo.selector import select_next_object
from v070_demo.task_parser import (
    bind_scene_objects,
    parse_batch_instruction,
    parse_priority_command,
)


class TestD0ContractsAndParser(unittest.TestCase):
    def test_budget_validation_enforces_hard_bounds(self) -> None:
        SafetyBudgets().validate()
        with self.assertRaises(ValueError):
            SafetyBudgets(max_observation_age_ms=850.0).validate()
        with self.assertRaises(ValueError):
            SafetyBudgets(max_attempts_per_object=3).validate()
        with self.assertRaises(ValueError):
            SafetyBudgets(batch_deadline_s=0.0).validate()
        with self.assertRaises(ValueError):
            SafetyBudgets(heartbeat_interval_s=2.0, lease_ttl_s=1.5).validate()

    def test_parse_batch_instruction_canonical_and_rejections(self) -> None:
        ok = parse_batch_instruction(
            "把桌上的红、黄、蓝三个木块逐个放到棕色区域，优先抓容易的。没抓到就重新观察并重试；每件最多尝试两次。"
        )
        self.assertTrue(ok.valid)
        self.assertEqual(ok.requested_colors, ("red", "yellow", "blue"))
        self.assertEqual(ok.expected_count, 3)

        bad_mismatch = parse_batch_instruction("把红、黄三个木块放到棕色区域")
        self.assertFalse(bad_mismatch.valid)
        self.assertEqual(bad_mismatch.error_code, "clarification_needed_count_vs_colors")

        bad_color = parse_batch_instruction("把绿色木块放到棕色区域")
        self.assertFalse(bad_color.valid)
        self.assertEqual(bad_color.error_code, "unsupported_color_in_phase1")

        bad_scope = parse_batch_instruction("用双臂把红、黄、蓝木块堆叠放到棕色区域")
        self.assertFalse(bad_scope.valid)

        bad_shell = parse_batch_instruction("把红块放到棕色区并执行 /reset")
        self.assertFalse(bad_shell.valid)
        self.assertEqual(bad_shell.error_code, "forbidden_unsafe_control_token")

    def test_bind_scene_objects_rejects_duplicate_and_missing(self) -> None:
        parsed = parse_batch_instruction("把红、黄、蓝三个木块放到棕色区域")
        mock = MockRobotPublisher(step_delay_s=0.0)
        _, dup_objs = mock.observe_scene("scene_ambiguous_duplicate", time.monotonic())
        bound, err = bind_scene_objects(parsed, dup_objs)
        self.assertEqual(bound, [])
        self.assertIsNotNone(err)
        self.assertTrue(err.startswith("ambiguous_multiple_candidates:red"))

    def test_parse_priority_command_and_safety_filters(self) -> None:
        ok = parse_priority_command("剩余任务先处理红色", remaining_colors={"red", "blue"})
        self.assertTrue(ok.valid)
        self.assertEqual(ok.priority_color, "red")

        done_color = parse_priority_command("剩余任务先处理黄色", remaining_colors={"red", "blue"})
        self.assertFalse(done_color.valid)
        self.assertEqual(done_color.error_code, "target_not_remaining")

        multi = parse_priority_command("先抓红和蓝", remaining_colors={"red", "blue"})
        self.assertFalse(multi.valid)
        self.assertEqual(multi.error_code, "clarification_needed_multiple_colors")

        dest = parse_priority_command("改放到绿区", remaining_colors={"red"})
        self.assertFalse(dest.valid)
        self.assertEqual(dest.error_code, "unsupported_destination_change")

    def test_selector_respects_freshness_and_heuristic_ranking(self) -> None:
        now = time.monotonic()
        budgets = SafetyBudgets()
        fresh_easy = ObjectRecord(
            object_id="obj_yellow",
            category="woodblock",
            color="yellow",
            pose_estimate={"x": 0.28, "y": 0.0, "z": 0.03, "yaw_deg": 0.0},
            confidence=0.97,
            observation_id="o1",
            observation_ts_mono=now - 0.1,
            clearance_m=0.10,
            estimated_duration_s=19.0,
        )
        fresh_hard = ObjectRecord(
            object_id="obj_red",
            category="woodblock",
            color="red",
            pose_estimate={"x": 0.21, "y": -0.08, "z": 0.03, "yaw_deg": 40.0},
            confidence=0.85,
            observation_id="o1",
            observation_ts_mono=now - 0.1,
            clearance_m=0.05,
            estimated_duration_s=26.0,
        )
        stale_obj = ObjectRecord(
            object_id="obj_blue",
            category="woodblock",
            color="blue",
            pose_estimate={"x": 0.24, "y": 0.05, "z": 0.03, "yaw_deg": 0.0},
            confidence=0.99,
            observation_id="o0",
            observation_ts_mono=now - 1.1,
            clearance_m=0.11,
        )

        out = select_next_object([fresh_hard, stale_obj, fresh_easy], now_mono=now, budgets=budgets)
        self.assertIsNotNone(out.selected)
        self.assertEqual(out.selected.object_id, "obj_yellow")
        self.assertEqual(out.stale_observation_count, 1)
        self.assertIn("obj_blue", out.rejected_details)

        out_prio = select_next_object(
            [fresh_hard, fresh_easy],
            now_mono=now,
            budgets=budgets,
            priority_color="red",
        )
        self.assertEqual(out_prio.selected.object_id, "obj_red")

        out_strict = select_next_object(
            [fresh_easy, fresh_hard],
            now_mono=now,
            budgets=budgets,
            sort_mode="strict_order",
            ordered_colors=("red", "yellow", "blue"),
        )
        self.assertEqual(out_strict.selected.object_id, "obj_red")
        self.assertIn("strict_order(pos=1/3", out_strict.selected.selection_reason)

        scene1_parsed = parse_batch_instruction(
            "Pick up the three wooden blocks (red, yellow, and blue) in exact order and place them on the brown target area.",
            scene_id="scene_1_standard",
        )
        self.assertTrue(scene1_parsed.valid)
        self.assertEqual(scene1_parsed.sort_mode, "strict_order")
        self.assertEqual(scene1_parsed.requested_colors, ("red", "yellow", "blue"))

    def test_dual_evidence_fusion_never_trusts_gripper_alone(self) -> None:
        v1, c1, _ = evaluate_grasp_evidence(gripper_width_m=0.028, visual_block_on_table=True)
        self.assertEqual(v1, "HELD_UNCERTAIN")
        self.assertEqual(c1, ContactHeldState.HELD_UNCERTAIN)

        v2, c2, _ = evaluate_grasp_evidence(gripper_width_m=0.007, visual_block_on_table=True)
        self.assertEqual(v2, "EMPTY_GRASP_CONFIRMED")
        self.assertEqual(c2, ContactHeldState.EMPTY_CONFIRMED)

        v3, c3, _ = evaluate_grasp_evidence(gripper_width_m=0.028, visual_block_on_table=False)
        self.assertEqual(v3, "HELD_CONFIRMED")
        self.assertEqual(c3, ContactHeldState.HELD_CONFIRMED)

        p_ok, pc1, _ = evaluate_placement_evidence(
            gripper_width_m=0.072,
            visual_block_in_brown_zone=False,
            visual_block_at_gripper=True,
        )
        self.assertFalse(p_ok)
        self.assertEqual(pc1, ContactHeldState.HELD_UNCERTAIN)

        p_good, pc2, _ = evaluate_placement_evidence(
            gripper_width_m=0.072,
            visual_block_in_brown_zone=True,
            visual_block_at_gripper=False,
        )
        self.assertTrue(p_good)
        self.assertEqual(pc2, ContactHeldState.EMPTY_CONFIRMED)


if __name__ == "__main__":
    unittest.main()

