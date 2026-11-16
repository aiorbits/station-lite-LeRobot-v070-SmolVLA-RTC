"""Stage D2 unit tests for multi-object vision, dual evidence, and SDK trajectories."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from v070_demo.contracts import ContactHeldState, SafetyBudgets
from v070_demo.evidence_verifier import (
    verify_live_grasp_outcome,
    verify_live_placement_outcome,
)
from v070_demo.perception_adapter import (
    HIGH_CAM_GRID_CENTERS_PX,
    compute_brown_pad_slot_pose,
    detect_all_high_cam_blocks,
)
from v070_demo.task_parser import bind_scene_objects, parse_batch_instruction
from v070_demo.trajectory_planner import (
    ANCHOR_SEED_BROWN_LOW,
    ANCHOR_SEED_R8_GRASP,
    SDK_BUNDLE_SCHEMA,
    build_demo_d2_sdk_dryrun_bundle,
    plan_brown_to_next_grid_hover,
    plan_empty_grasp_retreat,
    plan_post_grasp_slot_transport,
    validate_cartesian_segment,
)


COLOR_BGR = {
    "red": (15, 15, 225),
    "yellow": (15, 225, 225),
    "blue": (225, 60, 15),
}


def make_synthetic_cam_high(cell_colors: dict[str, str]) -> np.ndarray:
    """Render a 640x480 synthetic overhead frame with 12x12 colored blocks at calibrated cells."""
    img = np.full((480, 640, 3), 230, dtype=np.uint8)
    for cell, color in cell_colors.items():
        cx, cy = HIGH_CAM_GRID_CENTERS_PX[cell]
        x0, y0 = int(round(cx)) - 6, int(round(cy)) - 6
        cv2.rectangle(img, (x0, y0), (x0 + 12, y0 + 12), COLOR_BGR[color], -1)
    return img


def make_synthetic_wrist_release(*, block_in_pad: bool = True, off_center_px: int = 0) -> np.ndarray:
    """Render a 640x480 wrist frame showing brown target pad and optional released block."""
    # Brown pad in BGR equivalent of muted HSV ~(16, 110, 120)
    hsv = np.zeros((480, 640, 3), dtype=np.uint8)
    hsv[:, :] = (0, 0, 200)  # Neutral bright tabletop
    cv2.rectangle(hsv, (285, 205), (420, 360), (16, 110, 120), -1)
    if block_in_pad:
        cx = 352 + off_center_px
        cy = 282
        cv2.rectangle(hsv, (cx - 28, cy - 28), (cx + 28, cy + 28), (26, 220, 225), -1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


class TestD2PerceptionEvidenceTrajectory(unittest.TestCase):
    def test_multi_object_cam_high_detection_and_duplicate_ambiguity(self) -> None:
        now = time.monotonic()
        img = make_synthetic_cam_high({"R8": "red", "R2": "yellow", "R4": "blue"})
        objs = detect_all_high_cam_blocks(img, observation_id="cam0", observation_ts_mono=now)
        self.assertEqual({o.color for o in objs}, {"red", "yellow", "blue"})
        by_color = {o.color: o for o in objs}
        self.assertEqual(by_color["red"].grid_cell, "R8")
        self.assertEqual(by_color["yellow"].grid_cell, "R2")
        self.assertEqual(by_color["blue"].grid_cell, "R4")
        self.assertTrue(all(o.clearance_m >= SafetyBudgets().min_clearance_m for o in objs))

        # Two red blocks -> must NOT guess; bind_scene_objects rejects with ambiguity
        dup_img = make_synthetic_cam_high({"R8": "red", "R1": "red", "R2": "yellow", "R4": "blue"})
        dup_objs = detect_all_high_cam_blocks(dup_img, observation_id="cam1", observation_ts_mono=now)
        parsed = parse_batch_instruction("把桌上的红、黄、蓝三个木块放到棕色区域")
        bound, err = bind_scene_objects(parsed, dup_objs)
        self.assertEqual(bound, [])
        self.assertIsNotNone(err)
        self.assertTrue(err.startswith("ambiguous_multiple_candidates:red"))

    def test_brown_pad_slots_bounded_inside_calibrated_zone(self) -> None:
        for slot in (0, 1, 2):
            pose = compute_brown_pad_slot_pose(slot)
            self.assertGreaterEqual(pose[0], 0.305)
            self.assertGreaterEqual(pose[1], 0.185)
        with self.assertRaises(ValueError):
            compute_brown_pad_slot_pose(3)

    def test_dual_evidence_grasp_and_placement_matrix(self) -> None:
        img_empty_r8 = make_synthetic_cam_high({"R2": "yellow", "R4": "blue"})
        img_still_r8 = make_synthetic_cam_high({"R8": "red", "R2": "yellow", "R4": "blue"})

        # 1. Held gripper + R8 cleared -> HELD_CONFIRMED
        held = verify_live_grasp_outcome(
            0.0275,
            cam_high_post_lift=img_empty_r8,
            source_cell="R8",
            target_color="red",
        )
        self.assertEqual(held.verdict, "HELD_CONFIRMED")
        self.assertEqual(held.contact_state, ContactHeldState.HELD_CONFIRMED)

        # 2. Empty gripper + R8 still occupied -> EMPTY_GRASP_CONFIRMED
        empty = verify_live_grasp_outcome(
            0.008,
            cam_high_post_lift=img_still_r8,
            source_cell="R8",
            target_color="red",
        )
        self.assertEqual(empty.verdict, "EMPTY_GRASP_CONFIRMED")
        self.assertEqual(empty.contact_state, ContactHeldState.EMPTY_CONFIRMED)

        # 3. Conflict: held gripper yet block still at R8 -> HELD_UNCERTAIN
        conflict = verify_live_grasp_outcome(
            0.0275,
            cam_high_post_lift=img_still_r8,
            source_cell="R8",
            target_color="red",
        )
        self.assertEqual(conflict.verdict, "HELD_UNCERTAIN")
        self.assertEqual(conflict.contact_state, ContactHeldState.HELD_UNCERTAIN)

        # 4. Knocked/lost: empty gripper yet R8 missing -> HELD_UNCERTAIN (never blind retry)
        lost = verify_live_grasp_outcome(
            0.008,
            cam_high_post_lift=img_empty_r8,
            source_cell="R8",
            target_color="red",
        )
        self.assertEqual(lost.verdict, "HELD_UNCERTAIN")

        # 5. Placement dual evidence: open gripper + block centered in brown target
        wrist_ok = make_synthetic_wrist_release(block_in_pad=True, off_center_px=0)
        p_ok = verify_live_placement_outcome(0.068, held_grip_m=0.0275, wrist_post_release=wrist_ok)
        self.assertTrue(p_ok.confirmed)
        self.assertEqual(p_ok.contact_state, ContactHeldState.EMPTY_CONFIRMED)

        wrist_missing_block = make_synthetic_wrist_release(block_in_pad=False)
        p_bad = verify_live_placement_outcome(0.068, held_grip_m=0.0275, wrist_post_release=wrist_missing_block)
        self.assertFalse(p_bad.confirmed)

    def test_recovery_and_inter_object_trajectories_and_sdk_bundle(self) -> None:
        r8_start = np.r_[ANCHOR_SEED_R8_GRASP["pose6"], 0.008]
        retreat = plan_empty_grasp_retreat(r8_start, target_cell="R8")
        stats_ret = validate_cartesian_segment(r8_start, retreat, loop_index=3)
        self.assertLessEqual(stats_ret["max_xyz_step_mm"], 15.0)
        self.assertLessEqual(stats_ret["max_grip_step_mm"], 15.0)
        self.assertGreaterEqual(retreat[-1, 2], 0.200)
        self.assertAlmostEqual(retreat[-1, 6], 0.075, places=3)

        brown_start = np.r_[ANCHOR_SEED_BROWN_LOW["pose6"], 0.0686]
        transit = plan_brown_to_next_grid_hover(brown_start, next_cell="R2")
        stats_tr = validate_cartesian_segment(brown_start, transit, loop_index=6)
        self.assertLessEqual(stats_tr["max_xyz_step_mm"], 15.0)
        # Ensure arm lifts vertically to >= 0.210 m BEFORE any lateral movement begins
        first_lateral_idx = int(np.argmax(np.linalg.norm(transit[:, :2] - brown_start[:2], axis=1) > 1e-4))
        self.assertGreater(first_lateral_idx, 0)
        self.assertGreaterEqual(float(transit[first_lateral_idx - 1, 2]), 0.209)

        r8_held = np.r_[ANCHOR_SEED_R8_GRASP["pose6"], 0.0280]
        slot1 = plan_post_grasp_slot_transport(r8_held, held_grip_m=0.0280, slot_index=1)
        self.assertAlmostEqual(slot1[-1, 0], 0.323, places=3)

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "g4_demo_d2_sdk_bundle.json"
            bundle = build_demo_d2_sdk_dryrun_bundle(out_path)
            self.assertEqual(bundle["schema"], SDK_BUNDLE_SCHEMA)
            self.assertFalse(bundle["hardware_connected"])
            self.assertEqual(len(bundle["cases"]), 10)
            loaded = json.loads(out_path.read_text(encoding="utf-8"))
            for c in loaded["cases"]:
                s_stats = validate_cartesian_segment(c["start_action"], c["actions"], loop_index=c["loop_index"])
                self.assertLessEqual(s_stats["max_xyz_step_mm"], 15.0)


if __name__ == "__main__":
    unittest.main()
