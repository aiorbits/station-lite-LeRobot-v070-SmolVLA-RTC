"""Stage D2 calibrated dual-evidence grasp & placement confirmation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .candidate_import import import_build_grasp_replay_cases
from .contracts import ContactHeldState

_grasp_cases = import_build_grasp_replay_cases()
verify_placement_completion = _grasp_cases.verify_placement_completion
detect_high_cam_grid_target = _grasp_cases.detect_high_cam_grid_target
detect_wrist_block_obb = _grasp_cases.detect_wrist_block_obb
detect_wrist_target_obb = _grasp_cases.detect_wrist_target_obb

# Physical 27mm woodblock closed-grip range verified across 11-13 and 11-16 runs (0.0268..0.0297 m).
HELD_GRIP_MIN_M = 0.018
HELD_GRIP_MAX_M = 0.045
EMPTY_CLOSED_MAX_M = 0.012


@dataclass(frozen=True)
class GraspVerificationResult:
    verdict: str  # "HELD_CONFIRMED", "EMPTY_GRASP_CONFIRMED", "HELD_UNCERTAIN"
    contact_state: ContactHeldState
    evidence: dict[str, Any]


@dataclass(frozen=True)
class PlacementVerificationResult:
    confirmed: bool
    contact_state: ContactHeldState
    evidence: dict[str, Any]


def verify_live_grasp_outcome(
    measured_grip_m: float,
    *,
    cam_high_post_lift=None,
    wrist_post_lift=None,
    source_cell: str | None = None,
    target_color: str = "yellow",
) -> GraspVerificationResult:
    """Fuse calibrated gripper feedback AND post-lift vision per Section 6.2.

    Never treats gripper width or action-chunk completion alone as proof of grasp.
    """
    if not np.isfinite(measured_grip_m):
        return GraspVerificationResult(
            verdict="HELD_UNCERTAIN",
            contact_state=ContactHeldState.HELD_UNCERTAIN,
            evidence={"reason": "non_finite_gripper"},
        )

    grip = float(measured_grip_m)
    grip_in_block_band = HELD_GRIP_MIN_M <= grip <= HELD_GRIP_MAX_M
    grip_empty_closed = grip <= EMPTY_CLOSED_MAX_M

    visual_block_still_at_cell: bool | None = None
    high_cam_det = None
    if cam_high_post_lift is not None and source_cell is not None:
        high_cam_det = detect_high_cam_grid_target(cam_high_post_lift, target_color=target_color)
        if high_cam_det is not None and high_cam_det.get("cell") == source_cell:
            visual_block_still_at_cell = True
        else:
            visual_block_still_at_cell = False

    wrist_obb = (
        detect_wrist_block_obb(wrist_post_lift, target_color=target_color)
        if wrist_post_lift is not None
        else None
    )

    evidence = {
        "measured_grip_m": round(grip, 4),
        "grip_in_block_band": grip_in_block_band,
        "grip_empty_closed": grip_empty_closed,
        "source_cell": source_cell,
        "visual_block_still_at_cell": visual_block_still_at_cell,
        "high_cam_post_lift_det": high_cam_det,
        "wrist_post_lift_obb": wrist_obb,
    }

    # Conflicting or missing visual evidence -> HELD_UNCERTAIN (stops safely!).
    if visual_block_still_at_cell is None and wrist_obb is None:
        return GraspVerificationResult(
            verdict="HELD_UNCERTAIN",
            contact_state=ContactHeldState.HELD_UNCERTAIN,
            evidence={**evidence, "reason": "missing_post_lift_visual_evidence"},
        )

    if grip_in_block_band:
        if visual_block_still_at_cell is False or (visual_block_still_at_cell is None and wrist_obb is not None):
            return GraspVerificationResult(
                verdict="HELD_CONFIRMED",
                contact_state=ContactHeldState.HELD_CONFIRMED,
                evidence=evidence,
            )
        return GraspVerificationResult(
            verdict="HELD_UNCERTAIN",
            contact_state=ContactHeldState.HELD_UNCERTAIN,
            evidence={**evidence, "reason": "gripper_held_but_block_still_on_table"},
        )

    if grip_empty_closed:
        if visual_block_still_at_cell is True and wrist_obb is None:
            return GraspVerificationResult(
                verdict="EMPTY_GRASP_CONFIRMED",
                contact_state=ContactHeldState.EMPTY_CONFIRMED,
                evidence=evidence,
            )
        reason = (
            "gripper_empty_yet_block_present_in_wrist"
            if wrist_obb is not None
            else "gripper_empty_and_target_missing_from_cell"
        )
        return GraspVerificationResult(
            verdict="HELD_UNCERTAIN",
            contact_state=ContactHeldState.HELD_UNCERTAIN,
            evidence={**evidence, "reason": reason},
        )

    return GraspVerificationResult(
        verdict="HELD_UNCERTAIN",
        contact_state=ContactHeldState.HELD_UNCERTAIN,
        evidence={**evidence, "reason": "gripper_width_ambiguous"},
    )


def verify_live_placement_outcome(
    measured_grip_m: float,
    *,
    held_grip_m: float,
    wrist_post_release=None,
    target_color: str = "yellow",
    require_target_color: bool = False,
    fallback_target_obb: dict[str, Any] | None = None,
) -> PlacementVerificationResult:
    """Wrap calibrated verify_placement_completion and require explicit visual confirmation."""
    if wrist_post_release is None:
        return PlacementVerificationResult(
            confirmed=False,
            contact_state=ContactHeldState.UNKNOWN,
            evidence={"reason": "missing_wrist_post_release_visual"},
        )

    raw_info = verify_placement_completion(
        measured_grip=measured_grip_m,
        held_grip=held_grip_m,
        wrist_image_or_bytes=wrist_post_release,
    )
    target_obb = raw_info.get("target_obb")
    used_fallback_target = False
    if target_obb is None and fallback_target_obb is not None:
        target_obb = fallback_target_obb
        used_fallback_target = True
    block_obb = (
        detect_wrist_block_obb(wrist_post_release, target_color=target_color)
        if require_target_color
        else raw_info.get("block_obb")
    )
    if target_obb is None or block_obb is None:
        return PlacementVerificationResult(
            confirmed=False,
            contact_state=ContactHeldState.UNKNOWN,
            evidence={
                **raw_info,
                "target_obb": target_obb,
                "used_fallback_target_obb": used_fallback_target,
                "target_color": target_color,
                "block_obb": block_obb,
                "confirmed": False,
                "reason": "missing_brown_target_or_block_obb_after_release",
            },
        )

    dist_px = float(
        math.hypot(
            float(block_obb["cx"]) - float(target_obb["cx"]),
            float(block_obb["cy"]) - float(target_obb["cy"]),
        )
    )
    max_allowed_px = float(max(float(target_obb["w"]), float(target_obb["h"])) * 0.85)
    visual_confirmed = bool(dist_px <= max_allowed_px)
    cy_px = float(block_obb.get("cy", 0.0))
    still_clutched_near_fingers = cy_px > 338.0 and measured_grip_m < 0.052
    confirmed = bool(raw_info.get("gripper_opened_ok", False) and visual_confirmed and not still_clutched_near_fingers)
    contact_state = ContactHeldState.EMPTY_CONFIRMED if confirmed else ContactHeldState.HELD_UNCERTAIN
    return PlacementVerificationResult(
        confirmed=confirmed,
        contact_state=contact_state,
        evidence={
            **raw_info,
            "target_obb": target_obb,
            "used_fallback_target_obb": used_fallback_target,
            "block_obb": block_obb,
            "dist_px": round(dist_px, 2),
            "max_allowed_dist_px": round(max_allowed_px, 2),
            "visual_confirmed": visual_confirmed,
            "target_color": target_color,
            "require_target_color": require_target_color,
            "still_clutched_near_fingers": still_clutched_near_fingers,
            "confirmed": confirmed,
        },
    )
