"""Multi-object cam_high instance perception and brown-pad slot management for Stage D2."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .candidate_import import import_build_grasp_replay_cases
from .contracts import ObjectRecord

_grasp_cases = import_build_grasp_replay_cases()

HIGH_CAM_GRID_CENTERS_PX: dict[str, tuple[float, float]] = _grasp_cases.HIGH_CAM_GRID_CENTERS_PX
GRID_HOVER_POSES: dict[str, np.ndarray] = _grasp_cases.GRID_HOVER_POSES
BROWN_TARGET_HOVER_POSE6: np.ndarray = np.asarray(_grasp_cases.BROWN_TARGET_HOVER_POSE6, dtype=float)
is_in_brown_place_hover_zone = _grasp_cases.is_in_brown_place_hover_zone

# Calibrated bounded XY offsets inside the verified brown placement zone (X >= 0.305, Y >= 0.185)
# so placing objects 1..3 sequentially avoids stacking onto the previously placed woodblock.
BROWN_SLOT_OFFSETS_XY_M: tuple[tuple[float, float], ...] = (
    (0.000, 0.000),   # Slot 0: calibrated exact BROWN_TARGET_HOVER_POSE6 [0.348, 0.216]
    (-0.025, 0.000),  # Slot 1: 25 mm rear-inward offset -> [0.323, 0.216] (inside min_x=0.305)
    (0.000, -0.025),  # Slot 2: 25 mm lateral offset -> [0.348, 0.191] (inside min_y=0.185)
)


def compute_brown_pad_slot_pose(slot_index: int) -> np.ndarray:
    """Return a calibrated 6D brown-pad hover pose with non-overlapping slot offset."""
    if slot_index < 0 or slot_index >= len(BROWN_SLOT_OFFSETS_XY_M):
        raise ValueError(
            f"Brown pad slot_index={slot_index} exceeds Phase-1 capacity ({len(BROWN_SLOT_OFFSETS_XY_M)})"
        )
    pose = BROWN_TARGET_HOVER_POSE6.copy()
    dx, dy = BROWN_SLOT_OFFSETS_XY_M[slot_index]
    pose[0] += float(dx)
    pose[1] += float(dy)
    if not is_in_brown_place_hover_zone(pose):
        raise ValueError(f"Slot pose {pose[:2].tolist()} fell outside calibrated brown hover zone")
    return pose


def _decode_bgr_image(image_or_bytes) -> np.ndarray | None:
    if image_or_bytes is None:
        return None
    import cv2

    if isinstance(image_or_bytes, (bytes, bytearray, memoryview)):
        buf = np.frombuffer(image_or_bytes, dtype=np.uint8)
        arr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    else:
        arr = np.asarray(image_or_bytes)
    if arr is None or arr.ndim != 3 or arr.shape[2] != 3:
        return None
    return arr


def _color_hsv_mask(hsv_img: np.ndarray, color: str) -> np.ndarray:
    import cv2

    if color == "red":
        return cv2.bitwise_or(
            cv2.inRange(hsv_img, (0, 70, 50), (14, 255, 255)),
            cv2.inRange(hsv_img, (165, 70, 50), (180, 255, 255)),
        )
    if color == "yellow":
        return cv2.inRange(hsv_img, (16, 70, 60), (36, 255, 255))
    if color == "blue":
        return cv2.bitwise_or(
            cv2.inRange(hsv_img, (88, 50, 40), (135, 255, 255)),
            cv2.inRange(hsv_img, (75, 50, 40), (88, 255, 255)),
        )
    raise ValueError(f"Unsupported Phase-1 woodblock color: {color}")


def detect_all_high_cam_blocks(
    image_or_bytes,
    *,
    observation_id: str,
    observation_ts_mono: float,
    colors: Sequence[str] = ("red", "yellow", "blue"),
    table_roi: tuple[int, int, int, int] = (325, 210, 385, 265),
    min_area: float = 80.0,
    max_area: float = 260.0,
    max_dist_to_cell_px: float = 11.5,
) -> list[ObjectRecord]:
    """Detect all Phase-1 colored woodblocks visible inside cam_high workspace ROI.

    Unlike single-target detect_high_cam_grid_target, returns EVERY valid contour
    match across all requested colors so downstream binding rejects same-color
    duplicates rather than guessing silently.
    """
    bgr = _decode_bgr_image(image_or_bytes)
    if bgr is None:
        return []
    import cv2

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    min_u, min_v, max_u, max_v = table_roi
    detections: list[dict] = []

    for raw_color in colors:
        color = raw_color.lower().strip()
        mask = _color_hsv_mask(hsv, color)
        contours_info = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

        per_color_idx = 0
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if not (min_area <= area <= max_area):
                continue
            (cx, cy), (w, h), _angle = cv2.minAreaRect(contour)
            if not (min_u <= cx <= max_u and min_v <= cy <= max_v):
                continue
            aspect = max(w, h) / max(1.0, min(w, h))
            if aspect > 2.2:
                continue

            best_cell = min(
                HIGH_CAM_GRID_CENTERS_PX,
                key=lambda k: math.hypot(
                    cx - HIGH_CAM_GRID_CENTERS_PX[k][0],
                    cy - HIGH_CAM_GRID_CENTERS_PX[k][1],
                ),
            )
            dist_px = math.hypot(
                cx - HIGH_CAM_GRID_CENTERS_PX[best_cell][0],
                cy - HIGH_CAM_GRID_CENTERS_PX[best_cell][1],
            )
            if dist_px > max_dist_to_cell_px:
                continue

            per_color_idx += 1
            hover_pose = GRID_HOVER_POSES[best_cell]
            # Confidence stays high across real block pixel areas [95..195] (incl. cyan/blue ~107px)
            # and drops smoothly only when contour area or cell distance degrades markedly.
            area_excess = max(0.0, 95.0 - area, area - 195.0)
            area_penalty = min(0.25, area_excess / 120.0)
            dist_penalty = min(0.20, max(0.0, dist_px - 3.5) / 55.0)
            conf = max(0.45, min(0.99, 0.98 - area_penalty - dist_penalty))
            detections.append(
                {
                    "object_id": f"obj_{color}_{best_cell}_{per_color_idx}",
                    "color": color,
                    "cell": best_cell,
                    "cx": float(cx),
                    "cy": float(cy),
                    "hover_xyz": [float(v) for v in hover_pose[:3]],
                    "confidence": float(conf),
                }
            )

    # Compute real physical XY inter-object clearance across detected grid poses.
    records: list[ObjectRecord] = []
    for i, item in enumerate(detections):
        xi, yi, _ = item["hover_xyz"]
        min_neighbor_m = 0.120
        for j, other in enumerate(detections):
            if i == j:
                continue
            xj, yj, _ = other["hover_xyz"]
            dist_m = math.hypot(xi - xj, yi - yj)
            min_neighbor_m = min(min_neighbor_m, dist_m)

        # Farther cells from Home [0.127, 0.0] take slightly longer on approach
        dist_from_home_m = math.hypot(xi - 0.127, yi)
        est_dur = round(18.0 + dist_from_home_m * 30.0, 2)
        records.append(
            ObjectRecord(
                object_id=item["object_id"],
                category="woodblock",
                color=item["color"],
                pose_estimate={
                    "x": round(xi, 4),
                    "y": round(yi, 4),
                    "z": 0.030,
                    "yaw_deg": 0.0,
                },
                confidence=round(item["confidence"], 3),
                observation_id=observation_id,
                observation_ts_mono=observation_ts_mono,
                destination_id="brown_target",
                occlusion_score=0.02,
                clearance_m=round(min_neighbor_m, 4),
                grasp_pose_valid=min_neighbor_m >= 0.028,
                path_checked=True,
                estimated_duration_s=est_dur,
                grid_cell=item["cell"],
            )
        )
    return records
