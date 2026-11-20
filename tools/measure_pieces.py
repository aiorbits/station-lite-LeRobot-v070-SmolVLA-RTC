#!/usr/bin/env python3
"""Measure how each DUPLO piece looks to the head camera (cam_high).

Run on the robot (ido) from ~/smolvla, with image-server running. No arm, no motors.
Place ONE piece in a grid cell, type a label (e.g. yellow_2x4_R5), press Enter.
For every coloured patch near the grid it prints the size the teacher's finder
sees, and whether that finder (detect_high_cam_grid_target / perception_adapter)
would accept it, using his exact colour ranges and limits. Empty label = quit.

Output: ~/smolvla/logs/piece_sizes/<label>.jpg (zoomed, marked-up crop) and
        ~/smolvla/logs/piece_sizes/sizes.csv (one row per patch).
"""
import base64
import csv
import math
import os
import sys
import time

import cv2
import numpy as np

REPO_SRC = os.path.expanduser("~/smolvla/src")
sys.path.insert(0, REPO_SRC)
from teleimager.image_client import ImageClient  # noqa: E402
from station_lite_smolvla_single_chunk import wait_frame, enc_jpeg_b64  # noqa: E402
from build_grasp_replay_cases import HIGH_CAM_GRID_CENTERS_PX as CELLS  # noqa: E402

# The teacher's limits (build_grasp_replay_cases.detect_high_cam_grid_target and
# v070_demo.perception_adapter defaults), copied so we can report each check.
TABLE_ROI = (325, 210, 385, 265)  # min_u, min_v, max_u, max_v
try:  # limits as the robot's copy of the code has them (after duplo_size_limits.patch)
    from build_grasp_replay_cases import (  # noqa: E402
        HIGH_CAM_BLOCK_MIN_AREA_PX as MIN_AREA,
        HIGH_CAM_BLOCK_MAX_AREA_PX as MAX_AREA,
        HIGH_CAM_BLOCK_MAX_ASPECT as MAX_ASPECT,
    )
except ImportError:  # teacher's original values
    MIN_AREA, MAX_AREA = 80.0, 260.0
    MAX_ASPECT = 2.2
MAX_DIST_TO_CELL_PX = 11.5
COLOUR_RANGES = {
    "red": [((0, 70, 50), (14, 255, 255)), ((165, 70, 50), (180, 255, 255))],
    "yellow": [((16, 70, 60), (36, 255, 255))],
    "green": [((35, 50, 40), (85, 255, 255))],
    "blue": [((88, 50, 40), (135, 255, 255)), ((75, 50, 40), (88, 255, 255))],  # blue + cyan/teal
}
DRAW = {"red": (0, 0, 255), "yellow": (0, 255, 255), "green": (0, 255, 0), "blue": (255, 128, 0)}
PAD = 40      # look this many px around the grid box (pieces bigger than a cell spill out)
ZOOM = 6      # crop is enlarged this much so the patches are visible
OUT_DIR = os.path.expanduser("~/smolvla/logs/piece_sizes")


def grab_head_bgr(client):
    """Same path as the guarded client: frame -> JPEG -> what the detector decodes."""
    frame = wait_frame(client.get_head_frame)
    jpg = base64.b64decode(enc_jpeg_b64(frame))
    return cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)


def measure(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    min_u, min_v, max_u, max_v = TABLE_ROI
    rows = []
    for colour, ranges in COLOUR_RANGES.items():
        mask = None
        for lo, hi in ranges:
            m = cv2.inRange(hsv, lo, hi)
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < 15:
                continue
            (cx, cy), (w, h), angle = cv2.minAreaRect(c)
            if not (min_u - PAD <= cx <= max_u + PAD and min_v - PAD <= cy <= max_v + PAD):
                continue
            aspect = max(w, h) / max(1.0, min(w, h))
            cell = min(CELLS, key=lambda k: math.hypot(cx - CELLS[k][0], cy - CELLS[k][1]))
            dist = math.hypot(cx - CELLS[cell][0], cy - CELLS[cell][1])
            fails = []
            if area < MIN_AREA:
                fails.append("too_small")
            if area > MAX_AREA:
                fails.append("too_big")
            if aspect > MAX_ASPECT:
                fails.append("too_long")
            if not (min_u <= cx <= max_u and min_v <= cy <= max_v):
                fails.append("outside_grid_box")
            if dist > MAX_DIST_TO_CELL_PX:
                fails.append("off_cell_centre")
            rows.append({
                "colour": colour, "area_px": round(area, 1),
                "long_px": round(max(w, h), 1), "short_px": round(min(w, h), 1),
                "aspect": round(aspect, 2), "angle_deg": round(angle, 1),
                "cx": round(cx, 1), "cy": round(cy, 1),
                "nearest_cell": cell, "dist_px": round(dist, 1),
                "teacher_accepts": "YES" if not fails else "no",
                "why_not": " ".join(fails), "contour": c,
            })
    rows.sort(key=lambda r: r["area_px"], reverse=True)
    return rows


def save_crop(img, rows, path):
    out = img.copy()
    min_u, min_v, max_u, max_v = TABLE_ROI
    for r in rows:
        cv2.drawContours(out, [r["contour"]], -1, DRAW[r["colour"]], 1)
    cv2.rectangle(out, (min_u, min_v), (max_u, max_v), (0, 200, 0), 1)
    for k, (u, v) in CELLS.items():
        cv2.circle(out, (int(u), int(v)), 1, (0, 0, 255), -1)
    u0, v0 = max(0, min_u - PAD), max(0, min_v - PAD)
    crop = out[v0:max_v + PAD, u0:max_u + PAD]
    big = cv2.resize(crop, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)
    for k, (u, v) in CELLS.items():
        cv2.putText(big, k, (int((u - u0) * ZOOM) + 6, int((v - v0) * ZOOM) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite(path, big)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "sizes.csv")
    fields = ["time", "label", "colour", "area_px", "long_px", "short_px", "aspect", "angle_deg",
              "cx", "cy", "nearest_cell", "dist_px", "teacher_accepts", "why_not"]
    new_file = not os.path.exists(csv_path)
    client = ImageClient(host="127.0.0.1", auto_subscribe=False)
    client.start_subscriptions()
    print(f"Teacher's finder accepts: area {MIN_AREA:.0f}-{MAX_AREA:.0f} px, aspect <= {MAX_ASPECT}, "
          f"inside grid box {TABLE_ROI}, <= {MAX_DIST_TO_CELL_PX} px from a cell centre.")
    try:
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if new_file:
                writer.writeheader()
            while True:
                label = input("\nPlace ONE piece, type a label (e.g. yellow_2x4_R5), Enter (empty = quit): ").strip()
                if not label:
                    break
                time.sleep(0.5)  # let the hand leave the picture
                img = grab_head_bgr(client)
                rows = measure(img)
                save_crop(img, rows, os.path.join(OUT_DIR, f"{label}.jpg"))
                if not rows:
                    print("  no coloured patch near the grid (piece outside the camera's grid area?)")
                for r in rows[:6]:
                    print(f"  {r['colour']:6s} area={r['area_px']:6.1f}px  {r['long_px']:.0f}x{r['short_px']:.0f}px "
                          f"aspect={r['aspect']:.2f}  at ({r['cx']:.0f},{r['cy']:.0f}) near {r['nearest_cell']} "
                          f"({r['dist_px']:.1f}px)  teacher accepts: {r['teacher_accepts']} {r['why_not']}")
                stamp = time.strftime("%H:%M:%S")
                for r in rows:
                    writer.writerow({"time": stamp, "label": label, **{k: r[k] for k in fields[2:]}})
                f.flush()
                print(f"  saved {os.path.join(OUT_DIR, label + '.jpg')}")
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
