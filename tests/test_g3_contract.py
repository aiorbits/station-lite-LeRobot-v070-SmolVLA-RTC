#!/usr/bin/env python3
"""Offline unit tests for G3 isolated v0.7.0 HTTP server validation & client response guards."""
from __future__ import annotations

import base64
import io
import json
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from g3_http_compatibility_gate import (  # noqa: E402
    validate_right7_health,
    validate_right7_predict_response,
)
from station_lite_smolvla_right7_server import (  # noqa: E402
    SOURCE_SCHEMA,
    parse_json_object,
    parse_server_args,
    validate_and_decode_payload,
)


def _jpeg_b64(size_wh: tuple[int, int]) -> str:
    img = Image.new("RGB", size_wh, color=(64, 128, 192))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class TestG3Contract(unittest.TestCase):
    def _valid_payload(self) -> dict:
        cam = _jpeg_b64((640, 480))
        return {
            "schema": SOURCE_SCHEMA,
            "task": "Pick up the yellow wooden block from the white grid and place it on the brown target area.",
            "session_id": "unit-sess",
            "seq": 0,
            "state": [0.0, 0.1, -0.1, 0.2, 0.0, -0.2, 0.068],
            "images_jpeg_base64": {
                "observation.images.cam_high": cam,
                "observation.images.cam_right_wrist": cam,
            },
        }

    def test_port_8010_refused_and_default_is_8011(self) -> None:
        args = parse_server_args(["--checkpoint", "/tmp/ckpt"])
        self.assertEqual(args.port, 8011)
        self.assertEqual(args.host, "127.0.0.1")
        with self.assertRaises(SystemExit):
            parse_server_args(["--checkpoint", "/tmp/ckpt", "--port", "8010"])

    def test_strict_json_and_payload_validation(self) -> None:
        with self.assertRaises(ValueError):
            parse_json_object(b'{"state": [NaN, 0, 0, 0, 0, 0, 0]}')
        with self.assertRaises(ValueError):
            parse_json_object(b"[1, 2, 3]")

        ok = self._valid_payload()
        state, imgs, task, seq = validate_and_decode_payload(ok)
        self.assertEqual(state.shape, (7,))
        self.assertEqual(imgs["observation.images.camera1"].shape, (480, 640, 3))
        self.assertEqual(imgs["observation.images.camera2"].shape, (480, 640, 3))
        self.assertEqual(seq, 0)
        self.assertTrue(task.startswith("Pick up"))

        for bad in (
            {**ok, "schema": "wrong"},
            {**ok, "task": "  "},
            {**ok, "seq": True},
            {**ok, "seq": -1},
            {**ok, "state": [0.0] * 6},
            {**ok, "state": {"bad": 1}},
            {**ok, "state": [True] * 7},
            {**ok, "state": [float("nan")] + [0.0] * 6},
            {**ok, "images_jpeg_base64": {"observation.images.cam_high": _jpeg_b64((640, 480))}},
            {
                **ok,
                "images_jpeg_base64": {
                    "observation.images.cam_high": _jpeg_b64((100, 100)),
                    "observation.images.cam_right_wrist": _jpeg_b64((640, 480)),
                },
            },
        ):
            with self.assertRaises(ValueError):
                validate_and_decode_payload(bad)

        # Valid 4:3 image at 320x240 resizes cleanly to 640x480
        resized = {
            **ok,
            "images_jpeg_base64": {
                "observation.images.cam_high": _jpeg_b64((320, 240)),
                "observation.images.cam_right_wrist": _jpeg_b64((320, 240)),
            },
        }
        _, r_imgs, _, _ = validate_and_decode_payload(resized)
        self.assertEqual(r_imgs["observation.images.camera1"].shape, (480, 640, 3))

    def test_health_and_stale_response_guards(self) -> None:
        health = {
            "schema": "station_lite_smolvla_right7_v1",
            "source_schema": SOURCE_SCHEMA,
            "checkpoint": "/tmp/ckpt",
            "chunk_size": 50,
            "state_dim": 7,
            "action_dim": 7,
            "task": "Pick up block",
            "device": "cuda",
            "seed_mode": "sequence",
            "hardware_connected": False,
            "candidate_runtime": {"lerobot_version": "0.7.0"},
        }
        validate_right7_health(health, "/tmp/ckpt", expected_seed_mode="sequence", expected_lerobot="0.7.0")
        with self.assertRaises(ValueError):
            validate_right7_health({**health, "hardware_connected": True}, "/tmp/ckpt")

        valid_resp = {
            "schema": "station_lite_smolvla_right7_v1",
            "source_schema": SOURCE_SCHEMA,
            "session_id": "s1",
            "seq": 2,
            "actions": np.zeros((50, 7), dtype=np.float32).tolist(),
            "action_dim": 7,
            "sampling_seed": 44,
            "hardware_connected": False,
        }
        arr = validate_right7_predict_response(
            valid_resp,
            expected_session_id="s1",
            expected_seq=2,
            expected_seed_mode="sequence",
            expected_seed_base=42,
        )
        self.assertEqual(arr.shape, (50, 7))
        for bad_resp in (
            {**valid_resp, "seq": 1},
            {**valid_resp, "session_id": "stale"},
            {**valid_resp, "sampling_seed": 42},
            {**valid_resp, "hardware_connected": True},
            {**valid_resp, "actions": np.zeros((49, 7)).tolist()},
        ):
            with self.assertRaises(ValueError):
                validate_right7_predict_response(
                    bad_resp,
                    expected_session_id="s1",
                    expected_seq=2,
                    expected_seed_mode="sequence",
                    expected_seed_base=42,
                )


if __name__ == "__main__":
    unittest.main()

