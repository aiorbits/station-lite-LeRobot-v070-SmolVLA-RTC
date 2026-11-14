#!/usr/bin/env python3
"""Isolated LeRobot v0.7.0 CUDA-only inference service for right7 SmolVLA.

Preserves the exact station_lite_smolvla_right7_server.py HTTP wire contract
while defaulting to localhost port 8011 and refusing production port 8010.
Never connects to robot hardware.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import importlib.metadata as metadata
import importlib.util
import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _require_torch():
    import torch
    return torch

SCHEMA = "station_lite_smolvla_right7_v1"
SOURCE_SCHEMA = "station_lite_smolvla_right7_request_v1"
TASK = "Pick up the red wooden block from the white grid and place it on the brown target area."
CAMERAS = {
    "observation.images.cam_high": "observation.images.camera1",
    "observation.images.cam_right_wrist": "observation.images.camera2",
}
IMAGE_SIZE = (640, 480)
MAX_REQUEST_BYTES = 64 * 1024 * 1024
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8011
RESERVED_PRODUCTION_PORT = 8010


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant not allowed: {value}")


def parse_json_object(raw_bytes: bytes) -> dict[str, Any]:
    """Decode strict UTF-8 JSON object rejecting NaN/Infinity literals."""
    text = raw_bytes.decode("utf-8")
    payload = json.loads(text, parse_constant=_strict_json_constant)
    if not isinstance(payload, dict):
        raise ValueError("request must be a JSON object")
    return payload


def validate_and_decode_payload(payload: dict[str, Any]) -> tuple[np.ndarray, dict[str, np.ndarray], str, int]:
    """Validate incoming right7 /predict JSON dict and decode camera images to uint8 RGB."""
    if payload.get("schema") != SOURCE_SCHEMA:
        raise ValueError(f"input schema must be {SOURCE_SCHEMA}")
    task = payload.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a non-empty string")
    seq = payload.get("seq")
    if type(seq) is not int or seq < 0 or seq > 1_000_000:
        raise ValueError("seq must be a nonnegative integer")

    raw_state = payload.get("state")
    if not isinstance(raw_state, (list, tuple)) or any(isinstance(v, bool) for v in raw_state):
        raise ValueError("right-arm state must be a 7-element numeric list")
    try:
        state = np.asarray(raw_state, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"right-arm state must contain 7 finite values: {exc}") from exc
    if state.shape != (7,) or not np.isfinite(state).all():
        raise ValueError("right-arm state must contain 7 finite values")

    encoded = payload.get("images_jpeg_base64")
    if not isinstance(encoded, dict):
        raise ValueError("images_jpeg_base64 must be an object")

    decoded_images: dict[str, np.ndarray] = {}
    bilinear = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
    for original, mapped in CAMERAS.items():
        value = encoded.get(original)
        if not isinstance(value, str) or not value:
            raise ValueError(f"missing image: {original}")
        try:
            raw = base64.b64decode(value, validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                image = image.convert("RGB")
                if image.width <= 0 or image.height <= 0:
                    raise ValueError(f"empty image for {original}")
                if image.width * IMAGE_SIZE[1] != image.height * IMAGE_SIZE[0]:
                    raise ValueError(f"wrong aspect ratio for {original}: {image.size}")
                if image.size != IMAGE_SIZE:
                    image = image.resize(IMAGE_SIZE, bilinear)
                rgb = np.asarray(image, dtype=np.uint8).copy()
        except (binascii.Error, OSError, ValueError) as exc:
            raise ValueError(f"invalid image for {original}: {exc}") from exc
        decoded_images[mapped] = rgb

    return state, decoded_images, task, seq


def _collect_processor_identity(checkpoint: Path, pre: Any, post: Any) -> dict[str, Any]:
    """Capture immutable runtime, processor step names, and checkpoint artifact hashes."""
    pkg_versions: dict[str, str | None] = {}
    for pkg in ("lerobot", "torch", "torchvision", "transformers", "accelerate", "huggingface-hub", "numpy"):
        try:
            pkg_versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pkg_versions[pkg] = None

    spec = importlib.util.find_spec("lerobot")
    lerobot_origin = str(Path(spec.origin).resolve()) if spec and spec.origin else None

    file_hashes: dict[str, str] = {}
    for name in (
        "config.json",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        "model.safetensors",
    ):
        candidate = checkpoint / name
        if candidate.is_file():
            file_hashes[name] = _sha256_file(candidate)

    torch = _require_torch()
    matmul_prec = (
        torch.get_float32_matmul_precision() if hasattr(torch, "get_float32_matmul_precision") else None
    )
    return {
        "sys_prefix": str(Path(sys.prefix).resolve()),
        "lerobot_origin": lerobot_origin,
        "packages": pkg_versions,
        "pre_steps": [type(step).__name__ for step in getattr(pre, "steps", [])],
        "post_steps": [type(step).__name__ for step in getattr(post, "steps", [])],
        "checkpoint_file_sha256": file_hashes,
        "precision": {
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": matmul_prec,
        },
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE", ""),
        "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE", ""),
    }


def verify_isolated_env(expected_prefix: str | None, expected_lerobot: str | None) -> None:
    """Refuse to serve if running outside the expected isolated v0.7.0 site-packages environment."""
    lerobot_ver = metadata.version("lerobot")
    if expected_lerobot and lerobot_ver != expected_lerobot:
        raise RuntimeError(f"lerobot version mismatch: actual={lerobot_ver}, expected={expected_lerobot}")
    if expected_prefix:
        exp_prefix = Path(expected_prefix).resolve()
        actual_prefix = Path(sys.prefix).resolve()
        if actual_prefix != exp_prefix:
            raise RuntimeError(f"sys.prefix mismatch: actual={actual_prefix}, expected={exp_prefix}")
        spec = importlib.util.find_spec("lerobot")
        origin = Path(spec.origin).resolve() if spec and spec.origin else None
        expected_pkg = exp_prefix / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/lerobot"
        if origin is None or not origin.is_relative_to(expected_pkg):
            raise RuntimeError(f"lerobot origin {origin} is outside expected {expected_pkg}")


class PolicyEngine:
    """CUDA-only right7 SmolVLA policy runner behind the HTTP compatibility contract."""

    def __init__(self, checkpoint: Path, seed: int = 42, seed_mode: str = "sequence"):
        torch = _require_torch()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required; refusing CPU inference")
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.checkpoint = checkpoint.resolve()
        self.device = "cuda"
        self.seed = seed
        if seed_mode not in ("sequence", "fixed"):
            raise ValueError("seed_mode must be sequence or fixed")
        self.seed_mode = seed_mode
        self.policy = SmolVLAPolicy.from_pretrained(str(self.checkpoint)).to("cuda").eval()
        self.pre, self.post = make_pre_post_processors(
            self.policy.config,
            pretrained_path=str(self.checkpoint),
            preprocessor_overrides={"device_processor": {"device": "cuda"}},
        )
        if self.policy.config.action_feature.shape != (7,) or self.policy.config.chunk_size != 50:
            raise ValueError("expected native right7 policy output shape (50, 7)")
        self.processor_identity = _collect_processor_identity(self.checkpoint, self.pre, self.post)
        self.lock = threading.Lock()

    def predict(self, payload: dict[str, Any]) -> tuple[np.ndarray, float, int]:
        torch = _require_torch()
        state, decoded_images, task, seq = validate_and_decode_payload(payload)
        observation: dict[str, Any] = {
            "observation.state": torch.from_numpy(state.copy()),
            "task": task,
        }
        for mapped, rgb in decoded_images.items():
            observation[mapped] = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        start = time.perf_counter()
        sampling_seed = self.seed if self.seed_mode == "fixed" else self.seed + seq
        with self.lock, torch.inference_mode():
            torch.manual_seed(sampling_seed)
            actions = self.post(self.policy.predict_action_chunk(self.pre(observation)))
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        result = actions[0].detach().float().cpu().numpy()
        if result.shape != (50, 7) or not np.isfinite(result).all():
            raise RuntimeError(f"invalid right7 model output: {result.shape}")
        return result, elapsed_ms, sampling_seed


def handler_for(engine: PolicyEngine, bound_port: int):
    """Build BaseHTTPRequestHandler bound to a PolicyEngine instance."""

    class Handler(BaseHTTPRequestHandler):
        def _read_exact_body(self, length: int) -> bytes:
            chunks: list[bytes] = []
            remaining = length
            while remaining > 0:
                chunk = self.rfile.read(remaining)
                if not chunk:
                    raise ValueError("incomplete request body")
                chunks.append(chunk)
                remaining -= len(chunk)
            return b"".join(chunks)

        def send_json(self, code: int, obj: dict[str, Any]) -> None:
            body = json.dumps(obj, allow_nan=False).encode("utf-8")
            try:
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                print(f"[disconnect] client closed connection during {code} response", flush=True)

        def do_GET(self) -> None:
            if self.path != "/health":
                self.send_json(404, {"error": "not found"})
                return
            self.send_json(200, {
                "schema": SCHEMA,
                "source_schema": SOURCE_SCHEMA,
                "checkpoint": str(engine.checkpoint),
                "chunk_size": 50,
                "state_dim": 7,
                "action_dim": 7,
                "state_contract": "right_joints_6_plus_gripper",
                "cameras": list(CAMERAS),
                "task": TASK,
                "device": "cuda",
                "gpu": _require_torch().cuda.get_device_name(0),
                "seed_base": engine.seed,
                "seed_mode": engine.seed_mode,
                "hardware_connected": False,
                "candidate_runtime": {
                    "server_role": "isolated_v070_g3",
                    "port": bound_port,
                    "lerobot_version": engine.processor_identity["packages"].get("lerobot"),
                    "processor_identity": engine.processor_identity,
                },
            })

        def do_POST(self) -> None:
            if self.path != "/predict":
                self.send_json(404, {"error": "not found"})
                return
            try:
                raw_len = self.headers.get("Content-Length")
                if raw_len is None:
                    raise ValueError("missing Content-Length header")
                length = int(raw_len)
                if not 0 < length <= MAX_REQUEST_BYTES:
                    raise ValueError("invalid request length")
                payload = parse_json_object(self._read_exact_body(length))
                actions, elapsed_ms, sampling_seed = engine.predict(payload)
                self.send_json(200, {
                    "schema": SCHEMA,
                    "source_schema": SOURCE_SCHEMA,
                    "session_id": payload.get("session_id"),
                    "seq": payload.get("seq"),
                    "actions": actions.tolist(),
                    "action_dim": 7,
                    "inference_ms": round(elapsed_ms, 3),
                    "sampling_seed": sampling_seed,
                    "hardware_connected": False,
                })
            except (ValueError, json.JSONDecodeError) as exc:
                self.send_json(400, {"error": str(exc)})
            except Exception as exc:
                print(f"[predict error] {type(exc).__name__}: {exc}", flush=True)
                self.send_json(500, {"error": f"inference failed: {type(exc).__name__}"})

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[{self.log_date_time_string()}] {fmt % args}", flush=True)

    return Handler


def parse_server_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments with hard protection against accidental production port 8010 use."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", default=DEFAULT_PORT, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--seed-mode", choices=("sequence", "fixed"), default="sequence")
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--verify-env", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expected-prefix", default=os.environ.get("EXPECTED_CONDA_PREFIX", ""))
    parser.add_argument("--expected-lerobot", default="0.7.0")
    args = parser.parse_args(argv)

    if args.port == RESERVED_PRODUCTION_PORT:
        parser.error(
            f"Refusing port {RESERVED_PRODUCTION_PORT}: reserved for production station_act server; "
            f"use port {DEFAULT_PORT} for isolated v0.7.0 G3 testing"
        )
    if not 1 <= args.port <= 65535:
        parser.error(f"invalid TCP port: {args.port}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_server_args(argv)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.verify_env:
        verify_isolated_env(args.expected_prefix, args.expected_lerobot)
    if not args.checkpoint.is_dir():
        raise SystemExit("checkpoint directory does not exist")
    torch = _require_torch()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required; no CPU fallback is available")

    engine = PolicyEngine(args.checkpoint, args.seed, args.seed_mode)
    server = ThreadingHTTPServer((args.host, args.port), handler_for(engine, args.port))
    print(
        f"[ready] {SCHEMA} checkpoint={engine.checkpoint} lerobot={engine.processor_identity['packages'].get('lerobot')} "
        f"seed_mode={engine.seed_mode} device=cuda gpu={torch.cuda.get_device_name(0)} "
        f"bind={args.host}:{args.port} hardware_connected=false",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

